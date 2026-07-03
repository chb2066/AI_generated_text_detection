import os
import re
import argparse
import warnings
import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import logging

# DDP 관련 import
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")


def setup_logging(rank):
    """로깅 설정 (rank 0에서만 출력)"""
    if rank == 0:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)
    return logging.getLogger(__name__)


@dataclass
class ImprovedConfig:
    # Model settings
    model_name: str = "klue/bert-base"
    max_paragraph_length: int = 500  # 문장 -> 문단으로 변경
    max_doc_length: int = 2000
    hidden_size: int = 768

    # Training settings - 차별적 학습률 (Gradient Accumulation 최적화)
    batch_size: int = 2              # 3 → 2 (메모리 절약)
    gradient_accumulation_steps: int = 8  # 4 → 8 (안정성 향상, effective batch: 2×8=16)
    backbone_learning_rate: float = 1e-6  # 백본용 낮은 학습률
    head_learning_rate: float = 1e-5      # 새 레이어용 높은 학습률
    weight_decay: float = 0.01
    num_epochs: int = 3  # 에포크 증가
    warmup_ratio: float = 0.1

    # EPA-MIL settings - 개선된 파라미터
    lambda_paragraph: float = 0.5  # 문장 -> 문단으로 변경
    pos_weight: float = 15.0  # pos_weight 증가
    focal_gamma: float = 3.0  # 더 강한 focusing
    focal_alpha: float = 0.75  # 클래스 밸런싱
    label_smoothing: float = 0.1  # 과적합 방지
    top_k_ratio: float = 0.15  # top-k 비율 증가

    # Temperature Scaling settings
    temperature: float = 1.5  # 초기 temperature
    calibration_method: str = "temperature"  # "temperature", "platt", "isotonic"

    # AUC Loss settings - 새로 추가
    auc_weight: float = 0.3  # AUC loss 비중
    auc_gamma: float = 1.0   # Differentiable AUC loss gamma

    # Regularization settings
    dropout_rate: float = 0.3
    layer_norm_eps: float = 1e-12
    gradient_clip_norm: float = 1.0

    # Data settings
    train_val_split: float = 0.9
    cv_folds: int = 5
    filter_long_documents: bool = True

    # System settings
    num_workers: int = 2
    seed: int = 42


class TemperatureScaling(nn.Module):
    """Temperature Scaling for calibration - 디바이스 문제 해결"""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)

    def forward(self, logits):
        """Apply temperature scaling to logits"""
        # 같은 디바이스로 temperature 이동
        device = logits.device
        if self.temperature.device != device:
            self.temperature.data = self.temperature.data.to(device)

        return logits / self.temperature

    def calibrate(self, val_loader, model, device, config):
        """Find optimal temperature using validation data"""
        model.eval()
        logits_list = []
        labels_list = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                num_paragraphs = batch['num_paragraphs'].to(device)

                # Temperature scaling 없이 로짓 추출
                outputs = model(input_ids, attention_mask, num_paragraphs, apply_temperature=False)
                doc_logits = outputs['document_logits']
                if doc_logits.dim() > 1:
                    doc_logits = doc_logits.squeeze(-1)

                logits_list.append(doc_logits.cpu())
                labels_list.append(labels.cpu())

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        # Grid search for optimal temperature (개선된 범위)
        best_temp = 1.0
        best_loss = float('inf')

        for temp in np.arange(0.9, 1.8, 0.05):  # 0.5-3.0, 0.1 → 0.9-1.8, 0.05 (더 세밀하고 보수적)
            scaled_logits = logits / temp
            loss = F.binary_cross_entropy_with_logits(scaled_logits, labels)

            if loss < best_loss:
                best_loss = loss
                best_temp = temp

        # 디바이스에 맞춰 temperature 업데이트
        self.temperature.data = torch.tensor([best_temp], device=device)
        return best_temp


class DifferentiableAUCLoss(nn.Module):
    """
    Differentiable AUC Loss using sigmoid approximation
    미분 가능한 AUC 근사 손실
    """

    def __init__(self, gamma=1.0):
        super(DifferentiableAUCLoss, self).__init__()
        self.gamma = gamma

    def forward(self, y_pred, y_true):
        if y_pred.dim() > 1:
            y_pred = y_pred.squeeze(-1)

        pos_mask = (y_true == 1)
        neg_mask = (y_true == 0)

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return F.binary_cross_entropy_with_logits(y_pred, y_true)

        pos_pred = y_pred[pos_mask]
        neg_pred = y_pred[neg_mask]

        # 모든 positive-negative 쌍에 대해 sigmoid 근사
        pos_pred_expanded = pos_pred.unsqueeze(1)  # (num_pos, 1)
        neg_pred_expanded = neg_pred.unsqueeze(0)  # (1, num_neg)

        # Sigmoid approximation of step function
        diff = pos_pred_expanded - neg_pred_expanded
        approx_auc = torch.sigmoid(self.gamma * diff).mean()

        # AUC는 최대화해야 하므로 negative loss
        return 1.0 - approx_auc


class FocalAUCLoss(nn.Module):
    """
    Focal Loss + AUC Loss 결합
    """

    def __init__(self, gamma=2.0, alpha=0.75, label_smoothing=0.1,
                 auc_weight=0.3, auc_gamma=1.0):
        super(FocalAUCLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.auc_weight = auc_weight

        # AUC Loss
        self.auc_loss = DifferentiableAUCLoss(gamma=auc_gamma)

    def forward(self, input, target):
        # Label smoothing
        if self.label_smoothing > 0:
            target_smooth = target * (1 - self.label_smoothing) + self.label_smoothing * 0.5
        else:
            target_smooth = target

        # Focal Loss 계산
        ce_loss = F.binary_cross_entropy_with_logits(input, target_smooth, reduction='none')
        p_t = torch.exp(-ce_loss)
        alpha_t = self.alpha * target_smooth + (1 - self.alpha) * (1 - target_smooth)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = (focal_weight * ce_loss).mean()

        # AUC Loss 계산
        auc_loss = self.auc_loss(input, target)

        # 결합
        total_loss = (1 - self.auc_weight) * focal_loss + self.auc_weight * auc_loss

        return total_loss


class ImprovedFocalLoss(nn.Module):
    """개선된 Focal Loss with label smoothing"""

    def __init__(self, gamma=3.0, alpha=0.75, label_smoothing=0.1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, input, target):
        # Label smoothing 적용
        if self.label_smoothing > 0:
            target = target * (1 - self.label_smoothing) + self.label_smoothing * 0.5

        # BCE loss with logits
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        p_t = torch.exp(-ce_loss)

        # Alpha balancing
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)

        # Focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class TextPreprocessor:
    """텍스트 전처리 및 문단 분할"""

    def __init__(self):
        pass

    def preprocess_text(self, text: str) -> str:
        """사전처리: 연속된 줄바꿈 정리"""
        # 연속된 줄바꿈을 하나의 줄바꿈으로
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()

    def split_paragraphs(self, text: str) -> List[str]:
        """문단 분할: \n 기준으로 분할"""
        text = self.preprocess_text(text)

        # \n으로 분할
        paragraphs = text.split('\n')

        # 빈 문단 제거 및 공백 정리
        paragraphs = [para.strip() for para in paragraphs if para.strip()]

        return paragraphs


def filter_documents_by_length(data: pd.DataFrame, max_doc_length: int, is_train: bool = True, rank: int = 0) -> Tuple[pd.DataFrame, Dict]:
    """문서 길이 기준으로 데이터 필터링"""
    preprocessor = TextPreprocessor()
    original_len = len(data)
    filtered_indices = []

    if rank == 0:
        print(f"📏 Filtering documents by max length: {max_doc_length}")

    iterator = data.iterrows()
    if rank == 0 and len(data) > 1000:
        iterator = tqdm(iterator, total=len(data), desc="Filtering documents")

    for idx, row in iterator:
        if is_train:
            text = row['full_text']
        else:
            text = row['paragraph_text']

        paragraphs = preprocessor.split_paragraphs(text)
        total_length = sum(len(para) for para in paragraphs)

        if total_length <= max_doc_length:
            filtered_indices.append(idx)

    filtered_data = data.iloc[filtered_indices].reset_index(drop=True)

    filter_stats = {
        'original_count': original_len,
        'filtered_count': len(filtered_data),
        'removed_count': original_len - len(filtered_data),
        'removal_rate': (original_len - len(filtered_data)) / original_len if original_len > 0 else 0
    }

    if rank == 0:
        print(f"📊 Document filtering results:")
        print(f"   Original documents: {filter_stats['original_count']:,}")
        print(f"   Kept documents: {filter_stats['filtered_count']:,}")
        print(f"   Removed documents: {filter_stats['removed_count']:,}")
        print(f"   Removal rate: {filter_stats['removal_rate']:.2%}")

        if is_train and len(filtered_data) > 0:
            label_dist = filtered_data['generated'].value_counts()
            print(f"   Label distribution after filtering:")
            print(f"     Generated=0: {label_dist.get(0, 0):,}")
            print(f"     Generated=1: {label_dist.get(1, 0):,}")

    return filtered_data, filter_stats


class EPAMILDataset(Dataset):
    """EPA-MIL을 위한 데이터셋 - 문단 기반"""

    def __init__(self,
                 data: pd.DataFrame,
                 tokenizer: AutoTokenizer,
                 config: ImprovedConfig,
                 is_train: bool = True,
                 rank: int = 0):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.is_train = is_train
        self.rank = rank
        self.preprocessor = TextPreprocessor()

        self.processed_data = self._process_data()

    def _process_data(self) -> List[Dict]:
        """데이터 전처리 및 문단 분할"""
        processed = []

        iterator = self.data.iterrows()
        if self.rank == 0 and len(self.data) > 1000:
            iterator = tqdm(iterator, total=len(self.data), desc="Processing filtered data (paragraph-based)")

        for idx, row in iterator:
            if self.is_train:
                text = row['full_text']
                label = row['generated']
                doc_id = f"train_{idx}"
            else:
                text = row['paragraph_text']
                label = None
                doc_id = row['ID']

            paragraphs = self.preprocessor.split_paragraphs(text)

            if not paragraphs:
                continue

            tokenized_paragraphs = []
            for para in paragraphs:
                tokens = self.tokenizer(
                    para,
                    truncation=True,
                    padding=False,
                    max_length=self.config.max_paragraph_length,
                    return_tensors=None
                )
                tokenized_paragraphs.append({
                    'input_ids': tokens['input_ids'],
                    'attention_mask': tokens['attention_mask']
                })

            processed.append({
                'doc_id': doc_id,
                'title': row['title'],
                'paragraphs': tokenized_paragraphs,
                'label': label,
                'num_paragraphs': len(tokenized_paragraphs),
                'total_length': sum(len(para) for para in paragraphs)
            })

        return processed

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]

        paragraphs = item['paragraphs']
        max_para_in_doc = min(len(paragraphs), self.config.max_doc_length // self.config.max_paragraph_length)

        if len(paragraphs) > max_para_in_doc:
            paragraphs = paragraphs[:max_para_in_doc]

        if not paragraphs:
            paragraphs = [{'input_ids': [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id],
                         'attention_mask': [1, 1]}]

        max_len = max(len(p['input_ids']) for p in paragraphs)
        max_len = min(max_len, self.config.max_paragraph_length)

        padded_input_ids = []
        padded_attention_mask = []

        for para in paragraphs:
            input_ids = para['input_ids'][:max_len]
            attention_mask = para['attention_mask'][:max_len]

            padding_length = max_len - len(input_ids)
            input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)

            padded_input_ids.append(input_ids)
            padded_attention_mask.append(attention_mask)

        return {
            'doc_id': item['doc_id'],
            'input_ids': torch.tensor(padded_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(padded_attention_mask, dtype=torch.long),
            'label': torch.tensor(item['label'], dtype=torch.float) if item['label'] is not None else None,
            'num_paragraphs': len(paragraphs)
        }


def collate_fn(batch):
    """커스텀 collate function - 문단 기반"""
    doc_ids = [item['doc_id'] for item in batch]
    labels = [item['label'] for item in batch if item['label'] is not None]
    num_paragraphs = [item['num_paragraphs'] for item in batch]

    max_paras = max(num_paragraphs)
    max_para_len = max(item['input_ids'].shape[1] for item in batch)

    batch_input_ids = []
    batch_attention_mask = []

    for item in batch:
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']

        current_para_len = input_ids.shape[1]
        if current_para_len < max_para_len:
            len_padding = max_para_len - current_para_len
            len_pad_ids = torch.zeros(input_ids.shape[0], len_padding, dtype=torch.long)
            len_pad_mask = torch.zeros(attention_mask.shape[0], len_padding, dtype=torch.long)

            input_ids = torch.cat([input_ids, len_pad_ids], dim=1)
            attention_mask = torch.cat([attention_mask, len_pad_mask], dim=1)

        current_num_paras = input_ids.shape[0]
        if current_num_paras < max_paras:
            para_padding = max_paras - current_num_paras
            para_pad_ids = torch.zeros(para_padding, max_para_len, dtype=torch.long)
            para_pad_mask = torch.zeros(para_padding, max_para_len, dtype=torch.long)

            input_ids = torch.cat([input_ids, para_pad_ids], dim=0)
            attention_mask = torch.cat([attention_mask, para_pad_mask], dim=0)

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)

    return {
        'doc_ids': doc_ids,
        'input_ids': torch.stack(batch_input_ids),
        'attention_mask': torch.stack(batch_attention_mask),
        'labels': torch.stack(labels) if labels else None,
        'num_paragraphs': torch.tensor(num_paragraphs, dtype=torch.long)
    }


class ImprovedEPAMILModel(nn.Module):
    """개선된 EPA-MIL 모델 - 문단 기반 (디바이스 문제 해결)"""

    def __init__(self, config: ImprovedConfig):
        super().__init__()
        self.config = config

        # 백본 모델
        self.backbone = AutoModel.from_pretrained(config.model_name)
        self.hidden_size = self.backbone.config.hidden_size

        # 개선된 문단 레벨 분류기 (멀티레이어)
        self.paragraph_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2, eps=config.layer_norm_eps),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.LayerNorm(self.hidden_size // 4, eps=config.layer_norm_eps),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_size // 4, 1)
        )

        # 개선된 어텐션 메커니즘 (768은 8로 나누어떨어짐)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,  # 768 / 8 = 96 (정확히 나누어떨어짐)
            dropout=config.dropout_rate,
            batch_first=True
        )

        # 개선된 문서 레벨 분류기 (멀티레이어)
        self.document_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2, eps=config.layer_norm_eps),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 4),
            nn.LayerNorm(self.hidden_size // 4, eps=config.layer_norm_eps),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_size // 4, 1)
        )

        # Temperature Scaling
        self.temperature_scaler = TemperatureScaling(config.temperature)

        # 초기화
        self._init_weights()

    def _init_weights(self):
        """개선된 가중치 초기화"""
        for module in [self.paragraph_classifier, self.document_classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def to(self, device):
        """디바이스 이동시 temperature도 함께 이동"""
        result = super().to(device)
        # Temperature scaler의 파라미터도 명시적으로 이동
        if hasattr(self, 'temperature_scaler'):
            self.temperature_scaler.temperature.data = self.temperature_scaler.temperature.data.to(device)
        return result

    def forward(self, input_ids, attention_mask, num_paragraphs, apply_temperature=False):
        batch_size, max_paras, max_len = input_ids.shape
        device = input_ids.device

        # 문단들을 배치로 변환
        input_ids_flat = input_ids.view(-1, max_len)
        attention_mask_flat = attention_mask.view(-1, max_len)

        # 유효한 문단만 선택
        valid_mask = attention_mask_flat.sum(dim=1) > 0

        if valid_mask.sum() == 0:
            return {
                'document_logits': torch.zeros(batch_size, device=device),
                'paragraph_logits': torch.zeros(batch_size, max_paras, device=device),
            }

        # 백본 인코딩
        valid_input_ids = input_ids_flat[valid_mask]
        valid_attention_mask = attention_mask_flat[valid_mask]

        outputs = self.backbone(
            input_ids=valid_input_ids,
            attention_mask=valid_attention_mask
        )

        # [CLS] 토큰 임베딩 추출
        paragraph_embeddings = outputs.last_hidden_state[:, 0]

        # 문단 레벨 로짓
        paragraph_logits_valid = self.paragraph_classifier(paragraph_embeddings).squeeze(-1)

        # 원래 shape으로 복원
        paragraph_logits_full = torch.zeros(batch_size * max_paras, device=device, dtype=paragraph_logits_valid.dtype)
        paragraph_logits_full[valid_mask] = paragraph_logits_valid
        paragraph_logits = paragraph_logits_full.view(batch_size, max_paras)

        paragraph_embeddings_full = torch.zeros(batch_size * max_paras, self.hidden_size, device=device, dtype=paragraph_embeddings.dtype)
        paragraph_embeddings_full[valid_mask] = paragraph_embeddings
        paragraph_embeddings = paragraph_embeddings_full.view(batch_size, max_paras, self.hidden_size)

        # 문서 레벨 어그리게이션
        document_logits = []
        for i in range(batch_size):
            num_para = num_paragraphs[i]
            if num_para == 0:
                document_logits.append(torch.tensor(0.0, device=device, dtype=paragraph_logits.dtype))
                continue

            # 유효한 문단들만 선택
            valid_para_embeds = paragraph_embeddings[i, :num_para]
            valid_para_logits = paragraph_logits[i, :num_para]

            # Top-k 선택 (비율 증가)
            k = max(1, int(self.config.top_k_ratio * num_para))
            top_k_indices = torch.topk(valid_para_logits, k=k, dim=0).indices

            # 어텐션 기반 어그리게이션
            selected_embeds = valid_para_embeds[top_k_indices].unsqueeze(0)
            attended_embeds, attention_weights = self.attention(
                selected_embeds, selected_embeds, selected_embeds
            )

            # 가중 평균 풀링
            doc_embed = attended_embeds.mean(dim=1)
            doc_logit = self.document_classifier(doc_embed).squeeze(-1)
            document_logits.append(doc_logit)

        document_logits = torch.stack(document_logits)

        # Temperature scaling 적용 (필요시)
        if apply_temperature:
            # Temperature scaler의 디바이스 확인 및 이동
            if self.temperature_scaler.temperature.device != device:
                self.temperature_scaler.temperature.data = self.temperature_scaler.temperature.data.to(device)
            document_logits = self.temperature_scaler(document_logits)

        return {
            'document_logits': document_logits,
            'paragraph_logits': paragraph_logits,
        }


class ImprovedEPAMILTrainer:
    """개선된 EPA-MIL DDP 학습 클래스 - 문단 기반"""

    def __init__(self, config: ImprovedConfig, rank: int, world_size: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        self.logger = setup_logging(rank)

        # 토크나이저 및 모델 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = ImprovedEPAMILModel(config).to(self.device)

        # DDP 설정
        self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=True)

        # AUC 기반 손실 함수로 변경
        self.doc_criterion = FocalAUCLoss(
            gamma=config.focal_gamma,      # Focal loss gamma
            alpha=config.focal_alpha,      # Focal loss alpha
            auc_weight=config.auc_weight,  # AUC loss 비중 (30%)
            auc_gamma=config.auc_gamma     # AUC loss gamma
        )

        self.para_criterion = ImprovedFocalLoss(
            gamma=config.focal_gamma,
            alpha=config.focal_alpha,
            label_smoothing=config.label_smoothing
        )

        # Mixed Precision 설정
        self.scaler = GradScaler()

        self.optimizer = None
        self.scheduler = None

    def _setup_optimizers(self, train_loader):
        """차별적 학습률을 위한 옵티마이저 설정 (개선된 스케줄러)"""
        # 백본과 헤드 파라미터 분리
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        # 차별적 학습률 적용
        self.optimizer = AdamW([
            {'params': backbone_params, 'lr': self.config.backbone_learning_rate, 'weight_decay': self.config.weight_decay},
            {'params': head_params, 'lr': self.config.head_learning_rate, 'weight_decay': self.config.weight_decay}
        ], eps=1e-8)

        # Cosine Annealing with Warm Restarts (개선된 스케줄러)
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=len(train_loader),  # 한 에포크마다 restart
            T_mult=1,               # restart 주기 유지
            eta_min=1e-8           # 최소 학습률
        )

    def train_epoch(self, train_loader, epoch):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        doc_losses = []
        para_losses = []

        train_loader.sampler.set_epoch(epoch)

        if self.rank == 0:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        else:
            progress_bar = train_loader

        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            num_paragraphs = batch['num_paragraphs'].to(self.device, non_blocking=True)

            # Mixed Precision Forward pass
            with autocast():
                outputs = self.model(input_ids, attention_mask, num_paragraphs)

                # 문서 레벨 손실 (AUC 기반)
                doc_logits = outputs['document_logits']
                if doc_logits.dim() > 1:
                    doc_logits = doc_logits.squeeze(-1)
                doc_loss = self.doc_criterion(doc_logits, labels)

                # 문단 레벨 손실 (bag labeling)
                para_logits = outputs['paragraph_logits']
                para_labels = labels.unsqueeze(1).expand_as(para_logits)

                # 유효한 문단들에 대해서만 손실 계산
                valid_para_mask = torch.zeros_like(para_logits, dtype=torch.bool)
                for i, num_para in enumerate(num_paragraphs):
                    if num_para > 0:
                        valid_para_mask[i, :num_para] = True

                if valid_para_mask.sum() > 0:
                    para_loss = self.para_criterion(
                        para_logits[valid_para_mask].float(),
                        para_labels[valid_para_mask].float()
                    )
                else:
                    para_loss = torch.tensor(0.0, device=self.device)

                # 전체 손실
                total_batch_loss = doc_loss + self.config.lambda_paragraph * para_loss

            # Backward pass
            self.scaler.scale(total_batch_loss).backward()

            # 그래디언트 축적
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # 그래디언트 클리핑
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.gradient_clip_norm)

                # 옵티마이저 스텝
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            # 손실 기록
            total_loss += total_batch_loss.item()
            doc_losses.append(doc_loss.item())
            para_losses.append(para_loss.item())

            # 프로그레스 바 업데이트
            if self.rank == 0:
                progress_bar.set_postfix({
                    'Loss': f"{total_batch_loss.item():.4f}",
                    'Doc(AUC)': f"{doc_loss.item():.4f}",
                    'Para': f"{para_loss.item():.4f}",
                    'LR': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })

        return {
            'total_loss': total_loss / len(train_loader),
            'doc_loss': np.mean(doc_losses),
            'para_loss': np.mean(para_losses)
        }

    def _calculate_calibration_error(self, preds, labels, n_bins=10):
        """Expected Calibration Error (ECE) 계산 - 수정된 버전"""
        # 입력을 NumPy 배열로 변환
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        # 빈 배열 체크
        if len(preds) == 0 or len(labels) == 0:
            return 0.0

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # 각 bin에 속하는 예측들
            in_bin = (preds > bin_lower) & (preds <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                # Boolean 인덱싱으로 해당 bin의 데이터 추출
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = preds[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def validate(self, val_loader):
        """검증 - 수정된 버전"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_logits = []

        with torch.no_grad():
            iterator = tqdm(val_loader, desc="Validation") if self.rank == 0 else val_loader

            for batch in iterator:
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['labels'].to(self.device, non_blocking=True)
                num_paragraphs = batch['num_paragraphs'].to(self.device, non_blocking=True)

                with autocast():
                    outputs = self.model(input_ids, attention_mask, num_paragraphs, apply_temperature=True)
                    doc_logits = outputs['document_logits']
                    if doc_logits.dim() > 1:
                        doc_logits = doc_logits.squeeze(-1)
                    preds = torch.sigmoid(doc_logits).cpu().numpy()
                    logits = doc_logits.cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(logits)

        # DDP에서 모든 프로세스의 결과 수집
        if dist.is_initialized():
            try:
                # 리스트를 텐서로 변환
                preds_tensor = torch.tensor(all_preds, dtype=torch.float32, device=self.device)
                labels_tensor = torch.tensor(all_labels, dtype=torch.float32, device=self.device)
                logits_tensor = torch.tensor(all_logits, dtype=torch.float32, device=self.device)

                # 모든 프로세스의 크기 수집
                preds_size = torch.tensor([preds_tensor.shape[0]], device=self.device)
                all_sizes = [torch.zeros_like(preds_size) for _ in range(self.world_size)]
                dist.all_gather(all_sizes, preds_size)

                # 최대 크기로 패딩
                max_size = max(size.item() for size in all_sizes)
                if preds_tensor.shape[0] < max_size:
                    padding_size = max_size - preds_tensor.shape[0]
                    padding = torch.zeros(padding_size, dtype=torch.float32, device=self.device)
                    preds_tensor = torch.cat([preds_tensor, padding])
                    labels_tensor = torch.cat([labels_tensor, padding])
                    logits_tensor = torch.cat([logits_tensor, padding])

                # gather
                gathered_preds = [torch.zeros_like(preds_tensor) for _ in range(self.world_size)]
                gathered_labels = [torch.zeros_like(labels_tensor) for _ in range(self.world_size)]
                gathered_logits = [torch.zeros_like(logits_tensor) for _ in range(self.world_size)]

                dist.all_gather(gathered_preds, preds_tensor)
                dist.all_gather(gathered_labels, labels_tensor)
                dist.all_gather(gathered_logits, logits_tensor)

                # 유효한 데이터만 추출하여 NumPy 배열로 변환
                all_preds = []
                all_labels = []
                all_logits = []
                for i, size in enumerate(all_sizes):
                    valid_size = size.item()
                    if valid_size > 0:
                        all_preds.extend(gathered_preds[i][:valid_size].cpu().numpy())
                        all_labels.extend(gathered_labels[i][:valid_size].cpu().numpy())
                        all_logits.extend(gathered_logits[i][:valid_size].cpu().numpy())

            except Exception as e:
                if self.rank == 0:
                    print(f"Warning: DDP gathering failed, using local results: {e}")
                # DDP 실패시 로컬 결과 사용
                pass

        # 최종적으로 NumPy 배열로 변환
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)

        # 빈 배열 체크
        if len(all_preds) == 0 or len(all_labels) == 0:
            if self.rank == 0:
                print("Warning: No valid predictions for evaluation")
            return {
                'auc': 0.0,
                'calibration_error': 0.0,
                'logits': [],
                'labels': []
            }

        try:
            auc = roc_auc_score(all_labels, all_preds)
        except Exception as e:
            if self.rank == 0:
                print(f"Warning: AUC calculation failed: {e}")
            auc = 0.0

        # Calibration 메트릭 계산
        try:
            calibration_error = self._calculate_calibration_error(all_preds, all_labels)
        except Exception as e:
            if self.rank == 0:
                print(f"Warning: Calibration error calculation failed: {e}")
            calibration_error = 0.0

        return {
            'auc': auc,
            'calibration_error': calibration_error,
            'logits': all_logits.tolist(),
            'labels': all_labels.tolist()
        }

    def calibrate_model(self, val_loader):
        """모델 보정 (Temperature Scaling) - 수정된 버전"""
        if self.rank == 0:
            print("🎯 Calibrating model with Temperature Scaling...")

        optimal_temp = None
        model_module = self.model.module if hasattr(self.model, 'module') else self.model

        # Temperature scaling 수행
        if self.config.calibration_method == "temperature":
            optimal_temp = model_module.temperature_scaler.calibrate(
                val_loader, model_module, self.device, self.config
            )
            if self.rank == 0:
                print(f"   Optimal temperature: {optimal_temp:.3f}")

        # 모든 프로세스에 temperature 동기화
        if dist.is_initialized():
            # rank 0의 temperature를 모든 프로세스에 브로드캐스트
            if self.rank == 0:
                temp_tensor = model_module.temperature_scaler.temperature.data.clone()
            else:
                temp_tensor = torch.zeros(1, device=self.device)

            dist.broadcast(temp_tensor, src=0)

            # 모든 프로세스에서 temperature 업데이트
            model_module.temperature_scaler.temperature.data = temp_tensor

        return optimal_temp if self.rank == 0 else None

    def train(self, train_data, val_data):
        """전체 학습 과정"""
        # 학습 데이터 필터링 비활성화
        filtered_train_data = train_data  # 필터링 없이 그대로 사용
        filtered_val_data = val_data      # 필터링 없이 그대로 사용
        if self.rank == 0:
            print("📄 Document length filtering is disabled - using all documents")

        # 필터링 후 데이터 확인
        if self.rank == 0:
            print(f"\n📋 Training data (no filtering applied):")
            print(f"   Train samples: {len(filtered_train_data):,}")
            print(f"   Validation samples: {len(filtered_val_data):,}")

            if len(filtered_train_data) == 0:
                raise ValueError("❌ No training data available!")
            if len(filtered_val_data) == 0:
                raise ValueError("❌ No validation data available!")

        # 데이터셋 생성
        train_dataset = EPAMILDataset(filtered_train_data, self.tokenizer, self.config, is_train=True, rank=self.rank)
        val_dataset = EPAMILDataset(filtered_val_data, self.tokenizer, self.config, is_train=True, rank=self.rank)

        # DistributedSampler 사용
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            drop_last=True
        )

        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            drop_last=False
        )

        # 데이터 로더 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=1
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            sampler=val_sampler,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=1
        )

        # 옵티마이저 설정
        self._setup_optimizers(train_loader)

        # 학습 루프
        best_auc = 0
        best_calibration = float('inf')

        for epoch in range(self.config.num_epochs):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)

            if self.rank == 0:
                self.logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
                self.logger.info(f"Train - Total Loss: {train_metrics['total_loss']:.4f}, "
                               f"Doc Loss (AUC): {train_metrics['doc_loss']:.4f}, "
                               f"Para Loss: {train_metrics['para_loss']:.4f}")
                self.logger.info(f"Val - AUC: {val_metrics['auc']:.4f}, "
                               f"Calibration Error: {val_metrics['calibration_error']:.4f}")

                # 최고 성능 모델 저장 (AUC 기준)
                if val_metrics['auc'] >= best_auc:
                    best_auc = val_metrics['auc']
                    self.save_model(f"best_model_auc_{best_auc:.4f}_epoch_{epoch+1}.pt")
                    print(f"🎯 New best AUC model saved: {best_auc:.4f}")

                # 최고 보정 모델 저장 (Calibration Error 기준)
                if val_metrics['calibration_error'] <= best_calibration:
                    best_calibration = val_metrics['calibration_error']
                    self.save_model(f"best_calibrated_model_ce_{best_calibration:.4f}_epoch_{epoch+1}.pt")
                    print(f"🎯 New best calibrated model saved: CE {best_calibration:.4f}")

            # 모든 프로세스 동기화
            dist.barrier()

        # 모델 보정
        if self.rank == 0:
            print("\n🎯 Starting model calibration...")

        optimal_temp = self.calibrate_model(val_loader)

        # 보정 후 최종 검증
        final_val_metrics = self.validate(val_loader)

        if self.rank == 0:
            print(f"\n✅ Training completed!")
            print(f"   Best AUC: {best_auc:.4f}")
            print(f"   Best Calibration Error: {best_calibration:.4f}")
            print(f"   Final AUC (after calibration): {final_val_metrics['auc']:.4f}")
            print(f"   Final Calibration Error: {final_val_metrics['calibration_error']:.4f}")

            # 최종 보정된 모델 저장
            self.save_model(f"final_calibrated_model_temp_{optimal_temp:.3f}.pt")

        return best_auc, best_calibration

    def predict(self, test_data):
        """예측 (보정된 확률 출력)"""
        self.model.eval()

        # 테스트 데이터 필터링
        if self.config.filter_long_documents:
            if self.rank == 0:
                print("⚠️  Applying length filtering to test data (may exclude some samples)")
            filtered_test_data, test_filter_stats = filter_documents_by_length(
                test_data, self.config.max_doc_length, is_train=False, rank=self.rank
            )
        else:
            filtered_test_data = test_data

        test_dataset = EPAMILDataset(filtered_test_data, self.tokenizer, self.config, is_train=False, rank=self.rank)

        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False,
            drop_last=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            sampler=test_sampler,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False
        )

        predictions = []
        doc_ids = []
        raw_logits = []

        with torch.no_grad():
            iterator = tqdm(test_loader, desc="Prediction") if self.rank == 0 else test_loader

            for batch in iterator:
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                num_paragraphs = batch['num_paragraphs'].to(self.device, non_blocking=True)

                with autocast():
                    # Temperature scaling 적용된 예측
                    outputs = self.model(input_ids, attention_mask, num_paragraphs, apply_temperature=True)
                    doc_logits = outputs['document_logits']
                    if doc_logits.dim() > 1:
                        doc_logits = doc_logits.squeeze(-1)

                    # 보정된 확률 계산
                    calibrated_preds = torch.sigmoid(doc_logits).cpu().numpy()
                    logits = doc_logits.cpu().numpy()

                predictions.extend(calibrated_preds)
                doc_ids.extend(batch['doc_ids'])
                raw_logits.extend(logits)

        # DDP에서 모든 프로세스의 결과 수집
        if dist.is_initialized():
            gathered_predictions = [None for _ in range(self.world_size)]
            gathered_doc_ids = [None for _ in range(self.world_size)]
            gathered_logits = [None for _ in range(self.world_size)]

            dist.all_gather_object(gathered_predictions, predictions)
            dist.all_gather_object(gathered_doc_ids, doc_ids)
            dist.all_gather_object(gathered_logits, raw_logits)

            if self.rank == 0:
                # 모든 결과 합치기 - 중복 제거 및 정렬
                all_predictions = []
                all_doc_ids = []
                all_logits = []
                seen_ids = set()

                for pred_list, id_list, logit_list in zip(gathered_predictions, gathered_doc_ids, gathered_logits):
                    for doc_id, pred, logit in zip(id_list, pred_list, logit_list):
                        if doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            all_doc_ids.append(doc_id)
                            all_predictions.append(pred)
                            all_logits.append(logit)

                # ID 순서대로 정렬
                id_pred_logit_tuples = list(zip(all_doc_ids, all_predictions, all_logits))
                id_pred_logit_tuples.sort(key=lambda x: x[0])

                if id_pred_logit_tuples:
                    sorted_ids, sorted_preds, sorted_logits = zip(*id_pred_logit_tuples)
                    return list(sorted_ids), list(sorted_preds), list(sorted_logits)
                else:
                    return [], [], []
            else:
                return [], [], []

        return doc_ids, predictions, raw_logits

    def save_model(self, path):
        """모델 저장 (rank 0에서만)"""
        if self.rank == 0:
            torch.save({
                'model_state_dict': self.model.module.state_dict(),
                'config': self.config,
                'temperature': self.model.module.temperature_scaler.temperature.item()
            }, path)
            self.logger.info(f"Model saved to {path}")

    def load_model(self, path):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.module.load_state_dict(checkpoint['model_state_dict'])

        # Temperature 로드
        if 'temperature' in checkpoint:
            self.model.module.temperature_scaler.temperature.data = torch.tensor([checkpoint['temperature']])

        if self.rank == 0:
            temp_val = checkpoint.get('temperature', 1.0)
            self.logger.info(f"Model loaded from {path} (Temperature: {temp_val:.3f})")


class ImprovedEPAMILTrainerSingle:
    """개선된 EPA-MIL 단일 GPU 학습/예측 클래스 - 문단 기반"""

    def __init__(self, config: ImprovedConfig, device: str = "cuda:0"):
        self.config = config
        self.device = torch.device(device)

        # 토크나이저 및 모델 초기화
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = ImprovedEPAMILModel(config).to(self.device)

        # AUC 기반 손실 함수로 변경
        self.doc_criterion = FocalAUCLoss(
            gamma=config.focal_gamma,      # Focal loss gamma
            alpha=config.focal_alpha,      # Focal loss alpha
            auc_weight=config.auc_weight,  # AUC loss 비중 (30%)
            auc_gamma=config.auc_gamma     # AUC loss gamma
        )

        self.para_criterion = ImprovedFocalLoss(
            gamma=config.focal_gamma,
            alpha=config.focal_alpha,
            label_smoothing=config.label_smoothing
        )

    def predict(self, test_data):
        """단일 GPU 예측 (보정된 확률) - 문단 기반"""
        self.model.eval()

        # 테스트 데이터 필터링
        if self.config.filter_long_documents:
            filtered_test_data, filter_stats = filter_documents_by_length(
                test_data, self.config.max_doc_length, is_train=False, rank=0
            )
            print(f"🔍 Paragraph-based AUC-Enhanced Features:")
            print(f"   Temperature Scaling: {self.model.temperature_scaler.temperature.item():.3f}")
            print(f"   Focal + AUC Loss: γ={self.config.focal_gamma}, α={self.config.focal_alpha}, AUC weight={self.config.auc_weight}")
            print(f"   Label Smoothing: {self.config.label_smoothing}")
            print(f"   Multi-layer Classifiers: Enabled")
            print(f"   Processing unit: Paragraphs (\\n separated)")
        else:
            filtered_test_data = test_data

        test_dataset = EPAMILDataset(filtered_test_data, self.tokenizer, self.config, is_train=False, rank=0)

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size * 4,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

        predictions = []
        doc_ids = []
        confidence_scores = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="🎯 Generating AUC-optimized calibrated predictions"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                num_paragraphs = batch['num_paragraphs'].to(self.device)

                with autocast():
                    # Temperature scaling 적용된 예측
                    outputs = self.model(input_ids, attention_mask, num_paragraphs, apply_temperature=True)
                    doc_logits = outputs['document_logits']
                    if doc_logits.dim() > 1:
                        doc_logits = doc_logits.squeeze(-1)

                    # 보정된 확률 계산
                    calibrated_preds = torch.sigmoid(doc_logits).cpu().numpy()

                    # 신뢰도 점수 계산 (확률이 0.5에서 얼마나 떨어져 있는지)
                    confidence = np.abs(calibrated_preds - 0.5) * 2  # 0~1 범위로 정규화

                predictions.extend(calibrated_preds)
                doc_ids.extend(batch['doc_ids'])
                confidence_scores.extend(confidence)

        # 확신도 통계 출력
        high_confidence_count = sum(1 for conf in confidence_scores if conf > 0.8)
        medium_confidence_count = sum(1 for conf in confidence_scores if 0.5 <= conf <= 0.8)
        low_confidence_count = sum(1 for conf in confidence_scores if conf < 0.5)

        print(f"\n📊 AUC-Enhanced Prediction Confidence Statistics:")
        print(f"   High confidence (>0.8): {high_confidence_count:,} ({high_confidence_count/len(confidence_scores):.1%})")
        print(f"   Medium confidence (0.5-0.8): {medium_confidence_count:,} ({medium_confidence_count/len(confidence_scores):.1%})")
        print(f"   Low confidence (<0.5): {low_confidence_count:,} ({low_confidence_count/len(confidence_scores):.1%})")
        print(f"   Average confidence: {np.mean(confidence_scores):.3f}")
        print(f"   Prediction range: [{min(predictions):.3f}, {max(predictions):.3f}]")

        return doc_ids, predictions, confidence_scores

    def load_model(self, path):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Temperature 로드
        if 'temperature' in checkpoint:
            self.model.temperature_scaler.temperature.data = torch.tensor([checkpoint['temperature']], device=self.device)
            print(f"✅ Model loaded with calibrated temperature: {checkpoint['temperature']:.3f}")
        else:
            print(f"⚠️  Temperature not found in checkpoint, using default: {self.model.temperature_scaler.temperature.item():.3f}")

        print(f"📂 Model loaded from {path}")


def setup_ddp():
    """DDP 초기화 - 경고 해결"""
    # Tokenizer 병렬화 경고 해결
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # OMP 스레드 설정
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = "1"

    # CUDA 메모리 최적화
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # DDP 초기화 - device_id 명시적으로 설정
    local_rank = int(os.environ["LOCAL_RANK"])

    # init_process_group에 device_id 전달
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=datetime.timedelta(seconds=1800)  # 30분 타임아웃
    )

    # 명시적으로 CUDA 디바이스 설정
    torch.cuda.set_device(local_rank)

    # CUDA 메모리 정리
    torch.cuda.empty_cache()

    # 프로세스 그룹이 올바르게 초기화되었는지 확인
    if dist.is_initialized():
        # device_ids를 명시적으로 지정하여 barrier 수행
        dist.barrier(device_ids=[local_rank])

    return local_rank


def cleanup_ddp():
    """DDP 정리 - 올바른 순서로"""
    if dist.is_initialized():
        try:
            # 모든 프로세스 동기화
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            dist.barrier(device_ids=[local_rank])

            # 프로세스 그룹 정리
            dist.destroy_process_group()
        except Exception as e:
            print(f"Warning: Error during DDP cleanup: {e}")
        finally:
            # CUDA 캐시 정리
            torch.cuda.empty_cache()


def set_seed(seed: int, rank: int):
    """시드 설정"""
    import random
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_train(config: ImprovedConfig, args):
    """학습 모드: DDP 사용"""
    local_rank = setup_ddp()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    set_seed(42, rank)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"🚀 Starting AUC-Enhanced EPA-MIL training on {world_size} GPUs")
        print(f"📊 Using model: {config.model_name}")
        print(f"📝 Processing unit: PARAGRAPHS (\\n separated)")
        print(f"⚡ Effective batch size: {config.batch_size * world_size * config.gradient_accumulation_steps}")
        print(f"🎯 AUC-Enhanced Features:")
        print(f"   Focal + AUC Loss: γ={config.focal_gamma}, α={config.focal_alpha}, AUC weight={config.auc_weight}")
        print(f"   AUC Loss gamma: {config.auc_gamma}")
        print(f"   Label Smoothing: {config.label_smoothing}")
        print(f"   Temperature Scaling: {config.temperature}")
        print(f"   Multi-layer Classifiers: Enabled")
        print(f"   Attention Heads: 8 (768/8=96 per head)")
        print(f"   Differential Learning Rates: Backbone={config.backbone_learning_rate:.1e}, Head={config.head_learning_rate:.1e}")
        print(f"📏 Max document length: {config.max_doc_length}")
        print(f"📏 Max paragraph length: {config.max_paragraph_length}")
        print(f"🔍 Document filtering: {'Enabled' if config.filter_long_documents else 'Disabled'}")

    # 명시적으로 device_ids 지정하여 barrier
    dist.barrier(device_ids=[local_rank])

    # 데이터 로드
    if rank == 0:
        if not args.train_file:
            raise ValueError("Train file required for training mode")
        train_df = pd.read_csv(args.train_file)
        print(f"📂 Loaded training data: {len(train_df):,} samples")
    else:
        train_df = None

    data_list = [train_df]
    dist.broadcast_object_list(data_list, src=0)
    train_df = data_list[0]

    # Train/Val 분할
    train_size = int(len(train_df) * config.train_val_split)
    train_data = train_df.iloc[:train_size].reset_index(drop=True)
    val_data = train_df.iloc[train_size:].reset_index(drop=True)

    if rank == 0:
        print(f"📊 Train/Val split: {len(train_data):,} / {len(val_data):,}")

    # 학습
    try:
        trainer = ImprovedEPAMILTrainer(config, rank, world_size)
        trainer.train(train_data, val_data)
    except Exception as e:
        if rank == 0:
            print(f"❌ Training failed: {e}")
        raise e
    finally:
        cleanup_ddp()


def run_predict(config: ImprovedConfig, args):
    """예측 모드: 단일 GPU 사용"""
    print("🔍 Starting AUC-Enhanced Single GPU Prediction (Paragraph-based)...")
    print(f"📝 Processing unit: PARAGRAPHS (\\n separated)")
    print(f"🎯 AUC-Enhanced Features Enabled:")
    print(f"   Focal + AUC Loss: γ={config.focal_gamma}, α={config.focal_alpha}, AUC weight={config.auc_weight}")
    print(f"   AUC Loss gamma: {config.auc_gamma}")
    print(f"   Temperature Scaling: {config.temperature}")
    print(f"   Label Smoothing: {config.label_smoothing}")
    print(f"   Multi-layer Classifiers: Enabled")
    print(f"📏 Max document length: {config.max_doc_length}")
    print(f"📏 Max paragraph length: {config.max_paragraph_length}")
    print(f"🔍 Document filtering: {'Enabled' if config.filter_long_documents else 'Disabled'}")

    os.makedirs(args.output_dir, exist_ok=True)

    # 데이터 로드
    if not args.test_file:
        raise ValueError("Test file required for prediction mode")
    if not args.model_path:
        raise ValueError("Model path required for prediction mode")

    test_df = pd.read_csv(args.test_file)
    print(f"📂 Loaded test data: {len(test_df):,} samples")

    # 단일 GPU 예측
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trainer = ImprovedEPAMILTrainerSingle(config, device)
    trainer.load_model(args.model_path)

    # 예측
    doc_ids, predictions, confidence_scores = trainer.predict(test_df)

    # 결과 저장
    result_df = pd.DataFrame({
        'ID': doc_ids,
        'generated': predictions,
        'confidence': confidence_scores
    })

    # 기본 제출 파일 (confidence 열 제외)
    submission_df = result_df[['ID', 'generated']].copy()

    output_path = os.path.join(args.output_dir, 'submission.csv')
    submission_df.to_csv(output_path, index=False)

    # 상세 결과 파일 (confidence 포함)
    detailed_path = os.path.join(args.output_dir, 'detailed_predictions.csv')
    result_df.to_csv(detailed_path, index=False)

    print(f"✅ Standard submission saved to {output_path}")
    print(f"📊 Detailed predictions saved to {detailed_path}")
    print(f"📈 Total predictions: {len(doc_ids)}")

    # 확신도 분석 출력
    high_conf_samples = result_df[result_df['confidence'] > 0.8]
    medium_conf_samples = result_df[(result_df['confidence'] >= 0.5) & (result_df['confidence'] <= 0.8)]
    low_conf_samples = result_df[result_df['confidence'] < 0.5]

    print(f"\n📋 Sample High Confidence Predictions (confidence > 0.8):")
    if len(high_conf_samples) > 0:
        print(high_conf_samples.head(5).to_string(index=False))
    else:
        print("   No high confidence predictions found")

    print(f"\n📋 Sample Low Confidence Predictions (confidence < 0.5):")
    if len(low_conf_samples) > 0:
        print(low_conf_samples.head(3).to_string(index=False))
    else:
        print("   No low confidence predictions found")

    # 통계 요약
    print(f"\n📊 Final AUC-Enhanced Prediction Summary:")
    print(f"   Generated=1 predictions: {sum(result_df['generated'] > 0.5):,}")
    print(f"   Generated=0 predictions: {sum(result_df['generated'] <= 0.5):,}")
    print(f"   High confidence (>0.8): {len(high_conf_samples):,}")
    print(f"   Medium confidence (0.5-0.8): {len(medium_conf_samples):,}")
    print(f"   Low confidence (<0.5): {len(low_conf_samples):,}")
    print(f"   Average prediction: {np.mean(predictions):.3f}")
    print(f"   Average confidence: {np.mean(confidence_scores):.3f}")
    print(f"   AUC optimization impact: Direct ROC-AUC targeting enabled")


def main():
    parser = argparse.ArgumentParser(description='Improved EPA-MIL with AUC-based Loss Functions')
    parser.add_argument('--train_file', type=str, help='Training data file')
    parser.add_argument('--test_file', type=str, help='Test data file')
    parser.add_argument('--model_name', type=str, default='klue/roberta-large', help='Model name')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train', help='Mode: train or predict')
    parser.add_argument('--model_path', type=str, help='Model path for prediction')
    parser.add_argument('--max_doc_length', type=int, default=2000, help='Maximum document length')
    parser.add_argument('--max_paragraph_length', type=int, default=500, help='Maximum paragraph length')
    parser.add_argument('--disable_filtering', action='store_true', help='Disable document length filtering')

    # 확신도 관련 파라미터
    parser.add_argument('--focal_gamma', type=float, default=3.0, help='Focal loss gamma')
    parser.add_argument('--focal_alpha', type=float, default=0.75, help='Focal loss alpha')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--temperature', type=float, default=1.5, help='Initial temperature for scaling')
    parser.add_argument('--calibration_method', type=str, default='temperature', choices=['temperature', 'platt', 'isotonic'])

    # AUC Loss 관련 파라미터
    parser.add_argument('--auc_weight', type=float, default=0.3, help='AUC loss weight')
    parser.add_argument('--auc_gamma', type=float, default=1.0, help='AUC loss gamma')

    args = parser.parse_args()

    # 개선된 설정
    config = ImprovedConfig()
    config.model_name = args.model_name
    config.max_doc_length = args.max_doc_length
    config.max_paragraph_length = args.max_paragraph_length
    config.filter_long_documents = not args.disable_filtering
    config.focal_gamma = args.focal_gamma
    config.focal_alpha = args.focal_alpha
    config.label_smoothing = args.label_smoothing
    config.temperature = args.temperature
    config.calibration_method = args.calibration_method
    config.auc_weight = args.auc_weight
    config.auc_gamma = args.auc_gamma

    if args.mode == 'train':
        run_train(config, args)
    elif args.mode == 'predict':
        run_predict(config, args)


if __name__ == "__main__":
    main()