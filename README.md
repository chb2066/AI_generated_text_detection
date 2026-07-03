# AI Generated Text Detection

한국어 문서가 AI에 의해 작성되었는지 여부를 예측하는 Multiple Instance Learning(MIL) 기반 문서 분류 모델 두 가지 구현을 제공합니다. 두 모델 모두 문서를 문단 단위로 나누어 인코딩한 뒤 문서 레벨 예측으로 집계하는 방식을 사용하지만, 집계와 예측 단계를 처리하는 순서가 다릅니다.

| 파일 | 모델 | 처리 흐름 |
|---|---|---|
| [EPA.py](EPA.py) | EPA-MIL (`ImprovedEPAMILModel`) | Embed → Predict(문단별) → Aggregate → Predict(문서 레벨) |
| [EAP.py](EAP.py) | EAP-MIL (`EAPModel`) | Embed → Aggregate → Predict(문서 레벨) |

## 개요

- **백본**: HuggingFace `transformers`의 한국어 사전학습 모델 (기본값 `klue/bert-base`, CLI에서 `klue/roberta-large` 등으로 변경 가능)
- **입력 단위**: 문서를 줄바꿈(`\n`) 기준으로 문단으로 분할한 뒤, 문단별로 토크나이즈하여 MIL 형태(bag of paragraphs)로 모델에 입력
- **손실 함수**: Focal Loss + differentiable AUC Loss를 결합한 `FocalAUCLoss`로 클래스 불균형과 ROC-AUC 최적화를 동시에 다룸
- **보정(Calibration)**: 검증 데이터 기반 Temperature Scaling으로 예측 확률의 과신(overconfidence)을 완화하고 Expected Calibration Error(ECE)를 관리
- **분산 학습**: `torch.distributed` + `DistributedDataParallel(DDP)` 기반 멀티 GPU 학습, Mixed Precision(AMP) 지원
- **예측**: 단일 GPU에서 보정된 확률과 신뢰도(confidence) 점수를 함께 출력

### EPA.py — EPA-MIL

- 문단 레벨 분류기와 문서 레벨 분류기를 모두 학습 (`doc_loss + lambda_paragraph * para_loss`)
- 문단 레벨 로짓으로 top-k 문단을 선택한 뒤 Multi-head Self-Attention으로 어그리게이션
- 스케줄러: `CosineAnnealingWarmRestarts`

### EAP.py — EAP-MIL

- 문서 레벨 손실만 학습 (문단별 예측 헤드 없음)
- 문단 임베딩의 norm 기반 중요도로 top-k 필터링 후 `mean` / `max` / `attention` / `weighted_attention` 중 선택 가능한 방식으로 어그리게이션
- 스케줄러: Linear warmup (`get_linear_schedule_with_warmup`)

두 스크립트 모두 독립적으로 실행 가능한 CLI(`argparse`)를 제공하며, 데이터 전처리(`TextPreprocessor`, `filter_documents_by_length`)와 학습/추론 파이프라인 구조가 거의 동일합니다.

## 설치

```bash
pip install -r requirements.txt
```

CUDA 지원 GPU와 [PyTorch의 CUDA 빌드](https://pytorch.org/get-started/locally/) 설치를 권장합니다. `torch.cuda.amp`, `torch.distributed`(NCCL 백엔드)를 사용하므로 학습에는 NVIDIA GPU가 필요합니다.

## 데이터 형식

**학습 데이터 (`--train_file`, CSV)**

| 컬럼 | 설명 |
|---|---|
| `full_text` | 문서 전체 텍스트 |
| `title` | 문서 제목 |
| `generated` | 라벨 (0: 사람 작성, 1: AI 생성) |

**테스트 데이터 (`--test_file`, CSV)**

| 컬럼 | 설명 |
|---|---|
| `ID` | 문서 식별자 |
| `title` | 문서 제목 |
| `paragraph_text` | 예측 대상 텍스트 |

## 사용법

### 학습 (멀티 GPU, DDP)

```bash
torchrun --nproc_per_node=<GPU_수> EPA.py \
  --mode train \
  --train_file data/train.csv \
  --output_dir ./outputs \
  --model_name klue/roberta-large

torchrun --nproc_per_node=<GPU_수> EAP.py \
  --mode train \
  --train_file data/train.csv \
  --output_dir ./eap_outputs \
  --model_name klue/roberta-large
```

학습 중 매 epoch마다 AUC 및 Calibration Error 기준 최고 성능 체크포인트가 저장되며, 학습 종료 후 Temperature Scaling으로 보정된 최종 모델(`final_*_calibrated_model_*.pt`)이 저장됩니다.

### 예측 (단일 GPU)

```bash
python EPA.py \
  --mode predict \
  --test_file data/test.csv \
  --model_path outputs/final_calibrated_model_temp_1.234.pt \
  --output_dir ./outputs

python EAP.py \
  --mode predict \
  --test_file data/test.csv \
  --model_path eap_outputs/final_eap_calibrated_model_temp_1.234.pt \
  --output_dir ./eap_outputs
```

예측 결과는 `<output_dir>/submission.csv`(`ID`, `generated` 확률)와 `<output_dir>/detailed_predictions.csv`(`confidence` 포함)로 저장됩니다.

### 주요 CLI 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--model_name` | HuggingFace 백본 모델 이름 | `klue/roberta-large` |
| `--max_doc_length` / `--max_paragraph_length` | 문서/문단 최대 길이(문자 수) 필터링 기준 | 2000 / 500 |
| `--disable_filtering` | 문서 길이 기반 필터링 비활성화 | False |
| `--focal_gamma` / `--focal_alpha` | Focal Loss 파라미터 | 3.0 / 0.75 |
| `--auc_weight` | 전체 손실에서 AUC Loss 비중 | 0.3 |
| `--label_smoothing` | 라벨 스무딩 계수 | EPA: 0.1, EAP: 0.05 |
| `--temperature` | Temperature Scaling 초기값 | EPA: 1.5, EAP: 2.0 |
| `--top_k_ratio` (EAP 전용) | 문단 어그리게이션 시 사용할 상위 비율 | 0.7 |

전체 옵션은 각 스크립트의 `--help`로 확인할 수 있습니다.

## 라이선스

별도 명시가 없는 한 저장소 소유자에게 저작권이 있습니다.
