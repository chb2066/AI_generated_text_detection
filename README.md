# AI Generated Text Detection

> **2025 SW중심대학 디지털 경진대회 (AI부문) — 생성형 AI(LLM)와 인간: 텍스트 판별 챌린지**
> 
> 주최: SW중심대학협의회 · 진행: DACON · 팀 프로젝트

문단 단위로 주어진 한국어 텍스트가 사람이 작성한 것(0)인지 생성형 AI가 작성한 것(1)인지 판별하여, 각 문단이 AI가 작성했을 확률(0~1)을 예측하는 과제입니다. 평가 데이터는 문단 단위 샘플로 구성되며, 같은 `title`을 가진 문단들은 하나의 글에 속하므로 **동일 글 내 문단 간 상호 참조가 허용**됩니다.

이 저장소는 해당 과제를 **Multiple Instance Learning(MIL)** 문제로 접근합니다. 같은 글에 속한 문단들을 하나의 bag으로 묶어 문서 레벨로 인코딩·집계함으로써, 대회 규칙이 허용하는 "동일 글 내 문단 간 상호 참조"를 자연스럽게 활용합니다. 두 모델 모두 문서를 문단 단위로 나누어 인코딩한 뒤 문서 레벨 예측으로 집계하지만, 집계와 예측 단계를 처리하는 순서가 다릅니다.

| 파일 | 모델 | 처리 흐름 |
|---|---|---|
| [EPA.py](EPA.py) | EPA-MIL (`ImprovedEPAMILModel`) | Embed → Predict(문단별) → Aggregate |
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
| `title` | 문서 제목 |
| `full_text` | 문서 전체 텍스트 |
| `generated` | 라벨 (0: 사람 작성, 1: AI 생성) |

**테스트 데이터 (`--test_file`, CSV)**

| 컬럼 | 설명 |
|---|---|
| `ID` | 문단(샘플) 식별자 |
| `title` | 문서 제목 (같은 `title`은 하나의 글에 속함) |
| `paragraph_index` | 글 내 문단 순서 |
| `paragraph_text` | 예측 대상 문단 텍스트 |

> 코드는 문단 텍스트(`paragraph_text`)와 `title`, `ID`를 사용합니다. `paragraph_index`는 파일에 포함되지만 학습/추론에는 직접 사용하지 않습니다.

**제출 파일 (`submission.csv`)**

| 컬럼 | 설명 |
|---|---|
| `ID` | 문단(샘플) 식별자 |
| `generated` | 예측 확률 (0~1) |

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

이 저장소의 코드는 [MIT License](LICENSE)를 따릅니다. 단, 사용하는 사전학습 모델(`klue/bert-base`, `klue/roberta-large` 등)과 대회 데이터는 각 출처의 라이선스 및 이용약관을 따릅니다.
