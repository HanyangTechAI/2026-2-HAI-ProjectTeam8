# 2026-2-HAI-ProjectTeam8 · GNN 기반 가상자산 이상거래 탐지

XGBoost vs GCN · GraphSAGE · GAT 비교, 그리고 Skip-GCN 하이브리드. 계획서 3장의 공통 기반을 구현한 저장소입니다.

- 계획서: [`docs/프로젝트_계획서.pdf`](docs/프로젝트_계획서.pdf) · 타임라인: [`docs/타임라인.pdf`](docs/타임라인.pdf)
- 계획서와 코드가 다르면 **이 README와 코드가 우선**입니다 (로더는 PyG가 아니라 원본 CSV를 직접 읽고, 결과 CSV는 `results/metrics/{model}.csv`로 모델별 분리).
- 일정: 11/1 공통 기반 완료 ✅ → 11/12 모델 3종 구현 → 11/19 합치기 → 11/26 마무리 → 11/29 최종 발표
- 현재 상태: 데이터 검증, XGBoost 베이스라인, GCN 1차 결과(튜닝 전)까지 완료. **GNN 조는 아래 [XGBoost 베이스라인 결과](#xgboost-베이스라인-결과)와 비교하는게 목적입니다. 논문에서 말했듯이, 각 gnn모델들의 결과 자체는 XGBoost보다 낮을 것으로 예상. 하이퍼파라미터 튜닝을 하며 각 모델의 성능을 최대한으로 끌러올리는게 우선 목표.**

## 규칙 (모든 조 필수)
1. 데이터는 `common.data.load_data()`, 평가는 `common.metrics`, 저장은 `common.utils`만 사용합니다. 직접 구현 금지.
2. 테스트 구간(시간 단계 35~49)은 튜닝·조기 종료·임계값 결정에 쓰지 않습니다. 최종 보고에만 사용합니다.
3. 시드 0~4 (최소 3회), 결과는 평균 ± 표준편차로 보고합니다.
4. `common/`을 바꿔야 하면 요청하고 전원에게 공지합니다.
5. 결과는 아래 형식으로 저장합니다 (노드 순서 = `load_data()` 순서 = txId 오름차순).

| 파일 | 형식 |
|---|---|
| `results/metrics/{model}.csv` | 한 줄 = (model, seed, split, threshold_type). 같은 키는 재실행 시 덮어씀. 컬럼은 `common/utils.py`의 `METRIC_COLUMNS` |
| `results/pred/pred_{model}_seed{n}.npy` | `(N,)` float32, 불법 확률 |
| `results/emb/emb_{model}_seed{n}.npy` | `(N, 64)` float32, 마지막 GNN 층 임베딩. **용량이 커서 git 제외** → 공유 드라이브에 올리기 |

model 이름: `xgb`, `gcn`, `sage`, `gat`, 하이브리드는 `hybrid_gcn` 등. threshold_type: `fixed_0.5`, `val_opt`.

## 데이터 준비
원본 CSV 3개(`elliptic_txs_features.csv`, `elliptic_txs_edgelist.csv`, `elliptic_txs_classes.csv`)가 `data/elliptic/` 아래에 있으면 됩니다 (`data/elliptic/raw/`도 인식).

- **방법 A (Kaggle):** Kaggle "Elliptic Data Set" 다운로드 후 압축을 풀어 CSV 3개를 `data/elliptic/`에 둡니다.
- **방법 B (PyG로 1회 내려받기):**
  ```python
  from torch_geometric.datasets import EllipticBitcoinDataset
  EllipticBitcoinDataset(root="data/elliptic")   # raw CSV가 data/elliptic/raw 에 저장됨
  ```
  (저장 위치는 PyG 버전에 따라 다를 수 있으니 CSV 3개가 어디에 생겼는지 확인하세요.)

## 실행 환경

| 환경 | 용도 | xgboost | 비고 |
|---|---|---|---|
| Colab (CPU) | **공식 기준선 결과** (`results/`), GNN 학습 | 3.4.1 (Colab pip에 3.4.2가 아직 없음) | `notebooks/colab_xgboost.ipynb` |
| Colab (T4 GPU) | GCN 학습 (`results/`) | - | `notebooks/colab_gcn.ipynb` |

- `requirements.txt`는 `xgboost>=3.4.1,<3.5`입니다. 환경·버전이 다르면 **같은 코드·시드여도 소수점 둘째 자리쯤 차이**가 날 수 있습니다 (튜닝에서 뽑히는 설정이 달라질 수 있기 때문).
- **학습·추론 시간은 같은 환경에서 잰 값끼리만 비교합니다.** 보고서 표에는 환경(CPU/GPU 종류)을 반드시 적습니다.
- GNN 조는 Colab에서 학습합니다 (`torch`, `torch-geometric` 설치). GPU 종류(T4 등)는 결과 `notes`나 보고서에 적어 주세요.

### Colab에서 돌리는 법 (요약)
1. 구글 드라이브 `내 드라이브/Colab Notebooks/elliptic/`에 CSV 3개를 올립니다.
2. Colab에서 `notebooks/colab_xgboost.ipynb`를 열고 위에서부터 차례로 실행합니다 (repo가 private이면 GitHub 토큰 필요).
3. 노트북 실행 결과는 `results_colab/`에 저장되고 드라이브에도 백업됩니다. 공식 결과는 이를 `results/`에 복사한 것입니다.

## 공통 기반 검증 (완료) — 재현 방법
```bash
pip install -r requirements.txt
python scripts/verify_data.py data/elliptic      # 모두 OK여야 함 (노드 203,769 / 불법 4,545 / 정상 42,019 / 미분류 157,205 등)
python -m baselines.xgb_baseline --data-root data/elliptic    # 튜닝 30회 + 시드 5개 (CPU에서 약 1분)
python scripts/collect_results.py                # 요약표
```
`verify_data.py`를 실제 Elliptic 데이터로 실행해 15개 검사를 모두 통과했습니다 (2026-10-28). 분할별 노드 수:

| split | labeled_nodes | illicit | licit | illicit_ratio |
|---|---|---|---|---|
| train (1~29) | 26,381 | 2,871 | 23,510 | 0.1088 |
| val (30~34) | 3,513 | 591 | 2,922 | 0.1682 |
| test (35~49) | 16,670 | 1,083 | 15,587 | 0.0650 |


## XGBoost 베이스라인 결과

테스트 구간(시간 단계 35~49), 시드 0~4 평균 ± 표준편차. 최종 모델은 Train+Val(1~34)로 다시 학습했고, 테스트는 1회만 평가했습니다. **환경: Colab CPU, xgboost 3.4.1.**

| 지표 | 임계값 0.5 | 검증 구간 최적 임계값 | 설명 |
|---|---|---|---|
| PR-AUC | 0.802 ± 0.002 | 0.802 ± 0.002 | **주 지표.** 임계값을 바꿔가며 그린 정밀도-재현율 곡선 아래 넓이(0~1, 클수록 좋음). 불법이 드문 데이터에서 신뢰할 만하고 임계값에 영향받지 않음. 무작위 예측은 테스트 불법 비율(0.065) |
| ROC-AUC | 0.937 ± 0.002 | 0.937 ± 0.002 | 불법 거래가 정상보다 높은 점수를 받을 확률(0.5=무작위, 1=완벽). 불법이 드물면 실제보다 좋게 보일 수 있어 보조로만 봄 |
| Precision | 0.818 ± 0.041 | 0.874 ± 0.048 | 정밀도. 불법이라고 예측한 것 중 진짜 불법의 비율. 낮으면 오탐(정상을 불법으로 오인)이 많다는 뜻 |
| Recall | 0.737 ± 0.001 | 0.733 ± 0.003 | 재현율. 진짜 불법 중 모델이 잡아낸 비율. 낮으면 놓치는 불법이 많다는 뜻 |
| **F1 (불법)** | **0.775 ± 0.018** | **0.797 ± 0.019** | 정밀도와 재현율의 조화평균. 둘 다 높아야 높음 |

- **임계값**은 "불법 확률이 얼마 이상이면 불법으로 판정할지"의 기준입니다. `0.5`는 고정값이고, `검증 구간 최적`은 검증 구간(30~34)에서 F1이 가장 높았던 값을 골라 테스트에 그대로 적용한 것입니다 (테스트에서는 다시 고르지 않음). PR-AUC와 ROC-AUC는 임계값과 무관해서 두 열이 같습니다.
- **±**는 시드 5개(0~4) 결과의 표준편차입니다. 작을수록 결과가 안정적입니다.
- 학습 5.4초 · 추론 0.059초 (Colab CPU). 속도 비교는 같은 환경에서 잰 값끼리만 합니다.
- 설정: `max_depth=8, learning_rate=0.1, min_child_weight=5, subsample=1.0, colsample_bytree=0.7`, 학습 구간 정상/불법 비로 `scale_pos_weight`. 튜닝 30회는 검증 PR-AUC 기준(검증 PR-AUC 0.99). 자세한 기록은 `results/xgb_tuning.csv`.
- 무작위 예측의 PR-AUC 기준선은 테스트 불법 비율인 0.065입니다. 원논문(Weber 2019)의 랜덤포레스트 불법 F1 약 0.8과 비슷한 수준입니다.

**GNN 조가 넘어야 할 기준:** PR-AUC **0.802**, F1 **0.78~0.80**. 같은 지표·같은 분할·같은 시드로 비교합니다.

### 결과 파일 위치
| 파일 | 내용 |
|---|---|
| `results/summary.csv` | 모델·임계값별 평균/표준편차 요약 |
| `results/metrics/xgb.csv` | 시드·분할·임계값별 상세 |
| `results/pred/pred_xgb_seed{n}.npy` | 시드별 불법 확률 (오류 분석·하이브리드용) |
| `results/timestep_f1_xgb.csv` | 시간 단계별 F1 |
| `results/xgb_feature_importance.csv`, `results/xgb_tuning.csv` | 특징 중요도, 튜닝 기록 |

## GCN 결과

테스트 구간(시간 단계 35~49), 시드 0~4 평균 ± 표준편차. 학습·평가 절차는 XGBoost와 같습니다 (Train 학습 + 검증 조기 종료 → Train+Val 재학습 → 테스트 1회 평가). **환경: Colab L4 GPU. 하이퍼파라미터는 검증 PR-AUC 기준 튜닝 후 값 (조기 종료 `patience=100`).**

| 지표 | 임계값 0.5 | 검증 구간 최적 임계값 | 설명 |
|---|---|---|---|
| PR-AUC | 0.662 ± 0.024 | 0.662 ± 0.024 | **주 지표.** 임계값과 무관 (XGBoost 0.802) |
| ROC-AUC | 0.888 ± 0.007 | 0.888 ± 0.007 | 불법이 정상보다 높은 점수를 받을 확률. 보조 지표 (XGBoost 0.937) |
| Precision | 0.724 ± 0.063 | 0.870 ± 0.030 | 불법이라고 예측한 것 중 진짜 불법의 비율 |
| Recall | 0.590 ± 0.017 | 0.452 ± 0.024 | 진짜 불법 중 잡아낸 비율 |
| **F1 (불법)** | **0.649 ± 0.026** | **0.595 ± 0.025** | 정밀도와 재현율의 조화평균 (XGBoost 0.775 / 0.797) |

지표와 임계값의 뜻은 [XGBoost 결과](#xgboost-베이스라인-결과)의 설명과 같습니다.

- 학습 6.2 ± 1.9초 · 추론 0.050초 (Colab L4 GPU, 전체 그래프 1회 추론). 파라미터 14,914개. XGBoost는 환경이 달라 속도를 직접 비교할 수 없고, 1차 결과(T4)와도 장치가 달라 직접 비교할 수 없습니다.
- 설정: 2층 GCN(`hidden=64`, 마지막 conv 출력이 64차원 임베딩) + 선형 분류기, full-batch 학습, `dropout=0.5`, Adam `lr=0.01`, `weight_decay=5e-4`, 최대 500 에폭, 검증 PR-AUC 기준 조기 종료(`patience=100`), 클래스 가중치는 학습 구간 정상/불법 비.
- 시드별 테스트 PR-AUC는 0.660 / 0.626 / 0.666 / 0.666 / 0.693이고, 조기 종료 시점(best_epoch)은 144 / 167 / 203 / 306 / 224입니다. 검증 PR-AUC는 0.893 ± 0.006입니다.
- 검증 최적 임계값은 시드마다 0.74~0.90으로 높고, 임계값 0.5보다 F1이 낮습니다. 검증 구간의 불법 비율(16.8%)이 테스트(6.5%)보다 높아 검증에서 고른 임계값이 테스트로 잘 옮겨지지 않는 것으로 보입니다.
- **같은 설정으로 다시 돌려도 값이 조금씩 달라집니다.** 같은 조합(`lr 0.01 / dropout 0.3 / wd 1e-3`)을 반복 실행한 검증 PR-AUC 평균이 patience 30에서 0.846 / 0.850 / 0.865, patience 100에서 0.886 / 0.886 / 0.891이었습니다. 시드를 고정해도 GPU 연산 순서가 달라지는 것이 원인으로 추정되며, 직접 확인하지는 않았습니다. 표의 값은 한 번 실행한 결과입니다.

### GCN 하이퍼파라미터 튜닝

계획서 4.1 범위(lr × dropout × weight_decay, 8개 조합)를 시드 0~4로 학습하고 **검증 PR-AUC 평균**으로 골랐습니다. 테스트 구간은 선택에 쓰지 않았습니다. 조기 종료 patience 30으로 먼저 돌린 뒤, patience를 100(최대 500 에폭)으로 늘려 다시 돌렸습니다.

| 조합 | lr | dropout | weight_decay | 검증 PR-AUC (patience 100) | 표준편차 | 검증 PR-AUC (patience 30) | 표준편차 |
|---|---|---|---|---|---|---|---|
| **1 (선택)** | **0.01** | **0.5** | **5e-4** | **0.8969** | **0.0040** | 0.8515 | 0.0405 |
| 2 | 0.005 | 0.5 | 5e-4 | 0.8910 | 0.0113 | 0.8420 | 0.0577 |
| 3 | 0.01 | 0.5 | 1e-3 | 0.8889 | 0.0075 | 0.8439 | 0.0483 |
| 4 | 0.01 | 0.3 | 5e-4 | 0.8882 | 0.0093 | 0.8360 | 0.0385 |
| 5 | 0.005 | 0.5 | 1e-3 | 0.8862 | 0.0045 | 0.8308 | 0.0571 |
| 6 | 0.01 | 0.3 | 1e-3 | 0.8855 | 0.0065 | 0.8651 | 0.0356 |
| 7 | 0.005 | 0.3 | 5e-4 | 0.8765 | 0.0156 | 0.8198 | 0.0510 |
| 8 | 0.005 | 0.3 | 1e-3 | 0.8739 | 0.0034 | 0.8118 | 0.0515 |

- **patience 변경이 하이퍼파라미터 조합보다 훨씬 큰 효과**를 냈습니다. 같은 조합(`lr 0.01 / dropout 0.3 / wd 1e-3`)에서 patience 30은 검증 PR-AUC 0.850 ± 0.030(best_epoch 22~136), patience 100은 0.891 ± 0.010(best_epoch 154~293)이었습니다 (`results/tuning/gcn/patience_p30_vs_p100_do0.3.csv`). patience 30은 학습을 일찍 끊어 시드 편차가 커졌습니다.
- 테스트 PR-AUC 참고값 (Colab 실행 출력 기준, 결과 파일은 덮어써져 저장소에 없음): 1차(튜닝 전) 0.482 ± 0.160 → 8조합 튜닝 후 patience 30 0.565 ± 0.127 → patience 100 0.661 ± 0.021 → 최종 0.662 ± 0.024.
- **조합 간 차이는 작습니다.** patience 100에서 1위와 2위의 차이는 0.006으로 같은 설정을 반복 실행했을 때의 변동(약 0.005~0.006)과 비슷해서 우열을 주장하기 어렵습니다. 인자별 평균은 dropout 0.5가 0.3보다 +0.010, lr 0.01이 0.005보다 +0.008, weight_decay 5e-4가 1e-3보다 +0.005로 경향은 약합니다. 계획서 기본값(lr 0.005, dropout 0.5, wd 5e-4)이 2위입니다.
- patience 30 그리드의 순위는 조기 종료 시점의 잡음에 흔들렸습니다. lr 0.005 조합 중 일부 시드의 best_epoch가 최대 456까지 가서 최대 에폭 500에 가까웠으므로, lr 0.005 쪽이 약간 불리하게 측정됐을 수 있습니다.
- **모델 간 비교 시 주의:** 조기 종료 patience 설정이 결과에 크게 영향을 주므로, 다른 모델과 비교할 때는 patience가 같은 조건인지 확인해야 합니다 (GCN 최종 설정은 patience 100 / 최대 500 에폭).
- 이 값들은 `notebooks/colab_gcn_tune.ipynb`로 만들었습니다.

### GCN 결과 파일 위치
| 파일 | 내용 |
|---|---|
| `results/metrics/gcn.csv` | 시드·분할·임계값별 상세 (`notes`에 best_epoch 기록) |
| `results/pred/pred_gcn_seed{n}.npy` | 시드별 불법 확률 (오류 분석·하이브리드용) |
| `results/emb/emb_gcn_seed{n}.npy` | 시드별 64차원 임베딩 (약 50MB씩, **git 제외**, 공유 드라이브 사용) |
| `results/timestep_f1_gcn.csv` | 시간 단계별 F1 |
| `results/summary.csv` | XGBoost·GCN 요약 (`scripts/collect_results.py`로 재생성) |
| `results/tuning/gcn/grid_p100.csv`, `grid_p30.csv` | 8조합 그리드 요약 (patience 100 / 30) |
| `results/tuning/gcn/patience_p30_vs_p100_do0.3.csv` | patience 30 vs 100 비교 (조합 `lr 0.01 / dropout 0.3 / wd 1e-3`) |
| `notebooks/colab_gcn_tune.ipynb` | 튜닝·최종 학습 Colab 노트북 |

## 모델 비교: XGBoost vs GCN

테스트 구간(시간 단계 35~49), 시드 0~4 평균. 같은 데이터·분할·지표·시드로 평가했습니다. **GCN은 하이퍼파라미터 튜닝 후 결과(patience 100)** 입니다. 다른 모델과 비교할 때는 조기 종료 patience 설정이 같은 조건인지 확인하세요. 

Elliptic 원논문(Weber et al., 2019)는 time step 43무렵에 다크마켓(불법 거래 시장)이 폐쇄되는 일이 있었다고함. 그 뒤로 학습된 모델들이 새로운 불법 패턴을 잡아내지 못 할 가능성이 있기에, 평가를 35-42와 43-49로 나누어서 결과를 출력함.

| 지표 | XGBoost | GCN | 차이 (GCN − XGBoost) |
|---|---|---|---|
| **PR-AUC** (주 지표) | 0.802 ± 0.002 | 0.662 ± 0.024 | −0.140 |
| ROC-AUC | 0.937 ± 0.002 | 0.888 ± 0.007 | −0.048 |
| Precision (임계값 0.5) | 0.818 ± 0.041 | 0.724 ± 0.063 | −0.093 |
| Recall (임계값 0.5) | 0.737 ± 0.001 | 0.590 ± 0.017 | −0.147 |
| F1 (임계값 0.5) | 0.775 ± 0.018 | 0.649 ± 0.026 | −0.126 |
| F1 (검증 구간 최적 임계값) | 0.797 ± 0.019 | 0.595 ± 0.025 | −0.202 |
| 시드별 PR-AUC 범위 | 0.799 ~ 0.804 | 0.626 ~ 0.693 | |
| 시간 단계 35~42 평균 F1 | 0.885 | 0.596 | −0.289 |
| 시간 단계 43~49 평균 F1 | 0.034 | 0.025 | −0.009 |
| 시간 단계 35~42 PR-AUC | 0.921 ± 0.001 | 0.782 ± 0.019 | −0.139 |
| 시간 단계 43~49 PR-AUC | 0.046 ± 0.001 | 0.040 ± 0.004 | −0.006 |
| 학습 시간 | 5.4초 (Colab CPU) | 6.2초 (Colab L4 GPU) | 환경이 달라 비교 불가 |
| 추론 시간 | 0.059초 (Colab CPU) | 0.050초 (Colab L4 GPU) | 환경이 달라 비교 불가 |

- 차이는 표에 적힌 값 기준입니다. ± 는 시드 5개의 표준편차입니다.
- 시간 단계별 F1은 검증 구간 최적 임계값 기준이고, 두 모델 모두 평균을 시드와 단계별로 낸 값입니다 (`results/timestep_f1_xgb.csv`, `results/timestep_f1_gcn.csv`).
- 속도는 같은 환경(같은 GPU 또는 CPU)에서 다시 재야 비교할 수 있습니다.
- 구간별 PR-AUC는 `results/pred/`의 예측값에서 해당 구간의 테스트 노드만 모아 시드마다 계산한 평균 ± 표준편차입니다. 무작위 예측의 PR-AUC 기준선은 그 구간의 불법 비율입니다 (35~42: 0.092, 43~49: 0.025).
- **43~49 구간은 두 모델 모두 사실상 구분 능력이 없습니다.** Elliptic 원논문(Weber et al., 2019)는 time step 43무렵에 다크마켓(불법 거래 시장)이 폐쇄되는 일이 있었다고함. 그 뒤로 학습된 모델들이 새로운 불법 패턴을 잡아내지 못 할 가능성이 있기에, 평가를 35-42와 43-49로 나누어서 결과를 출력함. >> 결과값은 35-42 구간으로 해석


### 테스트 구간 시간 단계별 불법 거래 수

라벨이 있는 노드 기준입니다. 43~46 단계는 불법이 아주 적어(45: 5건, 46: 2건) F1 같은 지표가 건수 몇 개에 크게 흔들립니다. 하지만 **48~49 단계는 불법이 36건, 56건(7.6%, 11.8%)으로 충분한데도 F1이 거의 0**이므로, 43 이후의 성능 붕괴는 건수 부족만이 아니라 거래 패턴 자체의 변화(분포 변화)가 주된 원인으로 보입니다.

| 시간 단계 | 라벨 있는 노드 | 불법 | 불법 비율 |
|---|---|---|---|
| 35 | 1,341 | 182 | 13.6% |
| 36 | 1,708 | 33 | 1.9% |
| 37 | 498 | 40 | 8.0% |
| 38 | 756 | 111 | 14.7% |
| 39 | 1,183 | 81 | 6.8% |
| 40 | 1,211 | 112 | 9.2% |
| 41 | 1,132 | 116 | 10.2% |
| 42 | 2,154 | 239 | 11.1% |
| 43 | 1,370 | 24 | 1.8% |
| 44 | 1,591 | 24 | 1.5% |
| 45 | 1,221 | 5 | 0.4% |
| 46 | 712 | 2 | 0.3% |
| 47 | 846 | 22 | 2.6% |
| 48 | 471 | 36 | 7.6% |
| 49 | 476 | 56 | 11.8% |
| **35~42 합계** | 9,983 | 914 | 9.2% |
| **43~49 합계** | 6,687 | 169 | 2.5% |

## GNN 조: 사용법
```python
from common.data import load_data, to_pyg          # torch/PyG 필요
from common.metrics import evaluate, best_threshold, evaluate_by_timestep
from common.utils import set_seed, Timer, save_rows, save_array

d = load_data("data/elliptic")          # standardize=True → d.x_std (GNN 입력), d.x (원본)
g = to_pyg(d)                           # torch_geometric Data: x(표준화), edge_index(무방향), y, time_step, train/val/test_mask
```
- 손실·평가는 `train_mask` / `val_mask` / `test_mask` 노드만 사용 (미분류 노드는 이웃으로만 쓰임).
- 클래스 가중치는 `[1, (학습 구간 정상 수)/(학습 구간 불법 수)]` (XGBoost와 동일).
- 모델은 `(logits, emb)`를 반환하고 임베딩은 64차원으로 통일합니다 (계획서 4장).
- 검증 구간으로 조기 종료·튜닝·`best_threshold` → Train+Val(1~34) 재학습 → 테스트 1회 평가.
- 모델 코드는 `models/`에 브랜치(`gcn`, `sage`, `gat`)로 올리고 Pull Request로 합칩니다.

### GCN 실행 (구현 완료)
`common/train_gnn.py`가 학습·조기 종료·재학습·평가·저장까지 모두 처리합니다. 모델은 `(logits, emb)`만 반환하면 됩니다.
```bash
python -m models.gcn --data-root data/elliptic --seeds 0 --max-epochs 5     # 빠른 점검
python -m models.gcn --data-root data/elliptic                              # 시드 0~4 본 학습
```

- 학습 절차: Train(1~29)으로 학습하고 검증 PR-AUC로 조기 종료 → 같은 에폭 수로 Train+Val(1~34) 재학습 → 테스트 1회 평가 (XGBoost와 동일한 규칙).
- 하이퍼파라미터는 계획서 4장 기본값(lr 0.005, weight decay 5e-4, 최대 300 에폭, patience 30, dropout 0.5)이고 `--lr`, `--dropout` 등으로 바꿀 수 있습니다.
- 코드의 기본값은 위 기본값 그대로입니다. **튜닝 후 최종 결과(GCN 결과 섹션)는 아래 인자로 재현합니다.**
```bash
python -m models.gcn --data-root data/elliptic --lr 0.01 --dropout 0.5 --weight-decay 5e-4 --patience 100 --max-epochs 500
```

## 구조
```
common/        data.py (로더·분할) / metrics.py (평가) / utils.py (시드·시간·저장) / train_gnn.py (GNN 공통 학습 루프)
baselines/     xgb_baseline.py
models/        gcn.py ✅ / (GNN 조) sage.py gat.py
hybrid/        (합치기 단계) skip_gcn.py
analysis/      (합치기 단계) shap_compare.py error_analysis.py
scripts/       verify_data.py, collect_results.py, make_synthetic.py
tests/         pytest -q  (가짜 Elliptic 형식 데이터로 공통 기반 검증)
results/       metrics/ pred/ emb/(git 제외)
```
