# 2026-2-HAI-ProjectTeam8 · GNN 기반 가상자산 이상거래 탐지

XGBoost vs GCN · GraphSAGE · GAT 비교, 그리고 Skip-GCN 하이브리드. 계획서 3장의 공통 기반을 구현한 저장소입니다.

- 계획서: [`docs/프로젝트_계획서.pdf`](docs/프로젝트_계획서.pdf) · 타임라인: [`docs/타임라인.pdf`](docs/타임라인.pdf)
- 계획서와 코드가 다르면 **이 README와 코드가 우선**입니다 (로더는 PyG가 아니라 원본 CSV를 직접 읽고, 결과 CSV는 `results/metrics/{model}.csv`로 모델별 분리).
- 일정: 11/1 공통 기반 완료 ✅ → 11/12 모델 3종 구현 → 11/19 합치기 → 11/26 마무리 → 11/29 최종 발표
- 현재 상태: 데이터 검증과 XGBoost 베이스라인까지 완료. **GNN 조는 아래 [XGBoost 베이스라인 결과](#xgboost-베이스라인-결과)를 넘는지가 비교 기준입니다.**

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
- Colab: `notebooks/colab_gcn.ipynb` (T4 GPU 권장). 결과는 `results/metrics/gcn.csv`, `results/pred/`, `results/emb/`(git 제외)에 저장됩니다.
- **GraphSAGE / GAT**: `models/gcn.py`를 복사해 `GCNConv`를 `SAGEConv` / `GATConv`로 바꾸고 `main("sage", ...)` / `main("gat", ...)`로 이름만 바꾸면 됩니다 (GAT는 heads=4, 헤드당 16차원 → 64차원).
- 학습 절차: Train(1~29)으로 학습하고 검증 PR-AUC로 조기 종료 → 같은 에폭 수로 Train+Val(1~34) 재학습 → 테스트 1회 평가 (XGBoost와 동일한 규칙).
- 하이퍼파라미터는 계획서 4장 기본값(lr 0.005, weight decay 5e-4, 최대 300 에폭, patience 30, dropout 0.5)이고 `--lr`, `--dropout` 등으로 바꿀 수 있습니다.

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

## 가짜 데이터로 파이프라인 점검
실제 데이터가 없는 환경에서도 코드가 도는지 확인할 수 있습니다. **성능 수치를 보고서에 쓰면 안 됩니다.**
```bash
python scripts/make_synthetic.py data/synthetic
python scripts/verify_data.py data/synthetic --lenient
pytest -q
```
