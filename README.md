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
4. `common/`을 바꿔야 하면 마린에게 요청하고 전원에게 공지합니다.
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

## 실행 환경 (헷갈리지 않게)

| 환경 | 용도 | xgboost | 비고 |
|---|---|---|---|
| 마린 Mac (Apple Silicon, CPU, conda 환경 `hai`) | **공식 기준선 결과** (`results/`) | 3.4.2 | PR-AUC 0.798, F1 0.78 |
| Colab (CPU) | 재현 점검(`results_colab/`), GNN 학습 | 3.4.1 (Colab pip에 3.4.2가 아직 없음) | `notebooks/colab_xgboost.ipynb` · PR-AUC 0.802, F1 0.77~0.80 |

- `requirements.txt`는 `xgboost>=3.4.1,<3.5`입니다. 환경마다 패치 버전이 달라 **같은 코드·시드여도 소수점 둘째 자리쯤 차이**가 날 수 있습니다 (튜닝에서 뽑히는 설정이 달라질 수 있기 때문).
- **학습·추론 시간은 환경마다 완전히 다르므로 같은 환경에서 잰 값끼리만 비교합니다.** 보고서 표에는 환경(Mac/Colab, CPU/GPU)을 반드시 적습니다.
- 정확도 비교의 기준선은 `results/`(Mac, xgboost 3.4.2)입니다. Colab 재현 결과는 따로 `results_colab/`에 저장해 덮어쓰지 않습니다.
- GNN 조는 Colab에서 학습합니다 (`torch`, `torch-geometric` 설치). GPU 종류(T4 등)는 결과 `notes`나 보고서에 적어 주세요.

### Colab에서 돌리는 법 (요약)
1. 구글 드라이브 `내 드라이브/Colab Notebooks/elliptic/`에 CSV 3개를 올립니다.
2. Colab에서 `notebooks/colab_xgboost.ipynb`를 열고 위에서부터 차례로 실행합니다 (repo가 private이면 GitHub 토큰 필요).
3. 결과는 `results_colab/`에 저장되고 드라이브에도 백업됩니다.

### Mac(conda)에서 자주 막히는 점
- `ModuleNotFoundError: xgboost` 또는 `libomp.dylib` 오류 → Homebrew가 아니라 **conda-forge**로 설치: `conda activate hai && conda install -c conda-forge xgboost`
- `python`이 conda 환경이 아닌 다른 파이썬(`python3` alias 등)을 가리킬 수 있습니다. `which python`이 `.../envs/hai/bin/python`인지 확인하고, 아니면 `$CONDA_PREFIX/bin/python`으로 직접 실행합니다.

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

> 구간마다 불법 비율이 다릅니다 (train 10.9% / val 16.8% / test 6.5%). 테스트 구간에서는 불법이 더 드물어 PR-AUC의 기준선(무작위 예측)이 낮고, 검증 구간에서 고른 임계값이 테스트에는 맞지 않을 수 있습니다. 보고서에서 이 점을 함께 해석합니다.

## XGBoost 베이스라인 결과

테스트 구간(시간 단계 35~49), 시드 0~4 평균 ± 표준편차. 최종 모델은 Train+Val(1~34)로 다시 학습했고, 테스트는 1회만 평가했습니다.

| 지표 | 임계값 0.5 | 검증 구간 최적 임계값 |
|---|---|---|
| PR-AUC | 0.798 ± 0.002 | 0.798 ± 0.002 |
| ROC-AUC | 0.930 ± 0.005 | 0.930 ± 0.005 |
| Precision | 0.827 ± 0.026 | 0.840 ± 0.040 |
| Recall | 0.737 ± 0.003 | 0.735 ± 0.005 |
| **F1 (불법)** | **0.779 ± 0.012** | **0.784 ± 0.016** |

- 학습 0.85초 · 추론 약 5ms. **환경: Mac(Apple Silicon, CPU), xgboost 3.4.2.** Colab/GPU 수치와 직접 비교하지 말 것 (위 [실행 환경](#실행-환경-헷갈리지-않게) 참고). Colab 재현 결과는 아래에 정리했습니다.
- 설정: `max_depth=8, learning_rate=0.1, min_child_weight=5, subsample=0.8, colsample_bytree=1.0`, 학습 구간 정상/불법 비로 `scale_pos_weight`. 튜닝 30회는 검증 PR-AUC 기준(검증 PR-AUC 0.99). 자세한 기록은 `results/xgb_tuning.csv`.
- 무작위 예측의 PR-AUC 기준선은 테스트 불법 비율인 0.065입니다. 원논문(Weber 2019)의 랜덤포레스트 불법 F1 약 0.8과 비슷한 수준입니다.

**GNN 조가 넘어야 할 기준:** PR-AUC **0.798**, F1 **0.78**. 같은 지표·같은 분할·같은 시드로 비교합니다.

### Colab 재현 결과 (xgboost 3.4.1, CPU)

같은 코드·같은 시드(0~4)를 Colab에서 다시 돌려 환경 차이를 확인했습니다 (`results_colab/`).

| 지표 (테스트, 평균 ± 표준편차) | Mac (3.4.2) | Colab (3.4.1) | 차이 |
|---|---|---|---|
| PR-AUC | 0.798 ± 0.002 | 0.802 ± 0.002 | +0.004 |
| ROC-AUC | 0.930 ± 0.005 | 0.937 ± 0.002 | +0.007 |
| F1, 임계값 0.5 | 0.779 ± 0.012 | 0.775 ± 0.018 | −0.004 |
| F1, 검증 구간 최적 임계값 | 0.784 ± 0.016 | 0.797 ± 0.019 | +0.013 |
| Precision (0.5 / 최적) | 0.827 / 0.840 | 0.818 / 0.874 | |
| Recall (0.5 / 최적) | 0.737 / 0.735 | 0.737 / 0.733 | |
| 학습 시간(초) | 0.85 | 5.4 | Colab CPU가 약 6배 느림 |
| 추론 시간(초) | 0.005 | 0.059 | 약 12배 느림 |

**해석**
- **정확도는 사실상 같은 수준입니다.** PR-AUC 차이(0.004)는 표준편차 약 2배 수준이고, F1 차이(−0.004~+0.013)는 표준편차(0.012~0.019) 안에 들어옵니다. 베이스라인은 **PR-AUC ≈ 0.80, F1 ≈ 0.78~0.80**이라고 보고합니다. GNN은 이 범위를 넘는지로 비교합니다.
- **환경이 달라도 구조는 같습니다.** 시간 단계 43부터 F1이 0~0.14로 무너지는 패턴(35~42는 0.77~0.97)이 Mac과 같습니다.
- **임계값은 불안정합니다.** 검증 구간에서 고른 최적 임계값이 시드마다 0.53~0.73으로 흔들려서, F1(최적 임계값)의 표준편차가 큽니다. 비교는 임계값에 의존하지 않는 **PR-AUC를 주 지표**로 하고 F1은 보조로 봅니다.
- **튜닝에서 뽑힌 설정이 환경마다 다릅니다.** Mac은 `colsample_bytree=1.0`, Colab은 `0.7`이 뽑혔고 (둘 다 `max_depth=8`, `learning_rate=0.1`, `min_child_weight=5`) 최종 트리 수도 시드마다 85~261개로 달랐습니다. 검증 PR-AUC가 0.99 근처에서 거의 비슷해 순위가 쉽게 바뀌기 때문입니다.
- **특징 중요도(gain) 순위는 환경마다 다릅니다.** 상위 5개가 Mac은 52, 45, 39, 66, 89, Colab은 40, 52, 99, 45, 54로 겹치는 건 52와 45뿐입니다. SHAP 비교에서 특징 번호를 근거로 말할 때는 한 환경의 결과만 쓰고 환경을 밝힙니다.
- **시간은 같은 환경끼리만 비교합니다.** GNN과 속도를 비교할 때는 Colab 값(학습 5.4초, 추론 0.059초)을 XGBoost 기준으로 씁니다.

### 결과 파일 위치
| 파일 | 내용 |
|---|---|
| `results/summary.csv` | 모델·임계값별 평균/표준편차 요약 (`scripts/collect_results.py`로 재생성) |
| `results/metrics/xgb.csv` | 시드·분할·임계값별 상세 |
| `results/pred/pred_xgb_seed{n}.npy` | 시드별 불법 확률 (오류 분석·하이브리드용) |
| `results/timestep_f1_xgb.csv` | 시간 단계별 F1 |
| `results/xgb_feature_importance.csv`, `results/xgb_tuning.csv` | 특징 중요도, 튜닝 기록 |
| `results_colab/` | Colab 재현 결과 (같은 형식, `summary.csv` 포함) |

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

## 구조
```
common/        data.py (로더·분할) / metrics.py (평가) / utils.py (시드·시간·저장)
baselines/     xgb_baseline.py
models/        (GNN 조) gcn.py sage.py gat.py
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
