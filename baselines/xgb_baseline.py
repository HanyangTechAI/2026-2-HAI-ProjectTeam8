"""XGBoost 베이스라인 (그래프 정보 없음, 원본 피처만 사용).

기준 (계획서 3.6):
  - 불균형: scale_pos_weight = (학습 구간 정상 수) / (학습 구간 불법 수)
  - 튜닝: 검증 구간(30~34) PR-AUC 기준 랜덤 탐색. 테스트 구간은 튜닝·조기종료·임계값 결정에 쓰지 않는다.
  - 시드별: ① train으로 조기종료 학습 → best_iteration, 검증 확률로 임계값(best_threshold) 결정
            ② train+val(1~34)을 best_iteration+1 트리로 재학습 → 테스트 1회 평가
  - train_sec = ②(최종 재학습) 소요 시간, infer_sec = 테스트 노드 예측 시간(5회 중앙값)
  - 저장: results/metrics/xgb.csv, results/pred/pred_xgb_seed{n}.npy, results/timestep_f1_xgb.csv, results/xgb_*.csv

실행:  python -m baselines.xgb_baseline --data-root data/elliptic
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from common.data import EllipticData, get_tabular, load_data, split_summary
from common.metrics import best_threshold, evaluate, evaluate_by_timestep
from common.utils import RESULTS_DIR, Timer, save_array, save_rows, set_seed

MODEL = "xgb"
BASE_PARAMS = dict(n_estimators=1000, learning_rate=0.05, max_depth=6,
                   subsample=0.8, colsample_bytree=0.8, min_child_weight=1)
SPACE = dict(max_depth=[4, 6, 8], learning_rate=[0.03, 0.05, 0.1], min_child_weight=[1, 5],
             subsample=[0.7, 0.8, 1.0], colsample_bytree=[0.7, 0.8, 1.0])


def make_clf(params: dict, spw: float, seed: int, n_jobs: int, early_stopping: bool) -> xgb.XGBClassifier:
    kw = dict(params)
    kw.update(scale_pos_weight=spw, eval_metric="aucpr", tree_method="hist",
              random_state=seed, n_jobs=n_jobs)
    if early_stopping:
        kw["early_stopping_rounds"] = 30
    return xgb.XGBClassifier(**kw)


def pos_weight(y_train: np.ndarray) -> float:
    return float((y_train == 0).sum() / max(1, (y_train == 1).sum()))


def tune(data: EllipticData, trials: int, n_jobs: int, seed: int = 0) -> tuple[dict, pd.DataFrame]:
    """검증 PR-AUC 기준 랜덤 탐색. train/val만 사용한다."""
    Xtr, ytr = get_tabular(data, data.train_mask)
    Xva, yva = get_tabular(data, data.val_mask)
    spw = pos_weight(ytr)
    grid = [dict(zip(SPACE, v)) for v in itertools.product(*SPACE.values())]
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(grid), size=min(trials, len(grid)), replace=False)
    log = []
    for i, gi in enumerate(picks):
        params = {**BASE_PARAMS, **grid[gi]}
        clf = make_clf(params, spw, seed, n_jobs, early_stopping=True)
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        score = float(average_precision_score(yva, clf.predict_proba(Xva)[:, 1]))
        log.append({**grid[gi], "val_pr_auc": score, "best_iteration": int(clf.best_iteration)})
        print(f"[tune {i + 1}/{len(picks)}] {grid[gi]} val PR-AUC={score:.4f}")
    log_df = pd.DataFrame(log).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
    best = {k: log_df.loc[0, k] for k in SPACE}
    best = {k: (int(v) if k in ("max_depth", "min_child_weight") else float(v)) for k, v in best.items()}
    return {**BASE_PARAMS, **best}, log_df


def run_seed(data: EllipticData, params: dict, seed: int, n_jobs: int, results_dir=None) -> list[dict]:
    set_seed(seed)
    Xtr, ytr = get_tabular(data, data.train_mask)
    Xva, yva = get_tabular(data, data.val_mask)
    Xte, yte = get_tabular(data, data.test_mask)
    spw = pos_weight(ytr)

    # ① 조기 종료 학습 → best_iteration, 검증 기반 임계값
    clf = make_clf(params, spw, seed, n_jobs, early_stopping=True)
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    n_trees = int(clf.best_iteration) + 1
    p_val = clf.predict_proba(Xva)[:, 1]
    thr = best_threshold(yva, p_val)

    # ② train+val 재학습 (고정 트리 수), 테스트 1회 평가
    trval = data.train_mask | data.val_mask
    Xall, yall = get_tabular(data, trval)
    final = make_clf({**params, "n_estimators": n_trees}, pos_weight(yall), seed, n_jobs, early_stopping=False)
    with Timer() as t_train:
        final.fit(Xall, yall, verbose=False)
    times = []
    for _ in range(5):
        with Timer() as t_inf:
            p_test = final.predict_proba(Xte)[:, 1]
        times.append(t_inf.sec)
    infer_sec = float(np.median(times))
    n_params = int(len(final.get_booster().trees_to_dataframe()))  # 트리 노드 수

    common = dict(model=MODEL, seed=seed, train_sec=t_train.sec, infer_sec=infer_sec,
                  n_params=n_params, device="cpu", notes=f"trees={n_trees}")
    rows = []
    for ttype, th in (("fixed_0.5", 0.5), ("val_opt", thr)):
        rows.append({**common, "split": "test", "threshold_type": ttype, **evaluate(yte, p_test, th)})
    # 검증 행은 임계값을 같은 구간에서 골랐으므로 참고용 (①의 모델 기준)
    rows.append({**{k: v for k, v in common.items() if k not in ("train_sec", "infer_sec", "n_params")},
                 "split": "val", "threshold_type": "val_opt", **evaluate(yva, p_val, thr)})

    save_array(final.predict_proba(data.x)[:, 1], "pred", MODEL, seed, results_dir)

    # 시간 단계별 F1 (테스트)
    ts = data.time_step[data.test_mask]
    by_step = evaluate_by_timestep(yte, p_test, ts, thr)
    step_df = pd.DataFrame([dict(model=MODEL, seed=seed, time_step=s, f1=f) for s, f in by_step.items()])
    out = Path(results_dir or RESULTS_DIR) / "timestep_f1_xgb.csv"
    if out.exists():
        old = pd.read_csv(out)
        old = old[old["seed"] != seed]
        step_df = pd.concat([old, step_df], ignore_index=True)
    step_df.to_csv(out, index=False)

    if seed == 0:  # 피처 중요도 (gain), 참고용
        gain = final.get_booster().get_score(importance_type="gain")  # {'f12': gain, ...} (사용된 피처만)
        imp = pd.DataFrame([{"feature_idx": int(k[1:]), "gain": v} for k, v in gain.items()])
        imp.sort_values("gain", ascending=False).to_csv(Path(results_dir or RESULTS_DIR) / "xgb_feature_importance.csv", index=False)

    print(f"[seed {seed}] test PR-AUC={rows[0]['pr_auc']:.4f}  F1@0.5={rows[0]['f1']:.4f}  "
          f"F1@val_opt={rows[1]['f1']:.4f} (thr={thr:.3f})  train={t_train.sec:.1f}s infer={infer_sec * 1000:.0f}ms")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/elliptic")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tune-trials", type=int, default=30)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)

    rdir = Path(args.results_dir or RESULTS_DIR)
    rdir.mkdir(parents=True, exist_ok=True)
    data = load_data(args.data_root, standardize=False)
    print(split_summary(data).to_string(index=False))

    t0 = time.time()
    params, log_df = tune(data, args.tune_trials, args.n_jobs)
    log_df.to_csv(rdir / "xgb_tuning.csv", index=False)
    print("선택된 설정:", {k: params[k] for k in SPACE}, f"(튜닝 {time.time() - t0:.0f}s)")

    all_rows = []
    for s in args.seeds:
        all_rows += run_seed(data, params, s, args.n_jobs, rdir)
    path = save_rows(MODEL, all_rows, rdir)
    print("저장:", path)


if __name__ == "__main__":
    main()
