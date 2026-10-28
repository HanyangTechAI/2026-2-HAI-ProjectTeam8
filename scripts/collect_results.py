"""모든 모델의 결과를 한 표로 취합 (마린·GCN조: 11/12 1차 비교표용).

  python scripts/collect_results.py

results/metrics/*.csv → results/metrics_all.csv, results/summary.csv(테스트 구간 시드 평균±표준편차)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import RESULTS_DIR  # noqa: E402


def collect(results_dir=RESULTS_DIR) -> pd.DataFrame:
    results_dir = Path(results_dir)
    files = sorted((results_dir / "metrics").glob("*.csv"))
    if not files:
        raise SystemExit("results/metrics/ 에 결과 파일이 없습니다.")
    df = pd.concat([pd.read_csv(f, dtype={"threshold_type": str}) for f in files], ignore_index=True)
    df.to_csv(results_dir / "metrics_all.csv", index=False)

    test = df[df["split"] == "test"]
    cols = ["pr_auc", "roc_auc", "precision", "recall", "f1", "train_sec", "infer_sec"]
    g = test.groupby(["model", "threshold_type"])[cols]
    summary = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).join(
        test.groupby(["model", "threshold_type"])["seed"].nunique().rename("n_seeds")).reset_index()
    summary.to_csv(results_dir / "summary.csv", index=False)
    return summary


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(collect().round(4).to_string(index=False))
