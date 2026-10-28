"""테스트용 가짜 데이터 생성 (Elliptic과 같은 CSV 형식).

실제 Elliptic이 아니다. 로더·분할·평가·XGBoost 파이프라인이 동작하는지 점검하는 용도이며,
성능 수치를 보고서에 쓰면 안 된다.
  python scripts/make_synthetic.py data/synthetic --per-step 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic(out_dir, per_step: int = 200, n_features: int = 165, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_steps = 49
    N = n_steps * per_step
    tx_ids = rng.permutation(np.arange(10_000, 10_000 + N * 3))[:N]  # 순서가 섞인 txId

    step = np.repeat(np.arange(1, n_steps + 1), per_step)
    # 라벨: 미분류 77%, 나머지 중 불법 약 10%
    r = rng.random(N)
    cls = np.where(r < 0.77, "unknown", np.where(rng.random(N) < 0.10, "1", "2"))
    illicit = cls == "1"

    X = rng.normal(size=(N, n_features))
    X[illicit, :12] += 1.0  # 불법 노드는 일부 피처가 이동
    feat = np.column_stack([tx_ids, step, X])

    # 간선: 같은 시간 단계 안에서만, 불법끼리 연결 확률을 높임
    edges = []
    for s in range(1, n_steps + 1):
        idx = np.where(step == s)[0]
        m = int(len(idx) * 1.15)
        a = rng.choice(idx, size=m)
        b = rng.choice(idx, size=m)
        ill_idx = idx[illicit[idx]]
        if len(ill_idx) > 1:
            k = len(ill_idx)
            a = np.concatenate([a, rng.choice(ill_idx, size=k)])
            b = np.concatenate([b, rng.choice(ill_idx, size=k)])
        edges.append(np.column_stack([tx_ids[a], tx_ids[b]]))
    edges = np.vstack(edges)

    pd.DataFrame(feat).to_csv(out / "elliptic_txs_features.csv", header=False, index=False)
    pd.DataFrame(edges, columns=["txId1", "txId2"]).to_csv(out / "elliptic_txs_edgelist.csv", index=False)
    pd.DataFrame({"txId": tx_ids, "class": cls}).to_csv(out / "elliptic_txs_classes.csv", index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--per-step", type=int, default=200)
    a = ap.parse_args()
    print("생성:", make_synthetic(a.out_dir, a.per_step))
