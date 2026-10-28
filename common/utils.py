"""공통 유틸: 시드 고정, 시간 측정, 결과 저장 형식.

결과 저장 규칙 (git 충돌 방지를 위해 모델별 파일로 분리):
  results/metrics/{model}.csv           한 줄 = (model, seed, split, threshold_type)
  results/pred/pred_{model}_seed{n}.npy (N,) float32 불법 확률, 노드 순서 = load_data() 순서
  results/emb/emb_{model}_seed{n}.npy   (N, 64) float32 GNN 임베딩 (용량이 커서 git 제외, 공유 드라이브 사용)
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

METRIC_COLUMNS = [
    "model", "seed", "split", "threshold_type", "threshold",
    "pr_auc", "roc_auc", "precision", "recall", "f1",
    "train_sec", "infer_sec", "n_params", "device", "notes",
]
_KEYS = ["model", "seed", "split", "threshold_type"]  # 같은 키의 행은 다시 실행하면 덮어쓴다


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class Timer:
    """with Timer() as t: ... ; t.sec  (GPU 사용 시 동기화 후 측정)"""

    @staticmethod
    def _sync():
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass

    def __enter__(self):
        self._sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._sync()
        self.sec = time.perf_counter() - self._t0


def save_rows(model: str, rows: list[dict], results_dir: str | Path | None = None) -> Path:
    """results/metrics/{model}.csv에 행을 저장. 같은 (model, seed, split, threshold_type) 행은 교체한다."""
    d = Path(results_dir or RESULTS_DIR) / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{model}.csv"
    new = pd.DataFrame(rows).reindex(columns=METRIC_COLUMNS)
    if path.exists():
        old = pd.read_csv(path, dtype={"threshold_type": str})
        merged = old.merge(new[_KEYS].drop_duplicates(), on=_KEYS, how="left", indicator=True)
        old = old[(merged["_merge"] == "left_only").to_numpy()]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    return path


def save_array(arr: np.ndarray, kind: str, model: str, seed: int, results_dir: str | Path | None = None) -> Path:
    """kind: 'pred' 또는 'emb'. 파일명 규칙 통일."""
    assert kind in ("pred", "emb")
    d = Path(results_dir or RESULTS_DIR) / kind
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{kind}_{model}_seed{seed}.npy"
    np.save(path, arr.astype(np.float32))
    return path
