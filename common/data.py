"""공통 데이터 로딩 (Elliptic Bitcoin Dataset).

모든 조(GCN/SAGE/GAT)와 XGBoost가 이 파일의 load_data()만 사용한다.
원본 CSV 3개를 직접 읽으므로 PyG 내부 구현에 의존하지 않는다.
  elliptic_txs_features.csv  (헤더 없음: txId, 시간 단계, 피처 165개)
  elliptic_txs_edgelist.csv  (헤더: txId1, txId2)
  elliptic_txs_classes.csv   (헤더: txId, class → 'unknown' / '1'(불법) / '2'(정상))

기준 (변경 금지, 계획서 3장):
  - 노드 순서: txId 오름차순 (PyG EllipticBitcoinDataset과 동일)
  - y: 0=정상, 1=불법(양성), 2=미분류
  - 분할(시간 단계): train 1~29 / val 30~34 / test 35~49, 라벨 있는 노드만 마스크에 포함
  - 시간 단계는 분할에만 사용하고 모델 입력 피처에서는 제외
  - 간선: 무방향·중복 제거·자기 루프 제거 (모든 GNN에 동일 적용)
  - GNN 입력 표준화: 시간 단계 <= 34 노드의 평균/표준편차 사용 (미래 정보 미사용)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

TRAIN_END = 29   # train: 1~29
VAL_END = 34     # val: 30~34, test: 35~49
N_STEPS = 49

FEATURE_FILE = "elliptic_txs_features.csv"
EDGE_FILE = "elliptic_txs_edgelist.csv"
CLASS_FILE = "elliptic_txs_classes.csv"

# 원본 데이터셋의 알려진 값 (Weber et al. 2019 / PyG 문서). scripts/verify_data.py에서 대조한다.
EXPECTED = dict(
    n_nodes=203_769, n_edges=234_355, n_features=165, n_steps=49,
    n_illicit=4_545, n_licit=42_019, n_unknown=157_205,
)

_CLASS_MAP = {"unknown": 2, "1": 1, "2": 0}  # '1'=불법 → 1, '2'=정상 → 0


@dataclass
class EllipticData:
    x: np.ndarray                 # (N, F) float32, 원본 값 (XGBoost용)
    x_std: Optional[np.ndarray]   # (N, F) float32, 표준화 값 (GNN용). standardize=False면 None
    edge_index: np.ndarray        # (2, E') int64, 무방향(양방향 저장)·중복/자기루프 제거
    y: np.ndarray                 # (N,) int64, 0 정상 / 1 불법 / 2 미분류
    time_step: np.ndarray         # (N,) int64, 1~49
    node_ids: np.ndarray          # (N,) int64, 원본 txId (오름차순)
    train_mask: np.ndarray        # (N,) bool
    val_mask: np.ndarray
    test_mask: np.ndarray
    n_edges_raw: int              # 원본 방향 간선 수

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.x.shape[1])

    @property
    def labeled_mask(self) -> np.ndarray:
        return self.y != 2


def _find_raw_dir(root: str | os.PathLike) -> Path:
    root = Path(root)
    for cand in (root, root / "raw"):
        if (cand / FEATURE_FILE).exists():
            return cand
    raise FileNotFoundError(
        f"'{FEATURE_FILE}'을(를) {root} 또는 {root / 'raw'} 에서 찾지 못했습니다.\n"
        "README의 '데이터 준비' 절차(Kaggle 다운로드 또는 PyG로 1회 내려받기)를 먼저 수행하세요."
    )


def load_data(root: str | os.PathLike = "data/elliptic", standardize: bool = True) -> EllipticData:
    """Elliptic 데이터를 읽어 공통 기준(분할·순서·간선)으로 반환한다."""
    raw = _find_raw_dir(root)

    # 피처: 헤더 없음. 0열 txId, 1열 시간 단계, 2열~ 피처
    feat = pd.read_csv(raw / FEATURE_FILE, header=None)
    feat = feat.sort_values(0, kind="stable").reset_index(drop=True)
    node_ids = feat[0].to_numpy(dtype=np.int64)
    time_step = feat[1].to_numpy(dtype=np.int64)
    x = feat.iloc[:, 2:].to_numpy(dtype=np.float32)
    del feat
    if len(np.unique(node_ids)) != len(node_ids):
        raise ValueError("txId가 중복되었습니다.")

    # 라벨: unknown→2, '1'(불법)→1, '2'(정상)→0
    cls = pd.read_csv(raw / CLASS_FILE, dtype={"class": str}).set_index("txId")
    y_series = cls["class"].map(_CLASS_MAP).reindex(node_ids)
    if y_series.isna().any():
        raise ValueError("라벨을 찾지 못한 노드가 있거나 알 수 없는 class 값이 있습니다.")
    y = y_series.to_numpy(dtype=np.int64)

    # 간선: txId → 노드 인덱스. 무방향·중복·자기루프 제거
    edges = pd.read_csv(raw / EDGE_FILE).to_numpy(dtype=np.int64)
    n_edges_raw = int(edges.shape[0])
    src = np.searchsorted(node_ids, edges[:, 0])
    dst = np.searchsorted(node_ids, edges[:, 1])
    ok = (src < len(node_ids)) & (dst < len(node_ids))
    if not ok.all() or not (node_ids[src] == edges[:, 0]).all() or not (node_ids[dst] == edges[:, 1]).all():
        raise ValueError("간선의 txId가 피처 파일에 없는 노드를 가리킵니다.")
    keep = src != dst
    a, b = np.minimum(src[keep], dst[keep]), np.maximum(src[keep], dst[keep])
    pair = np.unique(a * np.int64(len(node_ids)) + b)
    a, b = pair // len(node_ids), pair % len(node_ids)
    edge_index = np.vstack([np.concatenate([a, b]), np.concatenate([b, a])]).astype(np.int64)

    labeled = y != 2
    train_mask = labeled & (time_step <= TRAIN_END)
    val_mask = labeled & (time_step > TRAIN_END) & (time_step <= VAL_END)
    test_mask = labeled & (time_step > VAL_END)

    x_std = None
    if standardize:
        ref = x[time_step <= VAL_END]
        mu = ref.mean(axis=0)
        sd = ref.std(axis=0)
        sd[sd < 1e-8] = 1.0
        x_std = ((x - mu) / sd).astype(np.float32)

    return EllipticData(
        x=x, x_std=x_std, edge_index=edge_index, y=y, time_step=time_step, node_ids=node_ids,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, n_edges_raw=n_edges_raw,
    )


def get_tabular(data: EllipticData, mask: np.ndarray, standardized: bool = False):
    """XGBoost 등 표 형태 모델용 (X, y). 기본은 원본 값."""
    X = data.x_std if standardized else data.x
    if X is None:
        raise ValueError("standardize=True로 load_data()를 호출해야 x_std를 쓸 수 있습니다.")
    return X[mask], data.y[mask]


def split_summary(data: EllipticData) -> pd.DataFrame:
    """분할별 라벨 있는 노드 수 / 불법·정상 수 / 불법 비율 (README 기록용)."""
    rows = []
    for name, m in [("train (1~29)", data.train_mask), ("val (30~34)", data.val_mask), ("test (35~49)", data.test_mask)]:
        yy = data.y[m]
        n_ill, n_lic = int((yy == 1).sum()), int((yy == 0).sum())
        rows.append(dict(split=name, labeled_nodes=int(m.sum()), illicit=n_ill, licit=n_lic,
                         illicit_ratio=round(n_ill / max(1, n_ill + n_lic), 4)))
    return pd.DataFrame(rows)


def to_pyg(data: EllipticData, standardized: bool = True):
    """GNN 조용: EllipticData → torch_geometric.data.Data (torch/PyG 설치 필요, 지연 import).

    x는 표준화된 값(기본), 마스크/라벨/시간 단계/edge_index는 그대로 사용한다.
    """
    import torch
    from torch_geometric.data import Data

    x = data.x_std if standardized else data.x
    if x is None:
        raise ValueError("standardize=True로 load_data()를 호출해야 합니다.")
    return Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(data.edge_index),
        y=torch.from_numpy(data.y),
        time_step=torch.from_numpy(data.time_step),
        train_mask=torch.from_numpy(data.train_mask),
        val_mask=torch.from_numpy(data.val_mask),
        test_mask=torch.from_numpy(data.test_mask),
    )
