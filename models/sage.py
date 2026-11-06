"""GraphSAGE (Hamilton et al. 2017).  계획서 4장 공통 사양: 2층 conv + 선형 분류기, 임베딩 64차원.

  python -m models.sage --data-root data/elliptic --seeds 0 1 2 3 4

GCN과의 차이: 이웃 평균(aggr=mean)과 자기 자신의 표현을 별도 가중치로 변환해 더한다
(GCN은 자기 자신도 이웃처럼 정규화 평균에 섞음). 전체 그래프(full-batch)로 학습하며,
이웃 샘플링(NeighborLoader)은 메모리 문제가 있을 때만 쓴다(계획서 4.1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.train_gnn import main  # noqa: E402


class SAGE(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.5, aggr: str = "mean"):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden, aggr=aggr)
        self.conv2 = SAGEConv(hidden, hidden, aggr=aggr)   # 이 층의 출력 = 임베딩 (64차원)
        self.head = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        emb = self.conv2(h, edge_index)
        logits = self.head(F.dropout(F.relu(emb), p=self.dropout, training=self.training))
        return logits, emb


if __name__ == "__main__":
    main("sage", lambda in_dim, a: SAGE(in_dim, a.hidden, a.dropout))
