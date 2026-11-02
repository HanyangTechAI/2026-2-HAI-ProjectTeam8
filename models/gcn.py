"""GCN (Kipf & Welling 2017).  계획서 4장 공통 사양: 2층 conv + 선형 분류기, 임베딩 64차원.

  python -m models.gcn --data-root data/elliptic --seeds 0 1 2 3 4

GraphSAGE / GAT는 이 파일을 복사해서 conv 층(SAGEConv / GATConv)만 바꾸면 된다.
  - GAT: heads=4, 헤드당 16차원 → concat 64차원 (conv2도 동일하게 64차원으로 맞출 것)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.train_gnn import main  # noqa: E402


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)   # 이 층의 출력 = 임베딩 (64차원)
        self.head = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        emb = self.conv2(h, edge_index)
        logits = self.head(F.dropout(F.relu(emb), p=self.dropout, training=self.training))
        return logits, emb


if __name__ == "__main__":
    main("gcn", lambda in_dim, a: GCN(in_dim, a.hidden, a.dropout))
