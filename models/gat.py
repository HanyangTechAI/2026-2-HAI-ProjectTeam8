"""GAT (Veličković et al. 2018).  계획서 4장 공통 사양: 2층 conv + 선형 분류기, 임베딩 64차원.

  python -m models.gat --data-root data/elliptic --seeds 0 1 2 3 4
  python -m models.gat --data-root data/elliptic --seeds 0 --heads 8 --results-dir results/tune/gat_h8   # 튜닝 예

models/gcn.py를 복사해서 conv 층만 GATConv로 바꾼 파일.
  - heads=8 기본(기존 GAT 최종 설정: heads 8, lr 0.01, dropout 0.3), 헤드당 8차원 → concat 64차원.
  - hybrid/make_oof_emb.py는 GAT(in_dim, hidden, dropout)만 넘기므로 이 기본값(8)이 그대로 쓰인다.
  - 헤드 8개로 바꿔도 헤드당 8차원 × 8 = 64차원이라 임베딩 크기는 그대로.
  - GATConv의 dropout 인자 = 어텐션 계수(이웃 연결)에 거는 드롭아웃.
  - --heads는 common/train_gnn.py에 없는 GAT 전용 인자라, 여기서 먼저 읽고 나머지 인자만 main()에 넘긴다
    (공통 코드는 수정하지 않음).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.train_gnn import main  # noqa: E402


class GAT(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.5, heads: int = 8):
        super().__init__()
        assert hidden % heads == 0, "hidden은 heads로 나누어떨어져야 함"
        per_head = hidden // heads
        self.conv1 = GATConv(in_dim, per_head, heads=heads, dropout=dropout)   # → per_head × heads = 64
        self.conv2 = GATConv(hidden, per_head, heads=heads, dropout=dropout)   # 이 층의 출력 = 임베딩 (64차원)
        self.head = nn.Linear(hidden, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.elu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        emb = self.conv2(h, edge_index)
        logits = self.head(F.dropout(F.elu(emb), p=self.dropout, training=self.training))
        return logits, emb


if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--heads", type=int, default=8, help="어텐션 헤드 수 (튜닝: 4 또는 8)")
    gat_args, rest = pre.parse_known_args()
    # GAT 기본값: patience 100, 최대 500에폭 (실행할 때 --patience / --max-epochs를 주면 그 값이 우선)
    if not any(a.startswith("--patience") for a in rest):
        rest += ["--patience", "100"]
    if not any(a.startswith("--max-epochs") for a in rest):
        rest += ["--max-epochs", "500"]
    print(f"GAT heads={gat_args.heads}, args={rest}")
    main("gat", lambda in_dim, a: GAT(in_dim, a.hidden, a.dropout, gat_args.heads), argv=rest)
