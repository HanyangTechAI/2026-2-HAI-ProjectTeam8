"""GCN 학습 루프 스모크 테스트 (torch / torch_geometric 없으면 건너뜀).  실행: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.gcn import GCN  # noqa: E402
from common.train_gnn import main  # noqa: E402
from scripts.make_synthetic import make_synthetic  # noqa: E402


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return make_synthetic(tmp_path_factory.mktemp("synth_gnn"), per_step=120, seed=2)


def test_gcn_end_to_end(synth, tmp_path):
    out = tmp_path / "res"
    main("gcn", lambda d, a: GCN(d, a.hidden, a.dropout),
         ["--data-root", str(synth), "--seeds", "0", "--max-epochs", "8", "--patience", "5",
          "--results-dir", str(out), "--cpu"])
    m = pd.read_csv(out / "metrics" / "gcn.csv", dtype={"threshold_type": str})
    assert set(m["split"]) == {"test", "val"}
    assert {"fixed_0.5", "val_opt"} <= set(m["threshold_type"])
    pred = np.load(out / "pred" / "pred_gcn_seed0.npy")
    emb = np.load(out / "emb" / "emb_gcn_seed0.npy")
    assert pred.ndim == 1 and emb.shape == (pred.shape[0], 64)
    assert np.all((pred >= 0) & (pred <= 1))
    assert (out / "timestep_f1_gcn.csv").exists()
