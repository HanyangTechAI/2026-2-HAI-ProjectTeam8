"""local 93개 피처 래퍼(hybrid/local_only.py) 테스트.  실행: pytest -q tests/test_local_only.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid import local_only as lo  # noqa: E402
from scripts.make_synthetic import make_synthetic  # noqa: E402


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return make_synthetic(tmp_path_factory.mktemp("synth_local"), per_step=40, seed=3)


def test_load_data_local_keeps_first_93_columns_and_everything_else(synth):
    full = lo._orig_load_data(synth, standardize=True)
    loc = lo.load_data_local(synth, standardize=True)
    assert loc.x.shape == (full.num_nodes, 93) and loc.x_std.shape == (full.num_nodes, 93)
    assert np.array_equal(loc.x, full.x[:, :93]) and np.array_equal(loc.x_std, full.x_std[:, :93])
    for k in ("y", "time_step", "edge_index", "train_mask", "val_mask", "test_mask", "node_ids"):
        assert np.array_equal(getattr(loc, k), getattr(full, k)), k
    assert lo.load_data_local(synth, standardize=False).x_std is None


def test_wrapper_runs_hybrid_with_93_raw_features(synth, tmp_path):
    pytest.importorskip("xgboost")
    tune = tmp_path / "tune.csv"
    pd.DataFrame([dict(max_depth=3, learning_rate=0.3, min_child_weight=1, subsample=1.0, colsample_bytree=1.0,
                       val_pr_auc=0.9, best_iteration=10)]).to_csv(tune, index=False)
    out = tmp_path / "res"
    lo.main(["hybrid.run_hybrid", "--mode", "raw", "--name", "xgb_local", "--data-root", str(synth),
             "--seeds", "0", "--tuning-csv", str(tune), "--results-dir", str(out)])
    m = pd.read_csv(out / "metrics" / "xgb_local.csv")
    assert m["notes"].str.contains("dim=93").all()
    imp = pd.read_csv(out / "hybrid_importance_xgb_local.csv")
    assert imp.loc[imp.group == "raw", "n_features"].iloc[0] == 93


def test_wrapper_requires_module_name():
    with pytest.raises(SystemExit):
        lo.main(["--mode", "raw"])
