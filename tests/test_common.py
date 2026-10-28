"""공통 기반 테스트 (가짜 Elliptic 형식 데이터 사용).  실행: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines import xgb_baseline  # noqa: E402
from common.data import TRAIN_END, VAL_END, get_tabular, load_data, split_summary  # noqa: E402
from common.metrics import best_threshold, evaluate, evaluate_by_timestep  # noqa: E402
from common.utils import METRIC_COLUMNS, save_rows  # noqa: E402
from scripts.make_synthetic import make_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    return make_synthetic(tmp_path_factory.mktemp("synth"), per_step=200, seed=1)


@pytest.fixture(scope="session")
def data(synth):
    return load_data(synth, standardize=True)


# ---------- 데이터 ----------
def test_node_order_and_shapes(data):
    assert np.all(np.diff(data.node_ids) > 0)                 # txId 오름차순
    assert data.x.shape == (data.num_nodes, 165)              # 시간 단계 제외 165차원
    assert data.x_std.shape == data.x.shape
    assert set(np.unique(data.y)) <= {0, 1, 2}
    assert data.time_step.min() == 1 and data.time_step.max() == 49


def test_masks_follow_time_split_and_exclude_unknown(data):
    tr, va, te = data.train_mask, data.val_mask, data.test_mask
    assert (tr & va).sum() == 0 and (tr & te).sum() == 0 and (va & te).sum() == 0
    assert data.time_step[tr].max() <= TRAIN_END
    assert data.time_step[va].min() > TRAIN_END and data.time_step[va].max() <= VAL_END
    assert data.time_step[te].min() > VAL_END
    assert (data.y[tr | va | te] != 2).all()                  # 미분류는 어떤 마스크에도 없음
    # 라벨 있는 노드는 모두 어느 한 구간에 들어간다
    assert (tr | va | te).sum() == (data.y != 2).sum()


def test_edges_undirected_clean_and_same_step(data):
    ei = data.edge_index
    assert (ei[0] != ei[1]).all()                              # 자기 루프 없음
    pairs = set(zip(ei[0].tolist(), ei[1].tolist()))
    assert len(pairs) == ei.shape[1]                           # 중복 없음
    assert all((b, a) in pairs for a, b in list(pairs)[:500])  # 대칭
    assert (data.time_step[ei[0]] == data.time_step[ei[1]]).all()


def test_standardization_uses_only_steps_up_to_34(data):
    ref = data.x_std[data.time_step <= VAL_END]
    assert np.abs(ref.mean(axis=0)).max() < 1e-3
    assert np.abs(ref.std(axis=0) - 1).max() < 1e-2


def test_get_tabular_returns_raw_by_default(data):
    X, y = get_tabular(data, data.train_mask)
    assert X.shape[0] == y.shape[0] == data.train_mask.sum()
    assert np.allclose(X, data.x[data.train_mask])


def test_split_summary_counts_match(data):
    s = split_summary(data)
    assert s["labeled_nodes"].sum() == (data.y != 2).sum()
    assert (s["illicit"] + s["licit"] == s["labeled_nodes"]).all()


def test_missing_data_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="데이터 준비"):
        load_data(tmp_path)


# ---------- 평가 함수 ----------
def test_perfect_and_worst_scores():
    y = np.array([0, 0, 0, 1, 1, 0, 1, 0])
    m = evaluate(y, y.astype(float))
    assert m["pr_auc"] == pytest.approx(1.0) and m["f1"] == pytest.approx(1.0)
    w = evaluate(y, 1 - y.astype(float))
    assert w["f1"] == 0.0 and w["recall"] == 0.0


def test_evaluate_matches_hand_computation():
    y = np.array([1, 1, 1, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.3, 0.7, 0.2, 0.1])
    m = evaluate(y, p, 0.5)       # 예측=[1,1,0,1,0,0] → TP2 FP1 FN1
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)


def test_single_class_gives_nan_auc_not_crash():
    m = evaluate(np.zeros(5), np.linspace(0, 1, 5))
    assert np.isnan(m["pr_auc"]) and np.isnan(m["roc_auc"])


def test_best_threshold_maximizes_f1_on_given_data():
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.2).astype(int)
    p = np.clip(0.5 * y + rng.normal(0.25, 0.15, 500), 0, 1)
    thr = best_threshold(y, p)
    best = evaluate(y, p, thr)["f1"]
    assert all(best >= evaluate(y, p, t)["f1"] - 1e-9 for t in np.linspace(0.05, 0.95, 19))
    assert best_threshold(np.zeros(10), rng.random(10)) == 0.5   # 불법 없음 → 기본값


def test_evaluate_by_timestep_skips_steps_without_illicit():
    y = np.array([1, 0, 0, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.2, 0.1, 0.8, 0.1])
    steps = np.array([35, 35, 36, 36, 37, 37])
    out = evaluate_by_timestep(y, p, steps, 0.5)
    assert set(out) == {35, 37}                  # 36단계는 불법이 없어 제외


# ---------- 결과 저장 ----------
def test_save_rows_replaces_same_key_and_keeps_columns(tmp_path):
    row = dict(model="m", seed=0, split="test", threshold_type="val_opt", f1=0.5)
    save_rows("m", [row], tmp_path)
    save_rows("m", [{**row, "f1": 0.7}, {**row, "seed": 1}], tmp_path)
    df = pd.read_csv(tmp_path / "metrics" / "m.csv")
    assert list(df.columns) == METRIC_COLUMNS
    assert len(df) == 2 and df.loc[df.seed == 0, "f1"].item() == 0.7


# ---------- XGBoost 베이스라인 (스모크) ----------
def test_xgb_baseline_end_to_end(synth, tmp_path):
    xgb_baseline.main(["--data-root", str(synth), "--seeds", "0", "1", "--tune-trials", "3",
                       "--n-jobs", "2", "--results-dir", str(tmp_path)])
    df = pd.read_csv(tmp_path / "metrics" / "xgb.csv", dtype={"threshold_type": str})
    assert list(df.columns) == METRIC_COLUMNS
    assert set(df.threshold_type) == {"fixed_0.5", "val_opt"}
    test = df[(df.split == "test") & (df.threshold_type == "val_opt")]
    assert len(test) == 2 and (test.pr_auc > 0.3).all()          # 가짜 데이터에는 학습 가능한 신호가 있음
    assert (test.train_sec > 0).all() and (test.infer_sec > 0).all()
    pred = np.load(tmp_path / "pred" / "pred_xgb_seed0.npy")
    d = load_data(synth, standardize=False)
    assert pred.shape == (d.num_nodes,) and pred.dtype == np.float32
    assert ((pred >= 0) & (pred <= 1)).all()
    assert (tmp_path / "timestep_f1_xgb.csv").exists()
    assert (tmp_path / "xgb_tuning.csv").exists()


def test_tuning_never_touches_test_split(synth, monkeypatch):
    """튜닝 함수는 test 마스크 행을 한 번도 요청하지 않아야 한다."""
    d = load_data(synth, standardize=False)
    seen = []
    real = xgb_baseline.get_tabular
    monkeypatch.setattr(xgb_baseline, "get_tabular",
                        lambda data, mask, **kw: (seen.append(int((mask & data.test_mask).sum())), real(data, mask, **kw))[1])
    xgb_baseline.tune(d, trials=2, n_jobs=2)
    assert seen and max(seen) == 0
