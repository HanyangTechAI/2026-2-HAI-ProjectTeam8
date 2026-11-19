"""하이브리드(hybrid/skip_gcn.py) 테스트: 가짜 Elliptic 데이터 + 가짜 임베딩.  실행: pytest -q tests/test_hybrid.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines import xgb_baseline  # noqa: E402
from common.data import load_data  # noqa: E402
from hybrid import skip_gcn  # noqa: E402
from scripts.make_synthetic import make_synthetic  # noqa: E402


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return make_synthetic(tmp_path_factory.mktemp("synth_hybrid"), per_step=120, seed=3)


@pytest.fixture(scope="module")
def data(synth):
    return load_data(synth, standardize=False)


@pytest.fixture(scope="module")
def tuning_csv(tmp_path_factory):
    """튜닝 기록 흉내: 1등이 첫 줄이 아니어도 val_pr_auc 최대인 줄을 골라야 한다. 트리는 작게 해서 테스트를 빠르게 한다."""
    p = tmp_path_factory.mktemp("tune") / "xgb_tuning.csv"
    pd.DataFrame([
        dict(max_depth=8, learning_rate=0.1, min_child_weight=5, subsample=1.0, colsample_bytree=0.7, val_pr_auc=0.50, best_iteration=10),
        dict(max_depth=3, learning_rate=0.3, min_child_weight=1, subsample=1.0, colsample_bytree=1.0, val_pr_auc=0.90, best_iteration=10),
    ]).to_csv(p, index=False)
    return p


def _write_emb(dirpath: Path, prefix: str, model: str, seeds, data, seed_offset=0):
    """불법 노드의 앞 4차원을 밀어 놓은 가짜 임베딩 (정답이 새겨진 naive 임베딩을 흉내)."""
    dirpath.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        rng = np.random.default_rng(100 + s + seed_offset)
        e = rng.normal(size=(data.num_nodes, 64)).astype(np.float32)
        e[data.y == 1, :4] += 2.0
        np.save(dirpath / f"{prefix}_{model}_seed{s}.npy", e)


@pytest.fixture(scope="module")
def emb_dir(tmp_path_factory, data):
    d = tmp_path_factory.mktemp("emb")
    for m, off in (("gcn", 0), ("sage", 7)):
        _write_emb(d, "emb", m, (0, 1), data, off)
    return d


def run(synth, tuning_csv, out, *extra):
    skip_gcn.main(["--data-root", str(synth), "--tuning-csv", str(tuning_csv), "--results-dir", str(out),
                   "--n-jobs", "1", *extra])


# ---------- 설정·이름 ----------
def test_default_names():
    f = skip_gcn.default_name
    assert f("hybrid", ["gcn"], "naive", False) == "hybrid_gcn_naive"
    assert f("hybrid", ["gcn"], "oof", False) == "hybrid_gcn"
    assert f("only", ["sage"], "oof", False) == "emb_only_sage"
    assert f("hybrid", ["gat", "gcn", "sage"], "naive", False) == "hybrid_all_naive"
    assert f("hybrid", ["gcn", "sage"], "oof", False) == "hybrid_gcn_sage"
    assert f("hybrid", ["gcn"], "oof", True) == "hybrid_gcn_retuned"
    assert f("raw", ["gcn"], "naive", False) == "xgb_check"


def test_baseline_params_use_best_val_row(tuning_csv):
    p = skip_gcn.load_baseline_params(tuning_csv)
    assert p["max_depth"] == 3 and isinstance(p["max_depth"], int)
    assert p["min_child_weight"] == 1 and isinstance(p["min_child_weight"], int)
    assert p["learning_rate"] == 0.3 and p["colsample_bytree"] == 1.0
    assert p["n_estimators"] == xgb_baseline.BASE_PARAMS["n_estimators"]  # 나머지는 베이스라인 기본값


def test_baseline_params_missing_file_and_columns(tmp_path):
    with pytest.raises(FileNotFoundError):
        skip_gcn.load_baseline_params(tmp_path / "nope.csv")
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"max_depth": [3]}).to_csv(bad, index=False)
    with pytest.raises(ValueError):
        skip_gcn.load_baseline_params(bad)


# ---------- 하이브리드 실행 ----------
def test_hybrid_naive_end_to_end(synth, data, tuning_csv, emb_dir, tmp_path):
    out = tmp_path / "res"
    run(synth, tuning_csv, out, "--emb-models", "gcn", "--emb-dir", str(emb_dir), "--seeds", "0", "1")
    m = pd.read_csv(out / "metrics" / "hybrid_gcn_naive.csv", dtype={"threshold_type": str})
    assert set(m["split"]) == {"test", "val", "train"}
    assert len(m) == 2 * 4                                   # 시드 2 × (test 2 + val + train)
    assert {"fixed_0.5", "val_opt"} == set(m[m["split"] == "test"]["threshold_type"])
    t = m[(m["split"] == "test") & (m["threshold_type"] == "fixed_0.5")]
    assert t["pr_auc"].between(0, 1).all() and t["f1"].between(0, 1).all()
    assert t["notes"].str.contains("kind=naive").all() and t["notes"].str.contains("dim=229").all()
    assert (t["train_sec"] > 0).all() and (t["n_params"] > 0).all()

    pred = np.load(out / "pred" / "pred_hybrid_gcn_naive_seed0.npy")
    assert pred.shape == (data.num_nodes,) and pred.dtype == np.float32 and np.all((pred >= 0) & (pred <= 1))
    assert (out / "timestep_f1_hybrid_gcn_naive.csv").exists()

    imp = pd.read_csv(out / "hybrid_importance_hybrid_gcn_naive.csv")
    s0 = imp[imp["seed"] == 0].set_index("group")
    assert set(s0.index) == {"raw", "gcn"}
    assert s0["gain_share"].sum() == pytest.approx(1.0, abs=1e-6)
    assert s0.loc["gcn", "n_features"] == 64 and s0.loc["raw", "n_features"] == 165
    assert s0.loc["gcn", "gain_share"] > 0.05                # 임베딩에 신호를 심어 뒀으니 트리가 써야 한다


def test_rerun_replaces_rows_instead_of_duplicating(synth, tuning_csv, emb_dir, tmp_path):
    out = tmp_path / "res"
    args = ("--emb-models", "gcn", "--emb-dir", str(emb_dir), "--seeds", "0")
    run(synth, tuning_csv, out, *args)
    run(synth, tuning_csv, out, *args)
    m = pd.read_csv(out / "metrics" / "hybrid_gcn_naive.csv", dtype={"threshold_type": str})
    assert len(m) == 4
    assert len(pd.read_csv(out / "hybrid_importance_hybrid_gcn_naive.csv")) == 2   # raw, gcn
    steps = pd.read_csv(out / "timestep_f1_hybrid_gcn_naive.csv")
    assert not steps.duplicated(["seed", "time_step"]).any()


def test_only_mode_and_two_models(synth, data, tuning_csv, emb_dir, tmp_path):
    out = tmp_path / "res"
    run(synth, tuning_csv, out, "--emb-models", "gcn", "sage", "--emb-dir", str(emb_dir), "--seeds", "0", "--mode", "only")
    m = pd.read_csv(out / "metrics" / "emb_only_gcn_sage_naive.csv", dtype={"threshold_type": str})
    assert m[m["split"] == "test"]["notes"].str.contains("dim=128").all()          # 임베딩 64 × 2, 원본 없음
    imp = pd.read_csv(out / "hybrid_importance_emb_only_gcn_sage_naive.csv")
    assert set(imp["group"]) == {"gcn", "sage"}
    run(synth, tuning_csv, out, "--emb-models", "gcn", "sage", "--emb-dir", str(emb_dir), "--seeds", "0")
    m2 = pd.read_csv(out / "metrics" / "hybrid_gcn_sage_naive.csv", dtype={"threshold_type": str})
    assert m2[m2["split"] == "test"]["notes"].str.contains("dim=293").all()        # 165 + 128
    imp2 = pd.read_csv(out / "hybrid_importance_hybrid_gcn_sage_naive.csv")
    assert set(imp2["group"]) == {"raw", "gcn", "sage"} and imp2["gain_share"].sum() == pytest.approx(1.0, abs=1e-6)


def test_oof_kind_reads_oof_files_and_names_result(synth, data, tuning_csv, tmp_path):
    oof = tmp_path / "emb_oof"
    _write_emb(oof, "emb_oof", "gcn", (0,), data)
    out = tmp_path / "res"
    run(synth, tuning_csv, out, "--emb-models", "gcn", "--emb-kind", "oof", "--emb-dir", str(oof), "--seeds", "0")
    m = pd.read_csv(out / "metrics" / "hybrid_gcn.csv", dtype={"threshold_type": str})
    assert m[m["split"] == "test"]["notes"].str.contains("kind=oof").all()


def test_retune_option(synth, tuning_csv, emb_dir, tmp_path):
    out = tmp_path / "res"
    run(synth, tuning_csv, out, "--emb-models", "gcn", "--emb-dir", str(emb_dir), "--seeds", "0", "--retune-trials", "2")
    assert (out / "metrics" / "hybrid_gcn_naive_retuned.csv").exists()
    log = pd.read_csv(out / "xgb_tuning_hybrid_gcn_naive_retuned.csv")
    assert len(log) == 2 and log["val_pr_auc"].is_monotonic_decreasing


# ---------- 베이스라인과 같은 프로토콜인가 ----------
def test_raw_mode_reproduces_the_baseline_exactly(synth, data, tuning_csv, tmp_path):
    """--mode raw는 임베딩 없이 베이스라인과 같은 입력이므로, 같은 설정·시드면 점수가 똑같이 나와야 한다."""
    params = skip_gcn.load_baseline_params(tuning_csv)
    base_rows = xgb_baseline.run_seed(data, params, 0, 1, tmp_path / "base")
    run(synth, tuning_csv, tmp_path / "chk", "--mode", "raw", "--seeds", "0")
    chk = pd.read_csv(tmp_path / "chk" / "metrics" / "xgb_check.csv", dtype={"threshold_type": str})
    for b in base_rows:
        if b["split"] != "test":
            continue
        c = chk[(chk["split"] == "test") & (chk["threshold_type"] == b["threshold_type"])].iloc[0]
        for k in ("pr_auc", "roc_auc", "precision", "recall", "f1", "threshold"):
            assert c[k] == pytest.approx(b[k], abs=1e-9), k
        assert c["notes"].split(";")[0] == b["notes"]            # 같은 트리 수
    p0 = np.load(tmp_path / "base" / "pred" / "pred_xgb_seed0.npy")
    p1 = np.load(tmp_path / "chk" / "pred" / "pred_xgb_check_seed0.npy")
    assert np.allclose(p0, p1, atol=1e-6)


# ---------- 잘못된 입력 ----------
def test_missing_embedding_files_fail_before_training(synth, tuning_csv, tmp_path):
    with pytest.raises(FileNotFoundError, match="임베딩 파일을 찾지 못했습니다"):
        run(synth, tuning_csv, tmp_path / "res", "--emb-models", "gcn", "--emb-dir", str(tmp_path / "none"), "--seeds", "0")
    assert not (tmp_path / "res" / "metrics").exists()


def test_oof_missing_gives_hint(synth, tuning_csv, tmp_path):
    with pytest.raises(FileNotFoundError, match="OOF"):
        run(synth, tuning_csv, tmp_path / "res", "--emb-models", "gcn", "--emb-kind", "oof",
            "--emb-dir", str(tmp_path / "none"), "--seeds", "0")


def test_load_embedding_rejects_lfs_pointer_wrong_shape_and_nan(tmp_path):
    pointer = tmp_path / "ptr.npy"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 52164992\n")
    with pytest.raises(RuntimeError, match="LFS"):
        skip_gcn.load_embedding(pointer, 10)
    wrong = tmp_path / "wrong.npy"
    np.save(wrong, np.zeros((9, 64), dtype=np.float32))
    with pytest.raises(ValueError, match="모양"):
        skip_gcn.load_embedding(wrong, 10)
    bad = tmp_path / "nan.npy"
    arr = np.zeros((10, 64), dtype=np.float32)
    arr[3, 5] = np.nan
    np.save(bad, arr)
    with pytest.raises(ValueError, match="NaN"):
        skip_gcn.load_embedding(bad, 10)
    ok = tmp_path / "ok.npy"
    np.save(ok, np.ones((10, 64), dtype=np.float32))
    assert skip_gcn.load_embedding(ok, 10).shape == (10, 64)
