"""OOF 임베딩 생성(hybrid/make_oof_emb.py) 테스트.  torch / torch_geometric 없으면 건너뜀.  실행: pytest -q tests/test_oof.py"""
import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common.data import load_data, to_pyg  # noqa: E402
from common.train_gnn import main as gnn_main  # noqa: E402
from hybrid import make_oof_emb as oof  # noqa: E402
from hybrid import run_hybrid  # noqa: E402
from models.gcn import GCN  # noqa: E402
from scripts.make_synthetic import make_synthetic  # noqa: E402

FACTORY = lambda d, a: GCN(d, a.hidden, a.dropout)  # noqa: E731
ARGS = argparse.Namespace(hidden=64, dropout=0.5, lr=0.01, weight_decay=5e-4)


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return make_synthetic(tmp_path_factory.mktemp("synth_oof"), per_step=100, seed=4)


@pytest.fixture(scope="module")
def ref_dir(synth, tmp_path_factory):
    """기준 임베딩(= 저장된 전체 재학습 모델의 임베딩)을 train_gnn으로 만든다."""
    out = tmp_path_factory.mktemp("ref")
    gnn_main("gcn", FACTORY, ["--data-root", str(synth), "--seeds", "0", "--max-epochs", "6", "--patience", "3",
                              "--results-dir", str(out), "--cpu", "--lr", "0.01"])
    return out


# ---------- 폴드 / 정렬 ----------
def test_time_folds_partition_steps_1_to_34_contiguously():
    folds = oof.time_folds(5)
    assert [(f[0], f[-1]) for f in folds] == [(1, 7), (8, 14), (15, 21), (22, 28), (29, 34)]
    assert np.array_equal(np.concatenate(folds), np.arange(1, 35))
    assert np.array_equal(np.concatenate(oof.time_folds(3)), np.arange(1, 35))


def test_procrustes_recovers_rotation_scale_shift_and_reflection():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(500, 16))
    Q, _ = np.linalg.qr(rng.normal(size=(16, 16)))          # 임의의 직교행렬
    Q[:, 0] *= -1                                           # 반사가 섞인 경우도 포함
    B = 2.5 * A @ Q + rng.normal(size=16) * 3
    f = oof.procrustes_fit(A, B)
    assert f["s"] == pytest.approx(2.5, rel=1e-6)
    assert np.allclose(oof.procrustes_apply(A, f), B, atol=1e-4)
    assert oof.rel_residual(oof.procrustes_apply(A, f), B) < 1e-5
    assert oof.mean_cosine(A, A) == pytest.approx(1.0)


def test_procrustes_fitted_on_subset_generalizes_and_is_label_free():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(2000, 8))
    Q, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    B = A @ Q + 0.01 * rng.normal(size=A.shape)
    f = oof.procrustes_fit(A[:300], B[:300])                # 일부(라벨 없는 노드)에서만 구해도
    assert oof.rel_residual(oof.procrustes_apply(A[300:], f), B[300:]) < 0.05   # 나머지에 잘 맞는다


def test_best_epochs_from_metrics(tmp_path):
    rows = [dict(model="gcn", seed=s, split="test", threshold_type="fixed_0.5", notes=f"best_epoch={e};hidden=64;dropout=0.5")
            for s, e in [(0, 144), (1, 167)]]
    rows += [dict(model="gcn", seed=0, split="val", threshold_type="val_opt", notes="best_epoch=1")]
    p = tmp_path / "gcn.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    assert oof.best_epochs_from_metrics("gcn", [0, 1], p) == {0: 144, 1: 167}
    with pytest.raises(ValueError):
        oof.best_epochs_from_metrics("gcn", [0, 2], p)
    with pytest.raises(FileNotFoundError):
        oof.best_epochs_from_metrics("gcn", [0], tmp_path / "none.csv")


# ---------- 핵심: 정답이 새지 않는가 ----------
def test_fold_embeddings_do_not_depend_on_that_folds_labels(synth):
    """블록 k의 라벨을 전부 뒤집어도 블록 k의 OOF 임베딩은 그대로여야 한다 (다른 블록은 달라져야 한다)."""
    data = load_data(synth, standardize=True)
    ref = np.random.default_rng(0).normal(size=(data.num_nodes, 64)).astype(np.float32)
    dev = torch.device("cpu")

    def run(d):
        g = to_pyg(d).to(dev)
        g.x, g.y = g.x.float(), g.y.long()
        return oof.oof_embeddings(FACTORY, ARGS, d, g, ref, seed=0, epochs=6, device=dev)[0]

    base = run(data)
    folds = oof.time_folds(5)
    held0 = np.isin(data.time_step, folds[0])
    y2 = data.y.copy()
    flip = held0 & (y2 != 2)
    assert flip.sum() > 20
    y2[flip] = 1 - y2[flip]
    changed = run(dataclasses.replace(data, y=y2))

    assert np.allclose(base[held0], changed[held0], atol=1e-5)          # 자기 블록의 라벨은 영향 없음
    other = np.isin(data.time_step, np.concatenate(folds[1:]))
    assert not np.allclose(base[other], changed[other], atol=1e-3)      # 다른 블록은 학습 라벨이 바뀌었으니 달라짐 (테스트가 둔감하지 않다는 확인)
    test_rows = data.time_step > 34
    assert np.array_equal(base[test_rows], ref[test_rows])              # 테스트 구간은 기준 임베딩 그대로


# ---------- 전체 흐름 ----------
def test_cli_end_to_end_and_hybrid_can_read_the_result(synth, ref_dir, tmp_path):
    out = tmp_path / "oof"
    oof.main(["--model", "gcn", "--data-root", str(synth), "--seeds", "0", "--best-epochs", "5", "--lr", "0.01",
              "--ref-emb-dir", str(ref_dir / "emb"), "--out-dir", str(out), "--cpu"])
    data = load_data(synth, standardize=False)
    ref = np.load(ref_dir / "emb" / "emb_gcn_seed0.npy")
    e = np.load(out / "emb_oof_gcn_seed0.npy")
    assert e.shape == ref.shape == (data.num_nodes, 64) and e.dtype == np.float32 and np.isfinite(e).all()
    te = data.time_step > 34
    assert np.array_equal(e[te], ref[te])                                # 테스트 노드 = 기준 임베딩
    assert not np.allclose(e[~te], ref[~te], atol=1e-3)                  # 학습·검증 노드는 새로 만든 값

    rep = pd.read_csv(out / "oof_report_gcn.csv")
    assert len(rep) == 5 and set(rep["fold"]) == {1, 2, 3, 4, 5}
    assert {"cos_raw", "cos_aligned_unlabeled", "cos_aligned_labeled", "oof_pr_auc", "oof_pr_auc_pooled"} <= set(rep.columns)
    assert (rep["n_unlabeled_fit"] > 0).all() and (rep["n_train"] > 0).all()

    # 하이브리드 코드가 이 파일을 그대로 읽는다 (--emb-kind oof → 결과 이름 hybrid_gcn)
    tune = tmp_path / "tune.csv"
    pd.DataFrame([dict(max_depth=3, learning_rate=0.3, min_child_weight=1, subsample=1.0, colsample_bytree=1.0,
                       val_pr_auc=0.9, best_iteration=10)]).to_csv(tune, index=False)
    run_hybrid.main(["--data-root", str(synth), "--tuning-csv", str(tune), "--results-dir", str(tmp_path / "res"),
                   "--n-jobs", "1", "--emb-models", "gcn", "--emb-kind", "oof", "--emb-dir", str(out), "--seeds", "0"])
    m = pd.read_csv(tmp_path / "res" / "metrics" / "hybrid_gcn.csv", dtype={"threshold_type": str})
    assert set(m["split"]) == {"test", "val", "train"}


def test_wrong_inputs_fail_clearly(synth, ref_dir, tmp_path):
    with pytest.raises(SystemExit):                                       # best-epochs 개수 불일치
        oof.main(["--model", "gcn", "--data-root", str(synth), "--seeds", "0", "1", "--best-epochs", "5",
                  "--out-dir", str(tmp_path / "o"), "--cpu"])
    with pytest.raises(FileNotFoundError, match="기준 임베딩"):
        oof.main(["--model", "gcn", "--data-root", str(synth), "--seeds", "0", "--best-epochs", "5",
                  "--ref-emb-dir", str(tmp_path / "none"), "--out-dir", str(tmp_path / "o"), "--cpu"])
    # GAT: models/gat.py가 있으면 GAT를 만들고 (PR #5 이후), 없으면 분명한 SystemExit
    if (Path(__file__).resolve().parents[1] / "models" / "gat.py").exists():
        assert type(oof.get_factory("gat")(16, ARGS)).__name__ == "GAT"
    else:
        with pytest.raises(SystemExit, match="gat"):
            oof.get_factory("gat")
