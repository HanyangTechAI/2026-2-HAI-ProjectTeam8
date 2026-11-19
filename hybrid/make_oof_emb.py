"""OOF(Out-Of-Fold) 임베딩 생성 — 하이브리드의 누수 처리 (방식 B).  torch / torch_geometric 필요.

왜: 저장된 임베딩(emb_{model}_seed{n}.npy)은 Train+Val(1~34)의 정답을 보고 학습한 GNN에서 나온 것이라,
    1~34 노드의 임베딩에 그 노드의 정답이 새겨져 있다 (README '하이브리드 누수 처리' 참고).
    OOF는 "어떤 노드의 임베딩은 그 노드의 정답을 본 적 없는 GNN이 만든 것"이 되도록 다시 만든다.

방법 (모델·시드마다):
  1. 시간 단계 1~34를 연속된 K개 블록으로 나눈다 (기본 5개: 1~7 / 8~14 / 15~21 / 22~28 / 29~34).
  2. 블록 k를 뺀 나머지 블록의 라벨 노드만으로 GNN을 학습한다 (같은 하이퍼파라미터·같은 시드,
     에폭 수 = 그 모델·시드 본 학습의 best_epoch로 고정. 검증 구간을 따로 두지 않으므로 조기 종료는 없음).
  3. 전체 그래프에 forward 해서 블록 k 노드의 임베딩만 가져온다. (이 GNN은 블록 k의 정답을 본 적이 없다.
     블록 k 노드의 피처와 간선은 쓰지만 라벨은 쓰지 않는다. 간선은 같은 시간 단계 안에만 있다.)
  4. 폴드마다 GNN이 따로 학습돼 임베딩 '좌표계'가 다르다. 그래서 블록 k의 **라벨 없는 노드(y=2)** 에서
     폴드 임베딩 → 기준 임베딩(저장된 전체 재학습 모델의 임베딩)으로 가는 직교 Procrustes 정렬
     (회전/반사 + 크기 + 이동)을 구해 블록 k 전체에 적용한다. 정렬에는 어떤 정답도 쓰이지 않는다.
  5. 테스트 구간(35~49) 노드는 기준 임베딩을 그대로 쓴다 (정답을 본 적 없는 노드이므로 누수 없음).

저장: results/emb_oof/emb_oof_{model}_seed{n}.npy  (N, 64) float32, 노드 순서 = load_data() 순서
      results/emb_oof/oof_report_{model}.csv        폴드별 진단 (정렬 품질, 폴드 모델의 OOF PR-AUC)

실행 예 (GCN 튜닝 후 설정):
  python -m hybrid.make_oof_emb --model gcn --lr 0.01 --dropout 0.5 --weight-decay 5e-4 --seeds 0 1 2 3 4
  python -m hybrid.make_oof_emb --model gcn --best-epochs 144 167 203 306 224 ...   # 에폭 수를 직접 지정
그 다음:
  python -m hybrid.skip_gcn --emb-models gcn --emb-kind oof
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.data import VAL_END, EllipticData, load_data, to_pyg  # noqa: E402
from common.train_gnn import class_weight, fit, predict  # noqa: E402
from common.utils import RESULTS_DIR, set_seed  # noqa: E402

DEFAULT_FOLDS = 5


# ---------------------------------------------------------------- 모델 / 설정
def get_factory(model: str):
    """models/*.py의 클래스를 train_gnn.main과 같은 방식 factory(in_dim, args)로 돌려준다."""
    if model == "gcn":
        from models.gcn import GCN
        return lambda d, a: GCN(d, a.hidden, a.dropout)
    if model == "sage":
        from models.sage import SAGE
        return lambda d, a: SAGE(d, a.hidden, a.dropout)
    if model == "gat":
        try:
            from models.gat import GAT
        except ImportError as e:
            raise SystemExit("models/gat.py 가 저장소에 없습니다. GAT 조가 코드를 올린 뒤 사용할 수 있습니다.") from e
        return lambda d, a: GAT(d, a.hidden, a.dropout)
    raise ValueError(f"알 수 없는 모델: {model}")


def best_epochs_from_metrics(model: str, seeds: list[int], metrics_csv: str | Path) -> dict[int, int]:
    """metrics/{model}.csv의 notes(best_epoch=...)에서 시드별 best_epoch를 읽는다."""
    path = Path(metrics_csv)
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. --metrics-csv 로 경로를 주거나 --best-epochs 로 직접 지정하세요.")
    m = pd.read_csv(path, dtype={"threshold_type": str})
    m = m[(m["split"] == "test") & (m["threshold_type"] == "fixed_0.5")]
    out = {}
    for s in seeds:
        r = m[m["seed"] == s]
        if not len(r):
            raise ValueError(f"{path} 에 시드 {s}의 테스트 행이 없습니다.")
        out[s] = int(r["notes"].astype(str).str.extract(r"best_epoch=(\d+)")[0].iloc[0])
    return out


# ---------------------------------------------------------------- 폴드·정렬
def time_folds(n_folds: int = DEFAULT_FOLDS, last_step: int = VAL_END) -> list[np.ndarray]:
    """시간 단계 1~last_step을 연속된 n_folds개 블록으로 나눈다 (34 → 7,7,7,7,6)."""
    return [np.asarray(b) for b in np.array_split(np.arange(1, last_step + 1), n_folds)]


def procrustes_fit(A: np.ndarray, B: np.ndarray) -> dict:
    """B ≈ s · (A − mu_a) @ R + mu_b 가 되는 직교 R(회전/반사), 크기 s, 이동을 구한다 (scaled orthogonal Procrustes)."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    mu_a, mu_b = A.mean(axis=0), B.mean(axis=0)
    A0, B0 = A - mu_a, B - mu_b
    U, S, Vt = np.linalg.svd(A0.T @ B0)
    R = U @ Vt
    s = float(S.sum() / max((A0 ** 2).sum(), 1e-12))
    return dict(mu_a=mu_a, mu_b=mu_b, R=R, s=s)


def procrustes_apply(X: np.ndarray, fit_: dict) -> np.ndarray:
    return (fit_["s"] * (np.asarray(X, dtype=np.float64) - fit_["mu_a"]) @ fit_["R"] + fit_["mu_b"]).astype(np.float32)


def mean_cosine(A: np.ndarray, B: np.ndarray) -> float:
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    num = (A * B).sum(axis=1)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-12
    return float((num / den).mean())


def rel_residual(A: np.ndarray, B: np.ndarray) -> float:
    """||A − B|| / ||B − mean(B)||  (0이면 완전히 일치)"""
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    return float(np.linalg.norm(A - B) / max(np.linalg.norm(B - B.mean(axis=0)), 1e-12))


# ---------------------------------------------------------------- OOF 본체
def oof_embeddings(factory, args, data: EllipticData, g, ref_emb: np.ndarray, seed: int, epochs: int,
                   device, n_folds: int = DEFAULT_FOLDS) -> tuple[np.ndarray, list[dict]]:
    """한 시드의 OOF 임베딩 (N, d)과 폴드별 진단 행을 만든다.

    1~34 노드 = 정렬된 OOF 임베딩, 35~49 노드 = ref_emb 그대로.
    """
    N, D = ref_emb.shape
    y, step = data.y, data.time_step
    out = ref_emb.astype(np.float32).copy()
    rows, oof_prob = [], np.full(N, np.nan, dtype=np.float32)

    for k, steps in enumerate(time_folds(n_folds)):
        held = np.isin(step, steps)                                  # 블록 k의 모든 노드 (라벨 없는 노드 포함)
        train_k = (y != 2) & (step <= VAL_END) & ~held               # 학습: 다른 블록의 라벨 노드만
        set_seed(seed)                                               # 기준 모델과 같은 초기 가중치에서 시작
        model = factory(g.x.shape[1], args).to(device)
        tmask = torch.from_numpy(train_k).to(device)
        fit(model, g, tmask, class_weight(y[train_k], device), args.lr, args.weight_decay, epochs)
        prob, emb = predict(model, g)

        unl = held & (y == 2)                                        # 정렬에는 라벨 없는 노드만 사용
        lab = held & (y != 2)
        fit_ = procrustes_fit(emb[unl], ref_emb[unl])
        aligned = procrustes_apply(emb[held], fit_)
        out[held] = aligned
        oof_prob[held] = prob[held]

        al_full = procrustes_apply(emb, fit_)                        # 진단용: 블록 전체에 같은 변환을 적용해 비교
        pr = float(average_precision_score(y[lab], prob[lab])) if len(np.unique(y[lab])) == 2 else float("nan")
        rows.append(dict(seed=seed, fold=k + 1, steps=f"{steps[0]}~{steps[-1]}", epochs=epochs,
                         n_train=int(train_k.sum()), n_unlabeled_fit=int(unl.sum()), n_labeled=int(lab.sum()),
                         cos_raw=mean_cosine(emb[unl], ref_emb[unl]),
                         cos_aligned_unlabeled=mean_cosine(al_full[unl], ref_emb[unl]),
                         resid_aligned_unlabeled=rel_residual(al_full[unl], ref_emb[unl]),
                         cos_aligned_labeled=mean_cosine(al_full[lab], ref_emb[lab]),
                         oof_pr_auc=pr, scale=fit_["s"]))
    # 폴드 모델의 OOF 확률 PR-AUC (1~34 라벨 노드 전체, 각 노드는 자기 정답을 안 본 모델의 예측)
    lab_all = (y != 2) & (step <= VAL_END)
    pooled = float(average_precision_score(y[lab_all], oof_prob[lab_all]))
    for r in rows:
        r["oof_pr_auc_pooled"] = pooled
    return out, rows


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="OOF 임베딩 생성 (하이브리드 누수 처리)")
    ap.add_argument("--model", required=True, choices=["gcn", "sage", "gat"])
    ap.add_argument("--data-root", default="data/elliptic")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--best-epochs", type=int, nargs="+", default=None,
                    help="시드별 학습 에폭 (--seeds와 같은 순서). 없으면 --metrics-csv의 best_epoch를 사용")
    ap.add_argument("--metrics-csv", default=None, help="기본: results/metrics/{model}.csv")
    ap.add_argument("--ref-emb-dir", default=str(ROOT / "results" / "emb"),
                    help="기준 임베딩(저장된 전체 재학습 모델의 임베딩) 폴더: emb_{model}_seed{n}.npy")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "emb_oof"))
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)

    if args.best_epochs is not None:
        if len(args.best_epochs) != len(args.seeds):
            raise SystemExit("--best-epochs 개수가 --seeds 개수와 달라요.")
        epochs = dict(zip(args.seeds, args.best_epochs))
    else:
        epochs = best_epochs_from_metrics(args.model, args.seeds, args.metrics_csv or RESULTS_DIR / "metrics" / f"{args.model}.csv")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"장치: {device} | 모델: {args.model} | lr={args.lr} dropout={args.dropout} wd={args.weight_decay} hidden={args.hidden} | 폴드 {args.folds}개: "
          + " / ".join(f"{s[0]}~{s[-1]}" for s in time_folds(args.folds)))
    print("시드별 학습 에폭:", epochs)

    data = load_data(args.data_root, standardize=True)
    g = to_pyg(data).to(device)
    g.x, g.y = g.x.float(), g.y.long()
    factory = get_factory(args.model)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    for seed in args.seeds:
        ref_path = Path(args.ref_emb_dir) / f"emb_{args.model}_seed{seed}.npy"
        if not ref_path.exists():
            raise FileNotFoundError(f"기준 임베딩이 없습니다: {ref_path}")
        ref = np.load(ref_path)
        if ref.ndim != 2 or ref.shape[0] != data.num_nodes:
            raise ValueError(f"{ref_path.name}: 모양 {ref.shape} (노드 수 {data.num_nodes}개 필요)")
        emb, rows = oof_embeddings(factory, args, data, g, ref, seed, epochs[seed], device, args.folds)
        np.save(out_dir / f"emb_oof_{args.model}_seed{seed}.npy", emb.astype(np.float32))
        report += [dict(model=args.model, **r) for r in rows]
        df = pd.DataFrame(rows)
        print(f"[seed {seed}] 에폭 {epochs[seed]} | 정렬 후 코사인(라벨 없는 노드) {df.cos_aligned_unlabeled.mean():.3f} "
              f"(정렬 전 {df.cos_raw.mean():.3f}) | 라벨 있는 노드 {df.cos_aligned_labeled.mean():.3f} | "
              f"폴드 모델의 OOF PR-AUC {df.oof_pr_auc_pooled.iloc[0]:.4f}")
    rep_path = out_dir / f"oof_report_{args.model}.csv"
    new = pd.DataFrame(report)
    if rep_path.exists():
        old = pd.read_csv(rep_path)
        new = pd.concat([old[~old["seed"].isin(args.seeds)], new], ignore_index=True)
    new.to_csv(rep_path, index=False)
    print("저장:", out_dir)


if __name__ == "__main__":
    main()
