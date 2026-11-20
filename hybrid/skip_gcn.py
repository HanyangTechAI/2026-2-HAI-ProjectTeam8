"""Skip-GCN 하이브리드: GNN 출력 + 원본 피처 → XGBoost  (누수 방지: 시간 단계 Out-of-Fold 스태킹)

[왜 OOF인가]
results/emb/의 임베딩은 시간 단계 1~34의 정답으로 학습(Train+Val 재학습)한 GNN에서 나왔다.
그대로 XGBoost에 붙이면 학습·검증 노드의 임베딩에 정답이 새겨져 있어
검증 PR-AUC가 ~1.0이 되고 테스트에서는 그 효과가 사라진다 (= 학습 쪽 누수).

[방법]
Elliptic은 간선이 같은 시간 단계 안에서만 존재한다(scripts/verify_data.py 확인).
  1. 시간 단계 1~34를 연속된 K=5개 묶음(fold)으로 나눈다: 1~7 / 8~14 / 15~21 / 22~28 / 29~34.
  2. fold k마다 나머지 fold의 정답으로만 GNN을 학습해 fold k 노드의 출력(임베딩·확률)을 만든다.
     fold k 노드와 그 이웃은 모두 같은 시간 단계 → 이 모델은 fold k의 정답을 전혀 보지 않는다.
  3. 테스트(35~49) 노드는 K개 fold 모델 출력의 평균.
  4. [원본 피처 + OOF 출력]으로 XGBoost를 베이스라인과 같은 절차·설정(고정)으로 학습한다.

[비교하는 하이브리드 (같은 fold 모델에서 한 번에 생성)]
  hybrid_{gnn}_oofemb          : OOF 64차원 임베딩 그대로 (팀 결정 방식)
  hybrid_{gnn}_oofemb_aligned  : fold마다 다른 임베딩 좌표계를 fold 1 모델 좌표계로 선형 정렬한 뒤 사용.
                                 정렬 변환은 '미분류 노드'(정답 없음)만으로 학습 → 누수 없음
  hybrid_{gnn}_oofprob         : 임베딩 대신 OOF 불법 확률 1열 (좌표계 문제 없음)
  hybrid_{gnn}_leaky           : (--with-leaky) results/emb의 기존 임베딩 그대로 = 누수 있음, 비교용
좌표계 문제: fold마다 따로 학습한 모델은 임베딩의 같은 열이 서로 다른 뜻일 수 있어
XGBoost가 한 fold에서 배운 규칙이 다른 fold·테스트에 맞지 않을 수 있다.

실행 예:
  python -m hybrid.skip_gcn --gnn sage --data-root data/elliptic
  python -m hybrid.skip_gcn --gnn gcn --lr 0.005 --weight-decay 5e-4
저장(하이브리드마다): results/metrics/{name}.csv, results/pred/pred_{name}_seed{n}.npy, results/timestep_f1_{name}.csv
      GNN OOF 출력: results/oof/{oofprob,oofemb,oofemb_aligned}_{gnn}_seed{n}.npy (SHAP 등에서 재사용)
빠른 점검: python -m hybrid.skip_gcn --gnn sage --seeds 0 --epochs 3 --results-dir /tmp/x
"""
from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path

import xgboost  # noqa: F401  (macOS libomp 로드 순서)
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.xgb_baseline import BASE_PARAMS, SPACE, make_clf, pos_weight  # noqa: E402
from common.data import VAL_END, load_data, to_pyg  # noqa: E402
from common.metrics import best_threshold, evaluate, evaluate_by_timestep  # noqa: E402
from common.train_gnn import class_weight, fit, predict  # noqa: E402
from common.utils import RESULTS_DIR, Timer, save_array, save_rows, set_seed  # noqa: E402

# 각 조의 최종 하이퍼파라미터 (README 기준). dropout·에폭은 results/metrics/{gnn}.csv의 notes에서 읽는다.
GNN_DEFAULTS = {
    "gcn": dict(cls="GCN", lr=0.005, weight_decay=5e-4),
    "sage": dict(cls="SAGE", lr=0.005, weight_decay=1e-3),
    "gat": dict(cls="GAT", lr=0.005, weight_decay=5e-4),
}
DEFAULT_FOLDS = [(1, 7), (8, 14), (15, 21), (22, 28), (29, 34)]


def baseline_xgb_params(results_dir: Path) -> dict:
    """베이스라인 튜닝에서 선택된 설정 (results/xgb_tuning.csv 1행). 하이브리드에서도 고정."""
    best = pd.read_csv(results_dir / "xgb_tuning.csv").iloc[0]
    p = {k: best[k] for k in SPACE}
    p = {k: (int(v) if k in ("max_depth", "min_child_weight") else float(v)) for k, v in p.items()}
    return {**BASE_PARAMS, **p}


def gnn_run_settings(gnn: str, results_dir: Path) -> dict[int, dict]:
    """시드별 best_epoch, dropout (최종 GNN 실행의 notes에서)."""
    df = pd.read_csv(results_dir / "metrics" / f"{gnn}.csv")
    df = df[(df.split == "test") & (df.threshold_type == "val_opt")]
    out = {}
    for _, r in df.iterrows():
        kv = dict(re.findall(r"(\w+)=([\d.]+)", str(r.notes)))
        out[int(r.seed)] = dict(epochs=int(kv["best_epoch"]), dropout=float(kv.get("dropout", 0.5)),
                                hidden=int(kv.get("hidden", 64)))
    return out


def train_fold_models(data, g, gnn_cls, cfg, folds, seed, device):
    """fold마다 '그 fold를 뺀 1~34 라벨'로 GNN을 학습. 반환: [(hold_mask, 확률[N], 임베딩[N, d])]"""
    ts = data.time_step
    labeled = data.y != 2
    in_dim = g.x.shape[1]
    out = []
    for lo, hi in folds:
        hold = (ts >= lo) & (ts <= hi)
        train = labeled & (ts <= VAL_END) & ~hold
        set_seed(seed)
        model = gnn_cls(in_dim, cfg["hidden"], cfg["dropout"]).to(device)
        fit(model, g, torch.from_numpy(train).to(device), class_weight(data.y[train], device),
            cfg["lr"], cfg["weight_decay"], cfg["epochs"])
        p, emb = predict(model, g)
        out.append((hold, p.astype(np.float32), emb.astype(np.float32)))
    return out


def assemble_oof(data, parts: list[np.ndarray], holds: list[np.ndarray]) -> np.ndarray:
    """1~34 노드는 자기 fold가 빠진 모델의 값, 35~49 노드는 fold 모델 평균."""
    first = parts[0]
    oof = np.full(first.shape, np.nan, dtype=np.float32)
    for hold, v in zip(holds, parts):
        oof[hold] = v[hold]
    late = data.time_step > VAL_END
    oof[late] = np.mean([v[late] for v in parts], axis=0)
    assert not np.isnan(oof).any()
    return oof


def align_to_reference(embs: list[np.ndarray], fit_mask: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
    """fold 모델마다 다른 임베딩 좌표계를 첫 fold 모델의 좌표계로 맞춘다.

    선형 변환 W (+ 절편)를 '미분류 노드'(정답 없음)의 임베딩만으로 최소제곱 학습 → 라벨을 쓰지 않으므로 누수 없음.
    반환: 정렬된 임베딩 목록, fold별 정렬 R² (1에 가까울수록 두 좌표계가 선형으로 잘 대응)
    """
    ref = embs[0]
    out, r2 = [ref], [1.0]
    yref = ref[fit_mask]
    for e in embs[1:]:
        A = np.hstack([e[fit_mask], np.ones((fit_mask.sum(), 1), np.float32)])
        W, *_ = np.linalg.lstsq(A, yref, rcond=None)
        aligned = (np.hstack([e, np.ones((len(e), 1), np.float32)]) @ W).astype(np.float32)
        resid = ((aligned[fit_mask] - yref) ** 2).sum()
        r2.append(float(1 - resid / ((yref - yref.mean(0)) ** 2).sum()))
        out.append(aligned)
    return out, r2


def run_xgb(name, X, data, params, seed, results_dir, extra_notes="") -> list[dict]:
    """xgb_baseline.run_seed와 같은 절차 (모델 이름·입력만 다름)."""
    set_seed(seed)
    y = data.y
    tr, va, te = data.train_mask, data.val_mask, data.test_mask
    clf = make_clf(params, pos_weight(y[tr]), seed, -1, early_stopping=True)
    clf.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    n_trees = int(clf.best_iteration) + 1
    p_val = clf.predict_proba(X[va])[:, 1]
    thr = best_threshold(y[va], p_val)
    p_tr = clf.predict_proba(X[tr])[:, 1]

    trval = tr | va
    final = make_clf({**params, "n_estimators": n_trees}, pos_weight(y[trval]), seed, -1, early_stopping=False)
    with Timer() as t_train:
        final.fit(X[trval], y[trval], verbose=False)
    times = []
    for _ in range(5):
        with Timer() as t_inf:
            p_test = final.predict_proba(X[te])[:, 1]
        times.append(t_inf.sec)
    n_params = int(len(final.get_booster().trees_to_dataframe()))
    notes = f"trees={n_trees};train_pr_auc={evaluate(y[tr], p_tr)['pr_auc']:.4f}{extra_notes}"
    common = dict(model=name, seed=seed, train_sec=t_train.sec, infer_sec=float(np.median(times)),
                  n_params=n_params, device="cpu", notes=notes)
    rows = [{**common, "split": "test", "threshold_type": tt, **evaluate(y[te], p_test, th)}
            for tt, th in (("fixed_0.5", 0.5), ("val_opt", thr))]
    rows.append({"model": name, "seed": seed, "device": "cpu", "notes": notes,
                 "split": "val", "threshold_type": "val_opt", **evaluate(y[va], p_val, thr)})

    save_array(final.predict_proba(X)[:, 1].astype(np.float32), "pred", name, seed, results_dir)
    by_step = evaluate_by_timestep(y[te], p_test, data.time_step[te], thr)
    step_df = pd.DataFrame([dict(model=name, seed=seed, time_step=s, f1=f) for s, f in by_step.items()])
    out = Path(results_dir or RESULTS_DIR) / f"timestep_f1_{name}.csv"
    if out.exists():
        old = pd.read_csv(out)
        step_df = pd.concat([old[old["seed"] != seed], step_df], ignore_index=True)
    step_df.to_csv(out, index=False)
    print(f"  [{name} seed {seed}] trees={n_trees} train PR-AUC={evaluate(y[tr], p_tr)['pr_auc']:.4f} "
          f"val PR-AUC={rows[2]['pr_auc']:.4f} | test PR-AUC={rows[0]['pr_auc']:.4f} "
          f"F1@val_opt={rows[1]['f1']:.4f} (thr={thr:.3f})", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn", default="sage", choices=list(GNN_DEFAULTS))
    ap.add_argument("--data-root", default="data/elliptic")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--with-leaky", action="store_true", help="기존 임베딩을 그대로 붙인 누수 하이브리드도 기록")
    ap.add_argument("--results-dir", default=None, help="출력 위치 (입력 설정은 항상 results/에서 읽음)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--epochs", type=int, default=None, help="빠른 점검용: fold 학습 에폭 강제 지정")
    args = ap.parse_args(argv)

    src = Path(RESULTS_DIR)                      # 입력: 베이스라인 설정, GNN 설정, 임베딩
    out_dir = Path(args.results_dir) if args.results_dir else src
    dflt = GNN_DEFAULTS[args.gnn]
    gnn_cls = getattr(importlib.import_module(f"models.{args.gnn}"), dflt["cls"])
    lr = args.lr if args.lr is not None else dflt["lr"]
    wd = args.weight_decay if args.weight_decay is not None else dflt["weight_decay"]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    data = load_data(args.data_root)
    g = to_pyg(data).to(device)
    # 누수 방지 전제 확인: 모든 간선이 같은 시간 단계 안에 있어야 fold 경계를 넘는 정보 흐름이 없다
    s, d = data.edge_index
    assert (data.time_step[s] == data.time_step[d]).all(), "시간 단계를 넘는 간선이 있음 → fold 분리 불가"

    params = baseline_xgb_params(src)
    runs = gnn_run_settings(args.gnn, src)
    print(f"GNN={args.gnn} lr={lr} wd={wd} folds={DEFAULT_FOLDS} device={device}")
    print(f"XGBoost 설정(베이스라인 고정): { {k: params[k] for k in SPACE} }")

    G = args.gnn
    variants = {f"hybrid_{G}_oofemb": [], f"hybrid_{G}_oofemb_aligned": [], f"hybrid_{G}_oofprob": []}
    if args.with_leaky:
        variants[f"hybrid_{G}_leaky"] = []
    oof_test, align_r2 = [], []
    unlabeled = data.y == 2
    (out_dir / "oof").mkdir(parents=True, exist_ok=True)   # common.save_array는 pred/emb 전용
    for seed in args.seeds:
        cfg = dict(**runs[seed], lr=lr, weight_decay=wd)
        if args.epochs:
            cfg["epochs"] = args.epochs
        with Timer() as t_gnn:
            fm = train_fold_models(data, g, gnn_cls, cfg, DEFAULT_FOLDS, seed, device)
        holds = [h for h, _, _ in fm]
        prob = assemble_oof(data, [p[:, None] for _, p, _ in fm], holds)[:, 0]
        emb_raw = assemble_oof(data, [e for _, _, e in fm], holds)
        aligned, r2 = align_to_reference([e for _, _, e in fm], unlabeled)
        emb_al = assemble_oof(data, aligned, holds)
        align_r2.append(r2)
        np.save(out_dir / "oof" / f"oofprob_{G}_seed{seed}.npy", prob)
        np.save(out_dir / "oof" / f"oofemb_{G}_seed{seed}.npy", emb_raw)
        np.save(out_dir / "oof" / f"oofemb_aligned_{G}_seed{seed}.npy", emb_al)
        o = evaluate(data.y[data.test_mask], prob[data.test_mask])
        oof_test.append(o["pr_auc"])
        print(f"[seed {seed}] GNN {len(DEFAULT_FOLDS)}-fold ({cfg['epochs']} epochs, {t_gnn.sec:.0f}s) | "
              f"OOF 확률 단독 test PR-AUC={o['pr_auc']:.4f} | 정렬 R² {[round(x, 3) for x in r2]}", flush=True)
        note = f";gnn_oof_sec={t_gnn.sec:.1f};folds={len(DEFAULT_FOLDS)}"
        feats = {f"hybrid_{G}_oofemb": emb_raw, f"hybrid_{G}_oofemb_aligned": emb_al,
                 f"hybrid_{G}_oofprob": prob[:, None]}
        if args.with_leaky:
            feats[f"hybrid_{G}_leaky"] = np.load(src / "emb" / f"emb_{G}_seed{seed}.npy")
        for name, F_ in feats.items():
            variants[name] += run_xgb(name, np.hstack([data.x, F_]), data, params, seed, out_dir,
                                      extra_notes="" if name.endswith("leaky") else note)

    for name, rows in variants.items():
        save_rows(name, rows, out_dir)

    def summary(rows, label):
        df = pd.DataFrame(rows)
        t = df[(df.split == "test") & (df.threshold_type == "val_opt")]
        v = df[df.split == "val"]
        print(f"{label:30s} test PR-AUC {t.pr_auc.mean():.3f} ± {t.pr_auc.std():.3f} | "
              f"F1(val_opt) {t.f1.mean():.3f} ± {t.f1.std():.3f} | val PR-AUC {v.pr_auc.mean():.3f}")
    print("\n=== 요약 (시드 평균 ± 표준편차) ===")
    print(f"{'GNN OOF 확률 단독':30s} test PR-AUC {np.mean(oof_test):.3f} ± "
          f"{np.std(oof_test, ddof=1) if len(oof_test) > 1 else 0:.3f}")
    print(f"{'임베딩 정렬 R² (fold 2~5 평균)':30s} {np.mean([r[1:] for r in align_r2]):.3f}")
    for name, rows in variants.items():
        summary(rows, name)


if __name__ == "__main__":
    main()
