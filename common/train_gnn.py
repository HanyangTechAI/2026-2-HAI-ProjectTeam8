"""GNN 공통 학습·평가 루프 (GCN · GraphSAGE · GAT 공용).  torch / torch_geometric 필요.

새 모델은 `models/xxx.py`에서 모델 클래스만 만들고 `main()`에 팩토리를 넘기면 된다 (models/gcn.py 참고).
모델 규칙: forward(x, edge_index) -> (logits[N, 2], emb[N, 64]).

학습 절차 (XGBoost 베이스라인과 동일한 규칙):
  ① Train(1~29)으로 학습, Val(30~34) PR-AUC로 조기 종료 → best_epoch, 검증 기반 임계값
  ② 새 모델을 Train+Val(1~34)로 best_epoch만큼 재학습 (시간 측정 대상)
  ③ 테스트(35~49)는 ②의 모델로 한 번만 평가 (임계값: 0.5 고정 / 검증 최적)
저장: results/metrics/{model}.csv, results/pred/pred_{model}_seed{n}.npy,
      results/emb/emb_{model}_seed{n}.npy (git 제외), results/timestep_f1_{model}.csv

실행 예: python -m models.gcn --data-root data/elliptic --seeds 0 1 2 3 4
빠른 점검: python -m models.gcn --data-root data/elliptic --seeds 0 --max-epochs 5
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.data import EllipticData, load_data, split_summary, to_pyg  # noqa: E402
from common.metrics import best_threshold, evaluate, evaluate_by_timestep  # noqa: E402
from common.utils import RESULTS_DIR, Timer, save_array, save_rows, set_seed  # noqa: E402


def class_weight(y: np.ndarray, device) -> torch.Tensor:
    """[1, 정상 수/불법 수]  (XGBoost의 scale_pos_weight와 같은 값)"""
    n_ill = max(int((y == 1).sum()), 1)
    return torch.tensor([1.0, float((y == 0).sum()) / n_ill], dtype=torch.float32, device=device)


@torch.no_grad()
def predict(model, g):
    """전체 노드에 대한 (불법 확률 [N], 임베딩 [N, d])"""
    model.eval()
    logits, emb = model(g.x, g.edge_index)
    return torch.softmax(logits, dim=1)[:, 1].cpu().numpy(), emb.cpu().numpy()


def fit(model, g, train_mask, weight, lr, weight_decay, epochs,
        val_mask=None, patience=None):
    """full-batch 학습. val_mask가 있으면 검증 PR-AUC로 조기 종료하고 최고 시점의 가중치를 복원한다.

    반환: (best_epoch, best_val_pr_auc)  — val_mask가 없으면 (epochs, nan)
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    y_val = g.y[val_mask].cpu().numpy() if val_mask is not None else None
    best_score, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        logits, _ = model(g.x, g.edge_index)
        loss = F.cross_entropy(logits[train_mask], g.y[train_mask], weight=weight)
        loss.backward()
        opt.step()
        if val_mask is None:
            continue
        p, _ = predict(model, g)
        score = float(average_precision_score(y_val, p[val_mask.cpu().numpy()]))
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif patience is not None and epoch - best_epoch >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        return best_epoch, best_score
    return epochs, float("nan")


def run_seed(name, factory, data: EllipticData, g, args, seed, device, device_name) -> list[dict]:
    in_dim = g.x.shape[1]
    y_np = data.y
    tr, va, te = data.train_mask, data.val_mask, data.test_mask
    tr_t, va_t = g.train_mask, g.val_mask
    trval_t = g.train_mask | g.val_mask

    # ① Train으로 학습 + 검증 조기 종료
    set_seed(seed)
    model = factory(in_dim, args).to(device)
    best_epoch, val_score = fit(model, g, tr_t, class_weight(y_np[tr], device), args.lr,
                                args.weight_decay, args.max_epochs, val_mask=va_t, patience=args.patience)
    p_val_all, _ = predict(model, g)
    p_val = p_val_all[va]
    thr = best_threshold(y_np[va], p_val)

    # ② Train+Val 재학습 (best_epoch 고정), 시간 측정
    set_seed(seed)
    final = factory(in_dim, args).to(device)
    with Timer() as t_train:
        fit(final, g, trval_t, class_weight(y_np[tr | va], device), args.lr,
            args.weight_decay, best_epoch)
    times = []
    for _ in range(5):
        with Timer() as t_inf:
            p_all, emb = predict(final, g)
        times.append(t_inf.sec)
    infer_sec = float(np.median(times))
    n_params = int(sum(p.numel() for p in final.parameters()))

    # ③ 테스트 1회 평가
    p_test, y_test = p_all[te], y_np[te]
    common = dict(model=name, seed=seed, train_sec=t_train.sec, infer_sec=infer_sec,
                  n_params=n_params, device=device_name,
                  notes=f"best_epoch={best_epoch};hidden={args.hidden};dropout={args.dropout}")
    rows = []
    for ttype, th in (("fixed_0.5", 0.5), ("val_opt", thr)):
        rows.append({**common, "split": "test", "threshold_type": ttype, **evaluate(y_test, p_test, th)})
    # 검증 행: ①(Train만 학습)의 모델 기준, 참고용
    rows.append({"model": name, "seed": seed, "device": device_name, "notes": common["notes"],
                 "split": "val", "threshold_type": "val_opt", **evaluate(y_np[va], p_val, thr)})

    save_array(p_all, "pred", name, seed, args.results_dir)
    save_array(emb, "emb", name, seed, args.results_dir)

    by_step = evaluate_by_timestep(y_test, p_test, data.time_step[te], thr)
    step_df = pd.DataFrame([dict(model=name, seed=seed, time_step=s, f1=f) for s, f in by_step.items()])
    out = Path(args.results_dir or RESULTS_DIR) / f"timestep_f1_{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        old = pd.read_csv(out)
        step_df = pd.concat([old[old["seed"] != seed], step_df], ignore_index=True)
    step_df.to_csv(out, index=False)

    print(f"[seed {seed}] best_epoch={best_epoch} val PR-AUC={val_score:.4f} | "
          f"test PR-AUC={rows[0]['pr_auc']:.4f}  F1@0.5={rows[0]['f1']:.4f}  "
          f"F1@val_opt={rows[1]['f1']:.4f} (thr={thr:.3f})  train={t_train.sec:.1f}s infer={infer_sec * 1000:.0f}ms")
    return rows


def main(name: str, factory, argv=None):
    """factory(in_dim, args) -> nn.Module.  각 models/*.py의 __main__에서 호출한다."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/elliptic")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--hidden", type=int, default=64, help="임베딩 차원 (계획서: 64로 통일)")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--cpu", action="store_true", help="GPU가 있어도 CPU 사용")
    args = ap.parse_args(argv)

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    device_name = torch.cuda.get_device_name(0) if use_cuda else "cpu"
    print(f"장치: {device_name}")

    data = load_data(args.data_root, standardize=True)
    print(split_summary(data).to_string(index=False))
    g = to_pyg(data).to(device)
    g.x = g.x.float()
    g.y = g.y.long()

    all_rows = []
    for seed in args.seeds:
        all_rows += run_seed(name, factory, data, g, args, seed, device, device_name)
    path = save_rows(name, all_rows, args.results_dir)
    print("저장:", path)
