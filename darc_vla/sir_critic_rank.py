# Copyright 2026 The DARC-VLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SIR ranking learner -- listwise Best-of-K / pairwise over existing tuples.

Follow-up to sir_critic_select.py (pointwise BCE-Q failed: eval AUROC 0.16,
Q-select 0/3, while Best-of-K 2/3 proves recoverable delta exist in the cheap
proposals).  GPT review (2026-08-12): do NOT solve "how to train Q", solve
"what structure do good interventions have".  Cheapest first, zero new
rollouts -- re-learn on the saved (x, delta, R) tuples:

  A. relative objective within a context instead of absolute P(R=1|s,delta):
       listwise Best-of-K (softmax cross-entropy vs the success indicator
       distribution, i.e. train objective == final usage argmax S(s,delta_k))
       and pairwise margin   L = softplus(m - S(s,delta+) + S(s,delta-))
     and a TINY bilinear compatibility scorer
       S(s, delta) = f(ctx)^T g(delta),  ctx = [state_C (8), A_C0 (35)],
     f,g are single linear maps to emb-dim (8/16).  ~3 orders of magnitude
     fewer parameters than the 50->128->128->1 concat-MLP.
  B. strict leave-trajectory-out; the metric is HELD-OUT TOP-1 CANDIDATE
     SUCCESS (argmax S over the K proposals -> did that delta rescue?), not
     BCE loss / AUROC.  Random / min-norm / best-of-K give the bounds.
     (min-norm is a zero-parameter selector: if it beats learned Q, the basin
     is wide and the real structure is "reject the harmful direction", which
     is GPT's rejection-model route.)

If held-out Top-1 is still ~0 (the collection executed every eval candidate,
so Top-1 is read off the checkpoint -- no new rollouts), the neural selector
is dropped and we move to the basin-geometry matrices (transfer matrix /
intervention-primitive map), which decide retrieval-SIR vs primitive-SIR vs
rejection-SIR.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_data(path):
    d = np.load(path)
    n = len(d["trial"])
    rows = []
    for i in range(n):
        rows.append({
            "trial": int(d["trial"][i]),
            "in_train": bool(d["in_train"][i]),
            "state": np.asarray(d["state"][i], dtype=np.float32),
            "base_chunk": np.asarray(d["base_chunk"][i], dtype=np.float32),
            "delta": np.asarray(d["delta"][i], dtype=np.float32),
            "sigma": float(d["sigma"][i]),
            "R": bool(d["R"][i]),
        })
    with open(path.replace(".npz", "_meta.json")) as f:
        meta = json.load(f)
    return rows, meta


def _ctx(r):
    return np.concatenate([r["state"], r["base_chunk"].reshape(-1)])


class Compatibility(nn.Module):
    """S(s, delta) = f(ctx)^T g(delta); tiny bilinear by design."""

    def __init__(self, ctx_dim, delta_dim, emb):
        super().__init__()
        self.f = nn.Linear(ctx_dim, emb)
        self.g = nn.Linear(delta_dim, emb)

    def forward(self, ctx, delta):
        return (self.f(ctx) * self.g(delta)).sum(-1)


def _group(rows):
    """rows -> {trial: (ctx, delta, R)} with trial order preserved."""
    groups = {}
    for r in rows:
        groups.setdefault(r["trial"], []).append(r)
    return groups


def _standardize(fit_rows, ctxs, deltas):
    ctx0 = np.stack([_ctx(r) for r in fit_rows])
    d0 = np.stack([r["delta"] for r in fit_rows])
    mu_c, sd_c = ctx0.mean(0), ctx0.std(0) + 1e-6
    mu_d, sd_d = d0.mean(0), d0.std(0) + 1e-6
    return (ctxs - mu_c) / sd_c, (deltas - mu_d) / sd_d


def _auroc(y, scores):
    y = np.asarray(y)
    s = np.asarray(scores)
    order = np.argsort(s, kind="mergesort")
    y = y[order]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.arange(1, len(y) + 1, dtype=np.float64)
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _run_fold(rows, meta, args, tr_trials, ev_trials):
    """Train on tr_trials, evaluate Top-1 success on ev_trials.  Returns
    (sel_rows, dict) with per-eval-trial selection and pooled aggregates."""
    tr_rows = [r for r in rows if r["trial"] in tr_trials]
    ev_rows = [r for r in rows if r["trial"] in ev_trials]
    if not tr_rows or not ev_rows:
        return [], {}

    ctx_tr = np.stack([_ctx(r) for r in tr_rows])
    delta_tr = np.stack([r["delta"] for r in tr_rows])
    ctx_tr_s, delta_tr_s = _standardize(tr_rows, ctx_tr, delta_tr)
    ctx_ev_s, delta_ev_s = _standardize(
        tr_rows, np.stack([_ctx(r) for r in ev_rows]),
        np.stack([r["delta"] for r in ev_rows]))

    tr_by_trial = {}
    for i, r in enumerate(tr_rows):
        tr_by_trial.setdefault(r["trial"], []).append(i)
    tr_ctx_groups = [ctx_tr_s[tr_by_trial[t]] for t in tr_trials]
    tr_delta_groups = [delta_tr_s[tr_by_trial[t]] for t in tr_trials]
    tr_R_groups = [[float(tr_rows[i]["R"]) for i in tr_by_trial[t]]
                   for t in tr_trials]

    model = Compatibility(43, 7, args.emb)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    device = "cpu"
    model.train()
    for _ in range(args.epochs):
        opt.zero_grad()
        if args.loss == "listwise":
            loss = _loss_listwise_tensors(
                model, tr_ctx_groups, tr_delta_groups, tr_R_groups,
                args.margin, args.w_supp, device)
        else:
            loss, _ = _loss_pairwise_tensors(
                model, tr_ctx_groups, tr_delta_groups, tr_R_groups,
                args.margin, device)
        loss.backward()
        opt.step()
    model.eval()

    rng = np.random.default_rng(args.seed)
    sel, y_all, s_all = [], [], []
    for t in ev_trials:
        g = [r for r in ev_rows if r["trial"] == t]
        gi = [i for i, r in enumerate(ev_rows) if r["trial"] == t]
        ctx = torch.tensor(ctx_ev_s[gi])
        delta = torch.tensor(delta_ev_s[gi])
        with torch.no_grad():
            s = model(ctx, delta).numpy()
        R = np.asarray([float(r["R"]) for r in g])
        top1 = bool(R[int(np.argmax(s))])
        rand = bool(R[int(rng.integers(0, len(g)))])
        minnorm = bool(R[int(np.argmin(np.linalg.norm(delta.numpy(), axis=1)))])
        best = bool(R.sum() > 0)
        order = np.argsort(s, kind="mergesort")[::-1]
        half = max(1, len(order) // 2)
        top_ok = sum(1 for i in order[:half] if R[int(i)] > 0)
        bot_ok = sum(1 for i in order[-half:] if R[int(i)] > 0)
        sel.append({"trial": t, "top1_Q": top1, "random": rand,
                    "min_norm": minnorm, "best_of_K": best,
                    "topQ_recovery": top_ok / half,
                    "botQ_recovery": bot_ok / half})
        y_all.append(R)
        s_all.append(s)
    y_all = np.concatenate(y_all)
    s_all = np.concatenate(s_all)
    n = len(sel)
    agg = {
        "n_eval_trials": n,
        "sr_top1_Q": sum(s["top1_Q"] for s in sel) / n if n else None,
        "sr_random": sum(s["random"] for s in sel) / n if n else None,
        "sr_min_norm": sum(s["min_norm"] for s in sel) / n if n else None,
        "sr_best_of_K": sum(s["best_of_K"] for s in sel) / n if n else None,
        "sr_random_exp": float(y_all.mean()) if len(y_all) else None,
        "auroc_eval": float(_auroc(y_all, s_all)) if len(y_all) else None,
    }
    return sel, agg


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rows, meta = _load_data(args.data_npz)
    train_trials = [int(t) for t in meta["train_trials"]]
    eval_trials = [int(t) for t in meta["eval_trials"]]

    per_trial = {}
    for r in rows:
        per_trial.setdefault(r["trial"], []).append(r)

    print(f"[rank] train trials {train_trials} | eval trials {eval_trials} "
          f"({args.loss}, emb {args.emb})")
    for t in sorted(per_trial):
        g = per_trial[t]
        pos = sum(r["R"] for r in g)
        tag = "EVAL" if t in eval_trials else "TRAIN"
        print(f"[rank] trial {t}: positives {pos}/{len(g)} {tag}")

    if args.cv:
        # leave-one-train-trial-out: 10 eval points instead of the fixed 3
        all_sel, all_agg = [], []
        for hold in train_trials:
            s, a = _run_fold(rows, meta, args,
                             [t for t in train_trials if t != hold], [hold])
            all_sel.extend(s)
            all_agg.append(a)
        n = len(all_sel)
        sr_q = sum(x["top1_Q"] for x in all_sel) / n
        sr_rand = sum(x["random"] for x in all_sel) / n
        sr_min = sum(x["min_norm"] for x in all_sel) / n
        sr_best = sum(x["best_of_K"] for x in all_sel) / n
        sr_rand_exp = sum(a["sr_random_exp"] for a in all_agg) / len(all_agg)
        auroc_ev = sum(a["auroc_eval"] for a in all_agg) / len(all_agg)
        top_pool = sum(x["topQ_recovery"] for x in all_sel) / n
        bot_pool = sum(x["botQ_recovery"] for x in all_sel) / n
        print(f"[rank CV] SR_Top1-Q {sr_q:.3f} | SR_Random {sr_rand_exp:.3f} "
              f"(draw {sr_rand:.3f}) | SR_min-norm {sr_min:.3f} | "
              f"SR_Best-of-K {sr_best:.3f} | AUROC {auroc_ev:.3f}")
        print(f"[rank CV] topQ {top_pool:.3f} vs botQ {bot_pool:.3f}")
        print(f"[rank CV] per-holdout: " + ", ".join(
            f"{x['trial']}:{'✓' if x['top1_Q'] else '✗'}" for x in all_sel))
        report = {
            "mode": "cv", "loss": args.loss, "emb": args.emb,
            "sr_top1_Q": sr_q, "sr_random": sr_rand_exp,
            "sr_random_draw": sr_rand, "sr_min_norm": sr_min,
            "sr_best_of_K": sr_best, "auroc_eval": auroc_ev,
            "topQ_recovery": top_pool, "botQ_recovery": bot_pool,
            "selection_rows": all_sel,
        }
    else:
        sel, agg = _run_fold(rows, meta, args, train_trials, eval_trials)
        for x in sel:
            print(f"[rank] trial {x['trial']}: top1-Q {x['top1_Q']} | "
                  f"random {x['random']} | min-norm {x['min_norm']} | "
                  f"best-of-K {x['best_of_K']}")
        print(f"[rank] SR_Top1-Q {agg['sr_top1_Q']:.3f} | "
              f"SR_Random {agg['sr_random_exp']:.3f} "
              f"(draw {agg['sr_random']:.3f}) | SR_min-norm "
              f"{agg['sr_min_norm']:.3f} | SR_Best-of-K {agg['sr_best_of_K']:.3f} "
              f"| AUROC {agg['auroc_eval']:.3f}")
        top_pool = sum(x["topQ_recovery"] for x in sel) / len(sel)
        bot_pool = sum(x["botQ_recovery"] for x in sel) / len(sel)
        print(f"[rank] topQ {top_pool:.3f} vs botQ {bot_pool:.3f}")
        report = {
            "mode": "fixed", "loss": args.loss, "emb": args.emb,
            "train_trials": train_trials, "eval_trials": eval_trials,
            "sr_top1_Q": agg["sr_top1_Q"], "sr_random": agg["sr_random_exp"],
            "sr_random_draw": agg["sr_random"], "sr_min_norm": agg["sr_min_norm"],
            "sr_best_of_K": agg["sr_best_of_K"], "auroc_eval": agg["auroc_eval"],
            "topQ_recovery": top_pool, "botQ_recovery": bot_pool,
            "selection_rows": sel,
        }

    report.update({"task": meta.get("task"),
                   "n_train_tuples": sum(1 for r in rows if r["trial"]
                                         in train_trials),
                   "n_eval_tuples": sum(1 for r in rows if r["trial"]
                                        in eval_trials)})
    print(json.dumps({k: v for k, v in report.items()
                      if k != "selection_rows"}, indent=2))
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[rank] wrote {args.out_json}")


# tensor-grouped loss helpers -------------------------------------------------

def _loss_listwise_tensors(model, ctxs, deltas, Rs, margin, w_supp, device):
    total = torch.tensor(0.0, device=device)
    for c, d, R in zip(ctxs, deltas, Rs):
        ct = torch.tensor(c, device=device)
        dt = torch.tensor(d, device=device)
        Rt = torch.tensor(R, device=device)
        s = model(ct, dt)
        n_pos = float(Rt.sum())
        if n_pos > 0:
            target = Rt / n_pos
            total = total + -(target * F.log_softmax(s, 0)).sum()
        else:
            total = total + w_supp * F.softplus(s + margin).mean()
    return total


def _loss_pairwise_tensors(model, ctxs, deltas, Rs, margin, device):
    total = torch.tensor(0.0, device=device)
    n_pairs = 0
    for c, d, R in zip(ctxs, deltas, Rs):
        ct = torch.tensor(c, device=device)
        dt = torch.tensor(d, device=device)
        s = model(ct, dt)
        R_arr = np.asarray(R)
        pos = np.where(R_arr > 0)[0]
        neg = np.where(R_arr == 0)[0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        s_pos = s[np.asarray(pos, dtype=int)][:, None]
        s_neg = s[np.asarray(neg, dtype=int)][None, :]
        total = total + F.softplus(margin - s_pos + s_neg).mean()
        n_pairs += len(pos) * len(neg)
    return total, n_pairs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-npz", required=True,
                    help="checkpoint from sir_critic_select.py "
                         "(--save-data-npz)")
    ap.add_argument("--loss", default="listwise",
                    choices=["listwise", "pairwise"])
    ap.add_argument("--emb", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--w-supp", type=float, default=0.5,
                    help="weight of suppress-all term for 0-positive contexts")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cv", action="store_true",
                    help="leave-one-train-trial-out CV (10 eval points) "
                         "instead of the fixed 3-trial holdout")
    ap.add_argument("--out-json", required=True)
    main(ap.parse_args())
