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

"""SIR-Critic + Q-select@K vs Random -- the life-or-death experiment.

Background (GPT review, 2026-08-12)
-----------------------------------
The intervention-mechanism ablations showed early recovery is mostly a *replan
trigger* (pure resample ~37.5% recovery on seed-7 failures at C=26) plus a
small-scale spatial bonus (random residual at sigma=0.15: 50%; sigma=0.3 is
worse).  A high-quality spatial correction (oracle chunk) still dominates
(67.6% best-of-C), so the open question is: can a cheap critic LEARN to pick,
out of K cheap random proposals, the residual that rescues the trajectory?

SIR-Critic (this script)
------------------------
  * a pure BCE regression:  Q(x) ~ P(R=1 | x),  x = [q_C (8), A_C0 (35), delta (7)]
    where q_C is the proprio state at the intervention step, A_C0 is pi0's own
    base action chunk, and delta is the 7-D structured residual (shared across
    the chunk) applied as  A_tilde = A_C0 + alpha * tanh(delta).
  * R is a terminal MC label in {0,1}: after the intervention the (frozen)
    base policy finishes the episode.  No TD, no target, no gamma -- there is
    nothing to bootstrap.
  * proposals: delta ~ N(0, sigma^2 I) with sigma drawn uniformly from the
    mixed scale set (0.1/0.15/0.3) to cover the perturbation scale space that
    the ablation showed is informative.

Metrics (the life-or-death contrast)
------------------------------------
  Random@1       : execute a uniformly random proposal            (baseline)
  Q-select@K     : execute argmax_Q among K proposals             (SIR)
  Best-of-K      : execute the best of the K (oracle ceiling)     (upper bound)
  Selection lift : SR(Q-select) - SR(Random)                      (must be > 0)
plus AUROC on held-out (x, R) tuples and a Top-Q vs Bottom-Q recovery split.

Protocol
--------
Phase 0: base-probe num-trials init states (seed-7 protocol, C=-1), collect the
         failed set.
Phase 1 (collection): for each failed trial, roll deterministically to the
         fixed intervention point C=26, record (state_C, base_chunk); then for
         each of K proposals sample a random delta and run ONE full rollout that
         executes A_C0 + alpha*tanh(delta) at C and hands back to base -> R.
         The collected (x, delta, R) tuples are checkpointed to --save-data-npz
         so a later crash only costs the critic/selection stage (--data-npz).
Phase 2 (critic): train the BCE critic on the TRAIN-split trials; evaluate on
         the held-out EVAL-split trials.
Phase 3 (selection): on each eval trial, score its K proposals, pick argmax,
         and compare Q-select / Random / Best-of-K using the *executed* R's.

Each candidate is genuinely executed during collection, so Q-select vs Random
is measured on real rollouts, and Best-of-K is the honest ceiling of the
proposal distribution.  Eval-trial tuples are never used for training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib

import numpy as np
import torch
import torch.nn as nn
import tqdm
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
_MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = math.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _state_from_obs(obs):
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)


def _obs(obs, prompt):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": _state_from_obs(obs),
        "prompt": prompt,
    }


def _get_libero_env(task, resolution, seed):
    task_bddl_file = (
        pathlib.Path(benchmark.get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env


def _sample_delta(rng, sigmas):
    """Random 7-D structured residual from a mixed scale set."""
    sigma = float(rng.choice(sigmas))
    delta = sigma * rng.standard_normal(7)  # NumPy2: float*float32 -> float64
    return delta.astype(np.float32), sigma


def _roll_to_C(env, policy, init_state, C, num_steps_wait, action_chunk,
               seed, prompt):
    """Roll deterministically up to the intervention step and return
    (state_C, base_chunk): the context the critic conditions on.  The pre-C
    trajectory is seed-deterministic, so every intervention rollout for this
    trial reaches the same state_C / base_chunk."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    policy.reset()
    env.reset()
    obs = env.set_init_state(init_state)

    action_plan = []
    steps = 0
    for t in range(C + num_steps_wait + action_chunk + 1):
        if t < num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            continue
        steps += 1
        if not action_plan:
            action_plan = list(policy.infer(_obs(obs, prompt))["actions"][:action_chunk])
        if (steps - 1) == C:
            base_chunk = np.asarray(
                policy.infer(_obs(obs, prompt))["actions"][:action_chunk],
                dtype=np.float32,
            )
            return _state_from_obs(obs), base_chunk
        action = action_plan.pop(0)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            raise RuntimeError(f"trajectory done before C={C}")
    raise RuntimeError(f"did not reach C={C}")


def _roll_intervene(env, policy, init_state, C, base_chunk, delta7, alpha,
                    seed, prompt, max_steps, num_steps_wait, action_chunk):
    """Full rollout; at step C execute base_chunk + alpha*tanh(delta7)
    (delta broadcast over the chunk), then hand control back to base.
    C=-1 => no intervention (plain base rollout).  Returns R in {0,1}."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    policy.reset()
    env.reset()
    obs = env.set_init_state(init_state)

    action_plan = []
    intervened = C < 0
    steps = 0
    for t in range(max_steps + num_steps_wait):
        if t < num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            continue
        steps += 1

        if not intervened and (steps - 1) == C:
            chunk = base_chunk + alpha * np.tanh(
                np.repeat(delta7[None, :], action_chunk, axis=0)
            )
            chunk = chunk.astype(np.float32)
            for a in chunk:
                obs, _, done, _ = env.step(a.tolist())
                if done:
                    return True
            intervened = True
            action_plan = []  # base replans from the post-intervention obs
            continue

        if not action_plan:
            action_plan = list(policy.infer(_obs(obs, prompt))["actions"][:action_chunk])
        action = action_plan.pop(0)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            return True
    return False


class Critic(nn.Module):
    """x -> logit P(R=1|x).  Dropout on for the small-data diagnostic."""

    def __init__(self, in_dim, dropout=0.0):
        super().__init__()
        d = lambda: (nn.Dropout(dropout) if dropout > 0 else nn.Identity())
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), d(),
            nn.Linear(128, 128), nn.ReLU(), d(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _feat_cols(feat, r):
    """Build the critic input from a collected row.  --feat selects which
    features: full (state+base+delta, 50-d), state_delta (15-d), delta (7-d).
    The 50-d input with ~80 train samples is the likely overfit culprit, so
    the cheaper feature sets are the small-data diagnostic."""
    if feat == "full":
        return np.concatenate([r["state"], r["base_chunk"].reshape(-1),
                               r["delta"]]).astype(np.float32)
    if feat == "state_delta":
        return np.concatenate([r["state"], r["delta"]]).astype(np.float32)
    if feat == "delta":
        return np.asarray(r["delta"], dtype=np.float32)
    raise ValueError(feat)


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


def _train_critic(X, y, epochs, lr, wd, seed, dropout=0.0):
    torch.manual_seed(seed)
    Xt = torch.tensor(np.asarray(X, dtype=np.float32))
    yt = torch.tensor(np.asarray(y, dtype=np.float32))
    model = Critic(Xt.shape[1], dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(Xt), yt)
        loss.backward()
        opt.step()
    model.eval()
    return model


def _save_data(path, rows, meta):
    """Checkpoint collected (x, delta, R) tuples so critic/selection can be
    re-run without recollecting (the expensive part)."""
    rec = {
        "trial": np.asarray([r["trial"] for r in rows], dtype=np.int64),
        "in_train": np.asarray([r["in_train"] for r in rows], dtype=bool),
        "state": np.stack([r["state"] for r in rows]),
        "base_chunk": np.stack([r["base_chunk"].reshape(-1) for r in rows]),
        "delta": np.stack([r["delta"] for r in rows]),
        "sigma": np.asarray([r["sigma"] for r in rows], dtype=np.float32),
        "R": np.asarray([r["R"] for r in rows], dtype=bool),
    }
    np.savez(path, **rec)
    with open(path.replace(".npz", "_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[critic] saved data -> {path} ({len(rows)} tuples)")


def _load_data(path):
    """Restore tuples + meta from a --save-data-npz checkpoint."""
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


def main(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    if args.data_npz:
        rows, meta = _load_data(args.data_npz)
        train_trials = [int(t) for t in meta["train_trials"]]
        eval_trials = [int(t) for t in meta["eval_trials"]]
        n_ok = int(meta["base_ok"])
        n_trials = int(meta["n_trials"])
        print(f"[critic] resumed from {args.data_npz}: {len(rows)} tuples | "
              f"train {train_trials} | eval {eval_trials}")
    else:
        suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
        task = suite.get_task(args.task_id)
        prompt = task.language
        env = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        init_states = suite.get_task_init_states(args.task_id)
        max_steps = args.max_steps or _MAX_STEPS_BY_SUITE[args.task_suite_name]
        print(
            f"[critic] task {args.task_id}: {prompt!r} | C={args.c} | "
            f"max_steps {max_steps} | alpha {args.alpha} | sigmas {args.sigmas}"
        )

        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
        from toolkits.standalone_eval_scripts.openpi import create_trained_policy

        cfg = get_openpi_config(
            args.config_name, model_path=args.pretrained_path, batch_size=1
        )
        policy = create_trained_policy(
            cfg, args.pretrained_path, sample_kwargs={"num_steps": args.num_steps}
        )
        print("[critic] policy loaded")

        # Phase 0: base probes
        base_seed = args.seed
        n_trials = min(args.num_trials, len(init_states))
        trials_fail = []
        n_ok = 0
        for trial in tqdm.tqdm(range(n_trials), desc="base probes"):
            seed_r = base_seed * 10000 + trial
            ok = _roll_intervene(
                env, policy, init_states[trial], C=-1, base_chunk=None,
                delta7=None, alpha=0.0, seed=seed_r, prompt=prompt,
                max_steps=max_steps, num_steps_wait=args.num_steps_wait,
                action_chunk=args.action_chunk,
            )
            if ok:
                n_ok += 1
            else:
                trials_fail.append(trial)
        print(f"[critic] base SR {n_ok}/{n_trials}; failed trials {trials_fail}")
        if len(trials_fail) < args.n_eval + 1:
            print("[critic] too few failed trials for a meaningful split; aborting")
            env.close()
            return

        eval_trials = trials_fail[-args.n_eval:]
        train_trials = trials_fail[:-args.n_eval]
        print(f"[critic] train trials {train_trials} | eval trials {eval_trials}")

        # Phase 1: collection -- for every failed trial, K executed proposals
        K = args.k
        rows = []
        for trial in tqdm.tqdm(trials_fail, desc="collection"):
            seed_r = base_seed * 10000 + trial
            state_C, base_chunk = _roll_to_C(
                env, policy, init_states[trial], args.c, args.num_steps_wait,
                args.action_chunk, seed_r, prompt,
            )
            for _ in range(K):
                delta, sigma = _sample_delta(rng, args.sigmas)
                R = _roll_intervene(
                    env, policy, init_states[trial], args.c, base_chunk, delta,
                    args.alpha, seed_r, prompt, max_steps, args.num_steps_wait,
                    args.action_chunk,
                )
                rows.append({
                    "trial": trial, "in_train": trial in train_trials,
                    "state": state_C, "base_chunk": base_chunk, "delta": delta,
                    "sigma": sigma, "R": bool(R),
                })
        env.close()

        meta = {
            "task": prompt,
            "task_suite": args.task_suite_name,
            "task_id": args.task_id,
            "C": args.c,
            "alpha": args.alpha,
            "sigmas": [float(s) for s in args.sigmas],
            "K": K,
            "max_steps": max_steps,
            "seed": args.seed,
            "base_ok": n_ok,
            "n_trials": n_trials,
            "trials_fail": trials_fail,
            "train_trials": train_trials,
            "eval_trials": eval_trials,
        }
        if args.save_data_npz:
            _save_data(args.save_data_npz, rows, meta)

    # Phase 2: critic on the train split, evaluated on the eval split
    tr = [r for r in rows if r["in_train"]]
    ev = [r for r in rows if not r["in_train"]]
    Xtr = np.stack([_feat_cols(args.feat, r) for r in tr])
    ytr = np.asarray([r["R"] for r in tr], dtype=np.float32)
    Xev = np.stack([_feat_cols(args.feat, r) for r in ev])
    yev = np.asarray([r["R"] for r in ev], dtype=np.float32)

    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-6
    Xtr_s = (Xtr - mu) / sd
    Xev_s = (Xev - mu) / sd

    model = _train_critic(Xtr_s, ytr, args.epochs, args.lr, args.wd,
                          args.seed, dropout=args.dropout)
    with torch.no_grad():
        qtr = torch.sigmoid(
            model(torch.tensor(Xtr_s, dtype=torch.float32))).numpy()
        qev = torch.sigmoid(
            model(torch.tensor(Xev_s, dtype=torch.float32))).numpy()

    auroc_tr = _auroc(ytr, qtr)
    auroc_ev = _auroc(yev, qev)
    print(f"[critic] AUROC train {auroc_tr:.3f} | eval {auroc_ev:.3f}")

    # Phase 3: Q-select@K vs Random@1 vs Best-of-K on eval trials
    sel = []
    for trial in eval_trials:
        rt = [r for r in ev if r["trial"] == trial]
        qs = qev[[i for i in range(len(ev)) if ev[i]["trial"] == trial]]
        idx = int(np.argmax(qs))
        q_best = bool(rt[idx]["R"])
        r_rand = rng.integers(0, len(rt))
        rand = bool(rt[r_rand]["R"])
        best = any(r["R"] for r in rt)
        order = np.argsort(qs, kind="mergesort")[::-1]
        half = max(1, len(order) // 2)
        top_ok = sum(1 for i in order[:half] if rt[int(i)]["R"])
        bot_ok = sum(1 for i in order[-half:] if rt[int(i)]["R"])
        sel.append({
            "trial": trial, "Q_select": q_best, "random": rand,
            "best_of_K": best, "topQ_recovery": top_ok / half,
            "botQ_recovery": bot_ok / half,
            "picked_delta_idx": idx, "Q_top_score": float(qs[idx]),
        })
        print(
            f"[critic] trial {trial}: Q-select {q_best} | random {rand} | "
            f"best-of-K {best} | topQ {top_ok}/{half} botQ {bot_ok}/{half}"
        )

    n_ev = len(sel)
    sr_qsel = sum(s["Q_select"] for s in sel) / n_ev
    sr_rand = sum(s["random"] for s in sel) / n_ev
    sr_best = sum(s["best_of_K"] for s in sel) / n_ev
    sr_rand_exp = float(yev.mean())  # expected Random@1 over all eval candidates
    top_pool = sum(s["topQ_recovery"] for s in sel) / n_ev
    bot_pool = sum(s["botQ_recovery"] for s in sel) / n_ev
    print(
        f"[critic] SR_Q-select@K {sr_qsel:.3f} | SR_Random {sr_rand_exp:.3f} "
        f"(draw {sr_rand:.3f}) | SR_Best-of-K {sr_best:.3f} | "
        f"lift {sr_qsel - sr_rand_exp:+.3f}"
    )
    print(f"[critic] topQ {top_pool:.3f} vs botQ {bot_pool:.3f}")

    report = {
        "task": meta["task"],
        "task_suite": meta["task_suite"],
        "task_id": meta["task_id"],
        "C": meta["C"],
        "alpha": meta["alpha"],
        "sigmas": meta["sigmas"],
        "K": meta["K"],
        "max_steps": meta["max_steps"],
        "seed": meta["seed"],
        "base_sr": n_ok / n_trials,
        "n_base_fail": len(meta["trials_fail"]),
        "train_trials": meta["train_trials"],
        "eval_trials": meta["eval_trials"],
        "n_train_tuples": len(tr),
        "n_eval_tuples": len(ev),
        "auroc_train": float(auroc_tr),
        "auroc_eval": float(auroc_ev),
        "sr_qselect_K": sr_qsel,
        "sr_random": sr_rand_exp,
        "sr_random_draw": sr_rand,
        "sr_best_of_K": sr_best,
        "selection_lift": sr_qsel - sr_rand_exp,
        "topQ_recovery": top_pool,
        "botQ_recovery": bot_pool,
        "recovery_by_sigma": {
            str(s): (
                int(sum(1 for r in rows if r["sigma"] == s and r["R"])),
                int(sum(1 for r in rows if r["sigma"] == s)),
            )
            for s in meta["sigmas"]
        },
        "selection_rows": sel,
        "rows": [
            {"trial": r["trial"], "sigma": r["sigma"], "R": r["R"],
             "in_train": r["in_train"]}
            for r in rows
        ],
    }
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[critic] wrote {args.out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pretrained-path",
                    default="/workspace/models/RLinf-Pi05-LIBERO-SFT")
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--task-suite-name", default="libero_10")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--num-trials", type=int, default=16)
    ap.add_argument("--action-chunk", type=int, default=5)
    ap.add_argument("--num-steps", type=int, default=5)
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=300,
                    help="collection horizon (truncated for speed)")
    ap.add_argument("--c", type=int, default=26,
                    help="fixed early intervention step (~0.05*T_base)")
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="residual scale: A_tilde = base + alpha*tanh(delta)")
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.1, 0.15, 0.3],
                    help="mixed proposal scales (coverage over perturbation "
                         "sizes; ablation showed sigma=0.3 harms recovery)")
    ap.add_argument("--k", type=int, default=8,
                    help="proposals per (trial, selection step)")
    ap.add_argument("--n-eval", type=int, default=3,
                    help="held-out trials for the Q-select vs Random test")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="dropout on the critic MLP (small-data regularization)")
    ap.add_argument("--feat", default="full",
                    choices=["full", "state_delta", "delta"],
                    help="critic input features: full=state+base_chunk+delta "
                         "(50-d, the default); state_delta=15-d; delta=7-d")
    ap.add_argument("--save-data-npz", default=None,
                    help="checkpoint collected (x,delta,R) tuples here")
    ap.add_argument("--data-npz", default=None,
                    help="resume from a --save-data-npz checkpoint (skips "
                         "collection; critic + selection only)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-json", required=True)
    main(ap.parse_args())
