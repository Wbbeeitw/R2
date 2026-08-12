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

"""SIR-VLA Stage-0 smoke: can a trainable refiner learn a useful EARLY single
intervention from terminal R in {0,1} alone?

Setup (the exact recipe that the oracle experiments validated)
--------------------------------------------------------------
  * base VLA frozen (pi0.5, libero_10 task 0)
  * ONE intervention at a fixed early step C (~0.05 * T_base), i.e. inside the
    basin-of-attraction window measured earlier (0.04-0.08 T; oracle recovery
    at fixed 0.05T is 41.2%, best-of-C 67.6-79.4%)
  * the refiner is a tiny MLP: ctx = [state_C (8), base_chunk (7*chunk_k)] ->
    residual delta_A (7*chunk_k); executed as A_tilde = base_chunk +
    alpha * tanh(delta_A) so the deviation is bounded.
  * terminal reward R in {0,1}; REINFORCE with a mean-R baseline; exploration
    is Gaussian noise on the residual.

Decision rule for the smoke
---------------------------
The oracle bound at this exact recipe is 41.2% (deterministic oracle chunk at
0.05T).  A refiner starts at ~0% (zero residual == do nothing == still fails).
If after a few REINFORCE iterations the *deterministic-eval* recovery is
clearly > 0 and trending up, single-intervention repair is LEARNABLE from
terminal R -- SIR-VLA Stage 0 passes.  If it stays at 0 despite noise, the
context (state+base chunk, no vision) is insufficient or terminal-R REINFORCE
cannot find the basin -- a diagnostic, not a dead end.

Notes
-----
  * smoke only: max_steps truncated to 300 for speed (same R definition,
    just shorter horizon), 6 failed trials, 10 iterations.
  * context has NO image -- deliberately, to test the cheapest viable
    refiner.  A later Stage adds vision features if this one shows signal.
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
    )


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


class Refiner(nn.Module):
    """ctx -> bounded residual delta_A.  Tiny by design."""

    def __init__(self, in_dim, chunk_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, chunk_dim),
        )

    def forward(self, x):
        return self.net(x)


def _roll_refiner(env, policy, refiner, init_state, C, max_steps,
                  num_steps_wait, action_chunk, alpha, sigma, seed, prompt,
                  eval_det):
    """One rollout with a single refiner intervention at step C.

    Returns (R, log_prob) where log_prob = -||eps||^2/2 for the exploration
    noise used (Gaussian), or None when eval_det (no noise).
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    policy.reset()
    env.reset()
    obs = env.set_init_state(init_state)

    action_plan = []
    intervened = False
    steps = 0
    lp = None
    for t in range(max_steps + num_steps_wait):
        if t < num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            continue
        steps += 1

        if not intervened and (steps - 1) == C:
            # base plans at C; refiner computes a bounded residual on it
            base_chunk = np.asarray(
                policy.infer(_obs(obs, prompt))["actions"][:action_chunk],
                dtype=np.float32,
            ).reshape(-1)
            state_C = _state_from_obs(obs)
            ctx = torch.tensor(
                np.concatenate([state_C, base_chunk]), dtype=torch.float32
            )
            delta = refiner(ctx).detach().numpy().reshape(-1)
            if not eval_det:
                eps = sigma * np.random.randn(*delta.shape)
                delta = delta + eps
                lp = -0.5 * float(np.sum(eps * eps))
            chunk = base_chunk + alpha * np.tanh(delta)
            chunk = chunk.reshape(action_chunk, 7)
            for a in chunk:
                obs, _, done, _ = env.step(a.tolist())
                if done:
                    return True, lp
            intervened = True
            action_plan = []  # base replans from the post-intervention obs
            continue

        if not action_plan:
            action_plan = list(policy.infer(_obs(obs, prompt))["actions"][:action_chunk])
        action = action_plan.pop(0)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            return True, lp
    return False, lp


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    prompt = task.language
    env = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    init_states = suite.get_task_init_states(args.task_id)
    max_steps = args.max_steps or _MAX_STEPS_BY_SUITE[args.task_suite_name]
    print(
        f"[smoke] task {args.task_id}: {prompt!r} | max_steps {max_steps} "
        f"(truncated for smoke speed)"
    )

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from toolkits.standalone_eval_scripts.openpi import create_trained_policy

    cfg = get_openpi_config(
        args.config_name, model_path=args.pretrained_path, batch_size=1
    )
    policy = create_trained_policy(
        cfg, args.pretrained_path, sample_kwargs={"num_steps": args.num_steps}
    )
    print("[smoke] policy loaded")

    # base failure probes: reuse the known seed-7 protocol
    base_seed = args.seed
    n_trials = min(args.num_trials, len(init_states))
    base_ok = {}
    for trial in tqdm.tqdm(range(n_trials), desc="base probes"):
        seed_r = base_seed * 10000 + trial
        ok, _ = _roll_refiner(
            env, policy, None, init_states[trial], C=-1, max_steps=max_steps,
            num_steps_wait=args.num_steps_wait, action_chunk=args.action_chunk,
            alpha=0.0, sigma=0.0, seed=seed_r, prompt=prompt, eval_det=True,
        )
        base_ok[trial] = ok
    n_ok = sum(base_ok.values())
    print(f"[smoke] base SR on probes: {n_ok}/{n_trials}")
    trials_fail = [t for t, ok in base_ok.items() if not ok][:args.max_fail]
    print(f"[smoke] using failed trials: {trials_fail}")

    refiner = Refiner(
        in_dim=8 + args.action_chunk * 7, chunk_dim=args.action_chunk * 7
    )
    opt = torch.optim.Adam(refiner.parameters(), lr=args.lr)
    C = args.c  # fixed early intervention step

    log = []
    for it in range(args.iters):
        Rs, lps = [], []
        for trial in trials_fail:
            seed_r = base_seed * 10000 + trial
            R, lp = _roll_refiner(
                env, policy, refiner, init_states[trial], C=C,
                max_steps=max_steps, num_steps_wait=args.num_steps_wait,
                action_chunk=args.action_chunk, alpha=args.alpha,
                sigma=args.sigma, seed=seed_r, prompt=prompt, eval_det=False,
            )
            Rs.append(float(R))
            lps.append(lp)
        baseline = float(np.mean(Rs)) if Rs else 0.0

        # REINFORCE; only successful trials pull (baseline-subtracted) so the
        # noise on failed trials is not pushed to explode.
        terms = []
        for R, lp in zip(Rs, lps):
            if lp is None or R <= 0:
                continue
            terms.append(-(R - baseline) * lp)
        if terms:
            loss = torch.tensor(np.mean(terms), dtype=torch.float32,
                                requires_grad=True)
            reg = args.l2 * torch.sum(
                torch.stack([p.norm() for p in refiner.parameters()])
            )
            loss = loss + reg
            opt.zero_grad()
            loss.backward()
            opt.step()

        # deterministic eval
        eval_r = []
        for trial in trials_fail:
            seed_r = base_seed * 10000 + trial
            R, _ = _roll_refiner(
                env, policy, refiner, init_states[trial], C=C,
                max_steps=max_steps, num_steps_wait=args.num_steps_wait,
                action_chunk=args.action_chunk, alpha=args.alpha,
                sigma=0.0, seed=seed_r, prompt=prompt, eval_det=True,
            )
            eval_r.append(float(R))
        rec = float(np.mean(eval_r)) if eval_r else 0.0
        n_rec = int(sum(eval_r))
        print(
            f"[smoke] it {it}: train R {baseline:.3f} ({int(sum(Rs))}/{len(Rs)}) "
            f"| eval recovery {rec:.3f} ({n_rec}/{len(eval_r)})"
        )
        log.append({
            "iter": it, "train_R": baseline, "n_train_success": int(sum(Rs)),
            "eval_recovery": rec, "n_eval_recovered": n_rec,
        })

    env.close()
    report = {
        "task": prompt,
        "task_suite": args.task_suite_name,
        "task_id": args.task_id,
        "max_steps": max_steps,
        "C": C,
        "alpha": args.alpha,
        "sigma": args.sigma,
        "lr": args.lr,
        "iters": args.iters,
        "base_sr": n_ok / n_trials,
        "trials_fail": trials_fail,
        "log": log,
        "final_eval_recovery": log[-1]["eval_recovery"] if log else None,
    }
    print(json.dumps({k: v for k, v in report.items()}, indent=2))
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[smoke] wrote {args.out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pretrained-path",
                    default="/workspace/models/RLinf-Pi05-LIBERO-SFT")
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--task-suite-name", default="libero_10")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--num-trials", type=int, default=12,
                    help="trials probed for base SR; failed ones (capped by "
                         "--max-fail) become the training set")
    ap.add_argument("--max-fail", type=int, default=6)
    ap.add_argument("--action-chunk", type=int, default=5)
    ap.add_argument("--num-steps", type=int, default=5)
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=300,
                    help="smoke horizon (truncated from 520 for speed)")
    ap.add_argument("--c", type=int, default=26,
                    help="fixed early intervention step (~0.05*T_base)")
    ap.add_argument("--alpha", type=float, default=0.3,
                    help="residual scale: A_tilde = base + alpha*tanh(delta)")
    ap.add_argument("--sigma", type=float, default=0.15,
                    help="Gaussian exploration std on delta")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-json", required=True)
    main(ap.parse_args())
