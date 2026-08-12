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

"""SIR-VLA online-pool oracle bound -- dataset-free cross-task basin check.

Why
---
The dataset-based oracle (sir_oracle_eval.py) needs a successful-rollout
dataset for *each* task, which only exists for libero_10 task 0.  This variant
builds the success pool by sampling the frozen base policy itself across
seeds, so it works on ANY LIBERO task with no external data.  It is also the
closer analogue of the refiner's real target: the refiner will have to produce
repairs inside the base policy's own distribution, not copy human demos.

Protocol
--------
Phase 1 (pool build + base SR): for each init state and each of ``--num-seeds``
seeds, roll out the base policy and add every SUCCESSFUL trajectory to an
online pool (state[8] -> action[7] per step).  The seed-0 rollout of each trial
also gives the base success/failure.

Phase 2 (intervention): for each baseline-failed trajectory, inject ONE oracle
chunk at position C (live state lookup into the online pool), then hand control
back to base.  Because the seed is fixed, the pre-C trajectory is identical to
the baseline -- the intervention is the only difference.

Metrics mirror sir_oracle_eval.py (sr_base / recovery / recovery_by_c_frac /
best-of-C).  The interpretation differs: a high recovery here means the
policy's own successful rollouts contain a basin of attraction for its own
failures (a "near-miss is fixable" structure); a low one means the failures are
policy-incompetence, not early divergence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib

import numpy as np
import pandas as pd
import torch
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
_ORACLE_DIMS = {
    "pos_grip": [0, 1, 2, 6],
    "full": [0, 1, 2, 3, 4, 5, 6],
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


class OnlineSuccessPool:
    """Nearest-state action-chunk retrieval over policy-sampled successes."""

    def __init__(self, dims):
        self.dims = np.asarray(dims, dtype=int)
        self._ep_states = []
        self._ep_actions = []

    def add_episode(self, states: np.ndarray, actions: np.ndarray):
        if len(states) == 0:
            return
        self._ep_states.append(states)
        self._ep_actions.append(actions)

    def finalize(self):
        ep_of, step_of = [], []
        for i, s in enumerate(self._ep_states):
            n = len(s)
            ep_of.append(np.full(n, i))
            step_of.append(np.arange(n))
        self.states = np.concatenate(self._ep_states, axis=0)
        self.ep_of = np.concatenate(ep_of)
        self.step_of = np.concatenate(step_of)
        self.n_eps = len(self._ep_states)
        print(
            f"[online] success pool: {len(self.states)} states "
            f"from {self.n_eps} episodes"
        )

    def nearest_chunk(self, state: np.ndarray, k: int) -> np.ndarray:
        d = np.sum(
            (self.states[:, self.dims] - state[self.dims][None, :]) ** 2, axis=1
        )
        idx = int(np.argmin(d))
        ep, step = int(self.ep_of[idx]), int(self.step_of[idx])
        acts = self._ep_actions[ep]
        chunk = acts[step : step + k]
        if len(chunk) < k:
            chunk = np.pad(chunk, ((0, k - len(chunk)), (0, 0)), mode="edge")
        return chunk


def _roll(env, policy, init_state, max_steps, num_steps_wait, action_chunk,
          pool, chunk_k, oracle_C, seed, prompt, collect=False):
    """One rollout; if oracle_C is not None, inject one oracle chunk at that
    step.  Returns (success, steps, states, actions) where states/actions are
    only non-trivial when collect=True (used to seed the online pool)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    policy.reset()
    env.reset()
    obs = env.set_init_state(init_state)

    action_plan = []
    oracle_done = oracle_C is None
    steps = 0
    states, actions = [], []
    for t in range(max_steps + num_steps_wait):
        if t < num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            continue
        steps += 1

        if oracle_C is not None and not oracle_done and (steps - 1) == oracle_C:
            chunk = pool.nearest_chunk(_state_from_obs(obs), chunk_k)
            for a in chunk:
                obs, _, done, _ = env.step(a.tolist())
                if done:
                    return True, steps, states, actions
            oracle_done = True
            action_plan = []  # base policy replans from the post-intervention obs
            continue

        if not action_plan:
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            observation = {
                "observation/image": img,
                "observation/wrist_image": wrist,
                "observation/state": _state_from_obs(obs),
                "prompt": prompt,
            }
            action_plan = list(policy.infer(observation)["actions"][:action_chunk])

        action = action_plan.pop(0)
        if collect:
            states.append(_state_from_obs(obs))
            actions.append(np.asarray(action, dtype=np.float32))
        obs, _, done, _ = env.step(action.tolist())
        if done:
            if collect:
                return True, steps, np.stack(states), np.stack(actions)
            return True, steps, None, None

    if collect:
        return False, steps, np.stack(states), np.stack(actions)
    return False, steps, None, None


def main(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if args.sweep_c:
        args.sweep_c = [float(x) for x in args.sweep_c.split(",")]

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    prompt = task.language
    env = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    init_states = suite.get_task_init_states(args.task_id)
    max_steps = _MAX_STEPS_BY_SUITE[args.task_suite_name]
    print(
        f"[online] task {args.task_id}: {prompt!r} | {len(init_states)} init "
        f"states | max_steps {max_steps}"
    )

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from toolkits.standalone_eval_scripts.openpi import create_trained_policy

    cfg = get_openpi_config(
        args.config_name, model_path=args.pretrained_path, batch_size=1
    )
    policy = create_trained_policy(
        cfg, args.pretrained_path, sample_kwargs={"num_steps": args.num_steps}
    )
    print("[online] policy loaded")

    pool = OnlineSuccessPool(_ORACLE_DIMS[args.oracle_dist])
    base_seed = args.seed

    # Phase 1: sample successes across seeds to build the pool + measure base SR
    n_trials = min(args.num_trials, len(init_states))
    rows = []
    for trial in tqdm.tqdm(range(n_trials), desc="phase1 build-pool"):
        for s in range(args.num_seeds):
            seed_r = base_seed * 10000 + trial * 100 + s
            ok, t_used, sts, acs = _roll(
                env, policy, init_states[trial], max_steps, args.num_steps_wait,
                args.action_chunk, pool, args.chunk_k, oracle_C=None,
                seed=seed_r, prompt=prompt, collect=True,
            )
            if ok:
                pool.add_episode(sts, acs)
            if s == 0:
                rows.append({
                    "trial": trial, "T": t_used, "base": bool(ok),
                    "C": None, "oracle": None, "recovery": None, "harm": None,
                })
    pool.finalize()
    if pool.n_eps == 0:
        print("[online] WARNING: no successful trajectories sampled; "
              "interventions will be vacuous (pool empty)")

    # Phase 2: single intervention on baseline-failed trajectories
    for row in rows:
        if row["base"] and not args.probe_success:
            continue
        trial = row["trial"]
        t_base = row["T"]
        seed_r = base_seed * 10000 + trial * 100  # same seed as phase-1 base
        if args.sweep_c:
            per_c = []
            for cf in args.sweep_c:
                C = max(int(cf * t_base), 0)
                ok_or, _, _, _ = _roll(
                    env, policy, init_states[trial], max_steps,
                    args.num_steps_wait, args.action_chunk, pool, args.chunk_k,
                    oracle_C=C, seed=seed_r, prompt=prompt,
                )
                per_c.append({"c_frac": cf, "C": C, "ok": bool(ok_or)})
            row["sweep"] = per_c
            row["oracle"] = any(p["ok"] for p in per_c)
            if row["base"]:
                row["harm"] = not row["oracle"]
            else:
                row["recovery"] = bool(row["oracle"])
        else:
            hi = (
                max(int(args.frac_early * t_base), 1)
                if args.intervention_mode == "early"
                else max(t_base, 1)
            )
            C = int(rng.integers(0, hi))
            ok_or, _, _, _ = _roll(
                env, policy, init_states[trial], max_steps,
                args.num_steps_wait, args.action_chunk, pool, args.chunk_k,
                oracle_C=C, seed=seed_r, prompt=prompt,
            )
            row["C"], row["oracle"] = C, bool(ok_or)
            if row["base"]:
                row["harm"] = not ok_or
            else:
                row["recovery"] = bool(ok_or)

    env.close()

    n = len(rows)
    n_base_ok = sum(r["base"] for r in rows)
    n_base_fail = n - n_base_ok
    n_recovered = sum(1 for r in rows if r["recovery"])
    n_harm = sum(1 for r in rows if r["harm"])
    sr_base = n_base_ok / n if n else 0.0
    sr_oracle = (n_base_ok + n_recovered) / n if n else 0.0
    recovery = n_recovered / n_base_fail if n_base_fail else None
    harm = n_harm / n_base_ok if (n_base_ok and args.probe_success) else None

    sweep_by_frac = {}
    n_swept = n_best = 0
    for r in rows:
        if "sweep" not in r:
            continue
        n_swept += 1
        if r["recovery"]:
            n_best += 1
        for p in r["sweep"]:
            e = sweep_by_frac.setdefault(p["c_frac"], [0, 0])
            e[1] += 1
            e[0] += 1 if p["ok"] else 0
    sweep_stats = {
        "n_swept": n_swept,
        "recovery_best": n_best / n_swept if n_swept else None,
        "recovery_by_c_frac": {
            k: tuple(v) for k, v in sorted(sweep_by_frac.items())
        },
    }

    report = {
        "task": prompt,
        "task_suite": args.task_suite_name,
        "task_id": args.task_id,
        "n_trials": n,
        "num_seeds": args.num_seeds,
        "pool_eps": pool.n_eps,
        "pool_states": int(len(pool.states)),
        "action_chunk": args.action_chunk,
        "chunk_k": args.chunk_k,
        "oracle_dist": args.oracle_dist,
        "sr_base": sr_base,
        "sr_oracle": sr_oracle,
        "il_oracle": sr_oracle - sr_base,
        "n_base_success": n_base_ok,
        "n_base_fail": n_base_fail,
        "n_recovered": n_recovered,
        "recovery_rate": recovery,
        "n_harm": n_harm,
        "harm_rate": harm,
        "sweep": sweep_stats,
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[online] wrote {args.out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pretrained-path",
                    default="/workspace/models/RLinf-Pi05-LIBERO-SFT")
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--task-suite-name", default="libero_10")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--num-trials", type=int, default=10)
    ap.add_argument("--num-seeds", type=int, default=5,
                    help="per-trial base-policy samples used to build the "
                         "online success pool (seed 0 of each trial is also "
                         "the base SR measurement)")
    ap.add_argument("--action-chunk", type=int, default=5)
    ap.add_argument("--chunk-k", type=int, default=5)
    ap.add_argument("--oracle-dist", default="pos_grip",
                    choices=list(_ORACLE_DIMS))
    ap.add_argument("--num-steps", type=int, default=5)
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--probe-success", action="store_true")
    ap.add_argument("--intervention-mode", default="uniform",
                    choices=["uniform", "early"])
    ap.add_argument("--frac-early", type=float, default=0.25)
    ap.add_argument("--sweep-c", default=None,
                    help="comma-separated candidate C fractions; each failed "
                         "trajectory gets ONE intervention at each candidate "
                         "(same seed) -> best-of-C")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-json", required=True)
    main(ap.parse_args())
