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

"""SIR-VLA Oracle-Intervention bound experiment (zero trainable intervention).

Question
--------
Does a SINGLE intervention have any chance of rescuing a failed LIBERO rollout?
If even an *oracle* intervention -- the exact action a successful trajectory
would take at (near) this state -- cannot turn a failure into a success, then
training a refiner that produces interventions is pointless: there is no basin
of attraction to recover into, and SIR-VLA dies. If it can, SIR-VLA's premise
(single-deviation experimental design) is alive and the refiner has a ceiling
to approach.

Protocol
--------
For each LIBERO init state, run a *seeded, reproducible* base rollout (no
intervention).  If it fails, resample the SAME init state + SAME seed, and at a
uniformly random step C inject ONE oracle action chunk (retrieved live from the
env state at C), then hand control back to the base policy:

    base, base, ..., [oracle chunk @C], base, base, ... , R in {0,1}

Because the seed is fixed, the trajectory is identical to the baseline up to C,
so the intervention is the *only* difference -- the cleanest possible credit.

Metrics
-------
  SR_base               baseline success rate (first online SR of the base VLA)
  SR_oracle             success rate after single oracle intervention
  recovery_rate         P(success | baseline fail)          <- SIR lifeline
  harm_rate             P(fail | baseline success) (with --probe-success)
  recovery_rate(c-bin)  recovery vs intervention progress   <- first core figure

Oracle source
-------------
The successful-rollout dataset (``recap_libero10_task0_train``): a chunk of
``chunk_k`` actions retrieved by nearest state lookup.  This is the strongest
intervention a refiner could plausibly learn, so it upper-bounds SIR.

Usage
-----
  cd /workspace/RLinf && \\
  EMBODIED_PATH=/workspace/RLinf/examples/sft LIBERO_REPO_PATH=/opt/venv/openpi/libero \\
  PYTHONPATH=/workspace/RLinf:/opt/venv/openpi/libero \\
  /opt/venv/openpi/bin/python darc_vla/sir_oracle_eval.py \\
      --pretrained-path /workspace/models/RLinf-Pi05-LIBERO-SFT \\
      --config-name pi05_libero --data-dir /workspace/datasets/recap_libero10_task0/libero10_task0_train \\
      --num-trials 50 --action-chunk 5 --chunk-k 5 --seed 7 --out-json /workspace/workspcae/sir_oracle.json
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


class SuccessActionPool:
    """Nearest-state action-chunk retrieval over the successful rollout dataset.

    The dataset stores state[8] = [pos3, axisangle3, gripper] and actions[7] =
    [arm6, gripper] at the same rate as the LIBERO env, so an env state can be
    looked up directly.
    """

    def __init__(self, data_dir: str, dims: list[int], eps: int | None = None):
        self.dims = np.asarray(dims, dtype=int)
        states, acts, ep_of, step_of = [], [], [], []
        self._ep_actions: dict[int, np.ndarray] = {}

        count = 0
        for chunk_dir in sorted(
            d for d in os.listdir(os.path.join(data_dir, "data"))
            if d.startswith("chunk-")
        ):
            p = os.path.join(data_dir, "data", chunk_dir)
            for fn in sorted(os.listdir(p)):
                if not fn.endswith(".parquet"):
                    continue
                ep = int(fn.split("_")[1].split(".")[0])
                df = pd.read_parquet(os.path.join(p, fn))
                if "is_success" in df.columns:
                    df = df[df["is_success"].fillna(False).astype(bool)]
                if len(df) == 0:
                    continue
                st = np.asarray(df["state"].tolist(), dtype=np.float32)
                ac = np.asarray(df["actions"].tolist(), dtype=np.float32)
                self._ep_actions[ep] = ac
                states.append(st)
                acts.append(ac)
                ep_of.append(np.full(len(st), ep))
                step_of.append(np.arange(len(st)))
                count += 1
                if eps is not None and count >= eps:
                    break
            if eps is not None and count >= eps:
                break

        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(acts, axis=0)
        self.ep_of = np.concatenate(ep_of, axis=0)
        self.step_of = np.concatenate(step_of, axis=0)
        print(
            f"[oracle] success pool: {len(self.states)} states "
            f"from {len(self._ep_actions)} episodes"
        )

    def nearest_chunk(self, state: np.ndarray, k: int) -> np.ndarray:
        """[k, 7] action chunk from the successful episode nearest to `state`."""
        d = np.sum(
            (self.states[:, self.dims] - state[self.dims][None, :]) ** 2, axis=1
        )
        idx = int(np.argmin(d))
        ep, step = self.ep_of[idx], self.step_of[idx]
        acts = self._ep_actions[ep]
        chunk = acts[step : step + k]
        if len(chunk) < k:
            chunk = np.pad(chunk, ((0, k - len(chunk)), (0, 0)), mode="edge")
        return chunk


def _roll(env, policy, init_state, max_steps, num_steps_wait, action_chunk,
          pool, chunk_k, oracle_C, seed, prompt):
    """One rollout; if oracle_C is not None, inject one oracle chunk at that
    step (live state lookup). Returns (success, steps)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    policy.reset()
    env.reset()
    obs = env.set_init_state(init_state)

    action_plan = []
    oracle_done = oracle_C is None
    steps = 0
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
                    return True, steps
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
        obs, _, done, _ = env.step(action.tolist())
        if done:
            return True, steps
    return False, steps


def main(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    prompt = task.language
    env = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    init_states = suite.get_task_init_states(args.task_id)
    max_steps = _MAX_STEPS_BY_SUITE[args.task_suite_name]
    print(
        f"[sir] task {args.task_id}: {prompt!r} | {len(init_states)} init states "
        f"| max_steps {max_steps}"
    )

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from toolkits.standalone_eval_scripts.openpi import create_trained_policy

    cfg = get_openpi_config(
        args.config_name, model_path=args.pretrained_path, batch_size=1
    )
    policy = create_trained_policy(
        cfg, args.pretrained_path, sample_kwargs={"num_steps": args.num_steps}
    )
    print("[sir] policy loaded")

    pool = SuccessActionPool(args.data_dir, _ORACLE_DIMS[args.oracle_dist])

    n_trials = min(args.num_trials, len(init_states))
    rows = []
    for trial in tqdm.tqdm(range(n_trials), desc="trials"):
        seed_r = args.seed * 10000 + trial

        ok_base, t_base = _roll(
            env, policy, init_states[trial], max_steps, args.num_steps_wait,
            args.action_chunk, pool, args.chunk_k, oracle_C=None,
            seed=seed_r, prompt=prompt,
        )
        row = {"trial": trial, "T": t_base, "base": bool(ok_base),
               "C": None, "oracle": None, "recovery": None, "harm": None}

        if ok_base and not args.probe_success:
            rows.append(row)
            continue

        # single oracle intervention at a uniform step C
        C = int(rng.integers(0, max(t_base, 1)))
        ok_or, _ = _roll(
            env, policy, init_states[trial], max_steps, args.num_steps_wait,
            args.action_chunk, pool, args.chunk_k, oracle_C=C,
            seed=seed_r, prompt=prompt,
        )
        row["C"], row["oracle"] = C, bool(ok_or)
        if ok_base:
            row["harm"] = not ok_or
        else:
            row["recovery"] = bool(ok_or)
        rows.append(row)

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

    bin_edges = [0.25, 0.5, 0.75, 1.01]
    bin_labels = ["q1", "q2", "q3", "q4"]
    rec_by_bin = {b: [0, 0] for b in bin_labels}
    for r in rows:
        if r["recovery"] is None:
            continue
        p = (r["C"] + 1) / max(r["T"], 1)
        for label, edge in zip(bin_labels, bin_edges):
            if p <= edge:
                rec_by_bin[label][1] += 1
                rec_by_bin[label][0] += 1 if r["recovery"] else 0
                break

    report = {
        "task": prompt,
        "task_suite": args.task_suite_name,
        "task_id": args.task_id,
        "n_trials": n,
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
        "recovery_by_progress_bin": {
            b: (rec_by_bin[b][0], rec_by_bin[b][1]) for b in bin_labels
        },
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[sir] wrote {args.out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pretrained-path",
                    default="/workspace/models/RLinf-Pi05-LIBERO-SFT")
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--data-dir", required=True,
                    help="LeRobot dataset dir for the success-action pool")
    ap.add_argument("--task-suite-name", default="libero_10")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--num-trials", type=int, default=50)
    ap.add_argument("--action-chunk", type=int, default=5)
    ap.add_argument("--chunk-k", type=int, default=5,
                    help="oracle intervention chunk length")
    ap.add_argument("--oracle-dist", default="pos_grip",
                    choices=list(_ORACLE_DIMS))
    ap.add_argument("--num-steps", type=int, default=5)
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--probe-success", action="store_true",
                    help="also run oracle on baseline-success trials (harm rate)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-json", required=True)
    main(ap.parse_args())
