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

"""SIR intervention-mechanism ablation: spatial correction vs replan trigger.

Motivation (GPT review, 2026-08-12)
-----------------------------------
The Stage-0 smoke found that a *random* residual intervention at the early
window recovers ~33% of failures -- far above the zero-intervention base.  Two
radically different mechanisms explain this:

  A. spatial correction: the perturbed action physically steers the robot back
     into pi0's competence basin (the residual itself matters).
  B. replan trigger: ANY change at C breaks the doomed deterministic
     trajectory; pi0 replans from a fresh observation and sometimes lands in a
     success branch (the specific deviation is irrelevant).

If B dominates, SIR might not need residual learning at all (or needs a
different one).  This script isolates the two with cheap ablations, all at the
fixed, validated intervention point C=26 (~0.05 T_base):

  zero      : execute pi0's own chunk unchanged  -> pipeline no-op control
  random    : execute base_chunk + alpha*tanh(delta), delta ~ N(0, sigma^2 I)
              (two scales to see the size effect)
  resample  : re-infer a NEW pi0 chunk from the SAME obs (pure replan, no
              spatial correction outside pi0's own distribution)

Decision rule
-------------
  random >> resample  => spatial perturbation genuinely helps (mechanism A)
  resample ~ random   => SIR's early recovery is mostly a replan trigger (B)
  resample > 0        => even without residuals, breaking determinism helps
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib

import numpy as np
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


def _roll_ablate(env, policy, init_state, C, mode, alpha, sigma, seed, prompt,
                 action_chunk, num_steps_wait, max_steps):
    """One rollout with a single intervention of the given mode at step C."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    policy.reset()
    env.reset()
    obs = env.set_init_state(init_state)

    action_plan = []
    intervened = False
    steps = 0
    for t in range(max_steps + num_steps_wait):
        if t < num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            continue
        steps += 1

        if not intervened and (steps - 1) == C:
            base_chunk = np.asarray(
                policy.infer(_obs(obs, prompt))["actions"][:action_chunk],
                dtype=np.float32,
            )
            if mode == "zero":
                chunk = base_chunk
            elif "random" in mode:
                delta = sigma * np.random.randn(action_chunk, 7).astype(np.float32)
                chunk = base_chunk + alpha * np.tanh(delta)
            elif mode == "resample":
                chunk = np.asarray(
                    policy.infer(_obs(obs, prompt))["actions"][:action_chunk],
                    dtype=np.float32,
                )
            else:
                raise ValueError(mode)
            for a in chunk:
                obs, _, done, _ = env.step(a.tolist())
                if done:
                    return True
            intervened = True
            action_plan = []
            continue

        if not action_plan:
            action_plan = list(policy.infer(_obs(obs, prompt))["actions"][:action_chunk])
        action = action_plan.pop(0)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            return True
    return False


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = suite.get_task(args.task_id)
    prompt = task.language
    env = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    init_states = suite.get_task_init_states(args.task_id)
    max_steps = _MAX_STEPS_BY_SUITE[args.task_suite_name]
    print(f"[ablate] task {args.task_id}: {prompt!r} | C={args.c} | max_steps {max_steps}")

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from toolkits.standalone_eval_scripts.openpi import create_trained_policy

    cfg = get_openpi_config(
        args.config_name, model_path=args.pretrained_path, batch_size=1
    )
    policy = create_trained_policy(
        cfg, args.pretrained_path, sample_kwargs={"num_steps": args.num_steps}
    )
    print("[ablate] policy loaded")

    base_seed = args.seed
    n_trials = min(args.num_trials, len(init_states))

    # base probes
    trials_fail = []
    n_ok = 0
    for trial in tqdm.tqdm(range(n_trials), desc="base probes"):
        seed_r = base_seed * 10000 + trial
        ok = _roll_ablate(
            env, policy, init_states[trial], C=-1, mode="zero", alpha=0.0,
            sigma=0.0, seed=seed_r, prompt=prompt,
            action_chunk=args.action_chunk, num_steps_wait=args.num_steps_wait,
            max_steps=max_steps,
        )
        if ok:
            n_ok += 1
        else:
            trials_fail.append(trial)
    print(f"[ablate] base SR {n_ok}/{n_trials}; failed trials {trials_fail}")

    modes = []
    modes.append({"name": "zero", "alpha": 0.0, "sigma": 0.0})
    for s in args.sigmas:
        modes.append({"name": f"random_s{str(s).replace('.', '_')}",
                      "alpha": args.alpha, "sigma": s})
    modes.append({"name": "resample", "alpha": 0.0, "sigma": 0.0})

    per_mode = {m["name"]: {"n_recovered": 0, "n_trials": 0} for m in modes}
    rows = []
    for trial in tqdm.tqdm(trials_fail, desc="interventions"):
        row = {"trial": trial}
        seed_r = base_seed * 10000 + trial
        for m in modes:
            ok = _roll_ablate(
                env, policy, init_states[trial], C=args.c, mode=m["name"],
                alpha=m["alpha"], sigma=m["sigma"], seed=seed_r, prompt=prompt,
                action_chunk=args.action_chunk,
                num_steps_wait=args.num_steps_wait, max_steps=max_steps,
            )
            row[m["name"]] = bool(ok)
            per_mode[m["name"]]["n_trials"] += 1
            per_mode[m["name"]]["n_recovered"] += int(ok)
        rows.append(row)

    env.close()

    recovery = {
        name: (d["n_recovered"], d["n_trials"])
        for name, d in per_mode.items()
    }
    report = {
        "task": prompt,
        "task_suite": args.task_suite_name,
        "task_id": args.task_id,
        "C": args.c,
        "alpha": args.alpha,
        "sigmas": args.sigmas,
        "base_sr": n_ok / n_trials,
        "n_base_fail": len(trials_fail),
        "recovery": {k: (v[0], v[1]) for k, v in recovery.items()},
        "recovery_rate": {k: (v[0] / v[1] if v[1] else None)
                          for k, v in recovery.items()},
        "rows": rows,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[ablate] wrote {args.out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pretrained-path",
                    default="/workspace/models/RLinf-Pi05-LIBERO-SFT")
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--task-suite-name", default="libero_10")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--num-trials", type=int, default=12)
    ap.add_argument("--action-chunk", type=int, default=5)
    ap.add_argument("--num-steps", type=int, default=5)
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--c", type=int, default=26,
                    help="fixed early intervention step (~0.05*T_base)")
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.15, 0.3],
                    help="random-residual scales to compare")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-json", required=True)
    main(ap.parse_args())
