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

"""SIR basin-geometry: intervention-primitive map (GPT route B, case C).

The verified-patch transfer matrix (B0) showed PTR 0.143 < RR(sigma=0.1) 0.25
with no universal rescue direction and no block structure -> case C: exact
continuous delta carries no reusable semantics across trajectories.  One level
coarser, this map asks: do successful interventions have *coarse directional*
structure?  If yes, primitive-SIR (a fixed bank of axis primitives) is viable
without any state-conditioned selector.

Design
------
  * M[j, m] = R(s_j, primitive_m): roll target j deterministically to C=26,
    execute  A~_C(j) = base_chunk_j + alpha*tanh(delta_m),  hand back to base.
  * primitives = single-axis 7-D residuals  delta_m = +/-scale * e_i  for
    i in {pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, grip}  (14 directions)
    plus the zero residual (= pure replan trigger, i.e. the resample baseline:
    A~ = base_chunk unchanged, base replans from the new obs).
  * scale = --scale (default 0.3): matches the component magnitude of the
    verified min-norm patches (0.08-0.27), avoiding both the saturated
    delta=+/-1 regime (harmful per ablation) and a sub-noise nudge.
  * targets = ALL 13 failed trials (kept, incl. the 5 hard failures) so the
    matrix shows recoverable-vs-irrecoverable against every primitive.
  * protocol identical to sir_transfer_matrix.py (same seeds, C, alpha), so
    M is comparable to T and to the collection baselines.

Metrics
-------
  PRR_m   = primitive recovery rate over targets (compare vs RR(0.1)=0.25
            and vs the zero/replan-trigger column)
  C_j     = any primitive rescues target j  (BestOf-15 coverability)
  zero    = resample-only baseline (mechanism B, ~0.375 in the ablation)
  plus the per-primitive vs zero lift to isolate direction from replanning.

Decision (mechanical)
---------------------
  >=2 primitives with PRR >> zero   -> primitive-SIR viable (fixed bank)
  high BestOf-15 but no general one -> primitive selection needed
                                       (rejection / light retrieval)
  both low                          -> case D: state-specific search
                                       (large data / vision becomes justified)
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

_PRIM_DIMS = ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z", "grip"]


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


def _roll_to_C(env, policy, init_state, C, num_steps_wait, action_chunk,
               seed, prompt):
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


def _load_data(path):
    """Restore tuples + meta from a sir_critic_select.py --save-data-npz."""
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
            "sigma": float(round(float(d["sigma"][i]), 4)),
            "R": bool(d["R"][i]),
        })
    with open(path.replace(".npz", "_meta.json")) as f:
        meta = json.load(f)
    return rows, meta


def _primitives(scale):
    prims = []
    for i, name in enumerate(_PRIM_DIMS):
        for sgn in (+1, -1):
            d = np.zeros(7, dtype=np.float32)
            d[i] = sgn * scale
            prims.append((f"{'+' if sgn > 0 else '-'}{name}", d))
    prims.append(("zero", np.zeros(7, dtype=np.float32)))
    return prims


def main(args):
    rows, meta = _load_data(args.data_npz)
    targets = [int(t) for t in meta["trials_fail"]]
    C = int(meta["C"])
    alpha = float(meta["alpha"])
    max_steps = int(meta["max_steps"])
    base_seed = int(meta["seed"])
    prims = _primitives(args.scale)
    prim_names = [p[0] for p in prims]

    # collection RR at sigma=0.1 (matched-random proposal baseline)
    rr01 = (sum(1 for r in rows if r["sigma"] == 0.1 and r["R"]),
            sum(1 for r in rows if r["sigma"] == 0.1))
    rr01_rate = rr01[0] / rr01[1] if rr01[1] else 0.0

    print(f"[prim] {meta['task']!r} | C={C} alpha={alpha} "
          f"max_steps={max_steps} | scale={args.scale}")
    print(f"[prim] targets {targets} (n={len(targets)}) | "
          f"primitives {len(prims)} | RR(0.1)={rr01_rate:.3f}")
    for j in targets:
        n_pos = sum(1 for r in rows if r["trial"] == j and r["R"])
        print(f"[prim] target {j}: collection positives {n_pos}/"
              f"{sum(1 for r in rows if r['trial'] == j)}")

    suite = benchmark.get_benchmark_dict()[meta["task_suite"]]()
    task = suite.get_task(int(meta["task_id"]))
    prompt = task.language
    env = _get_libero_env(task, LIBERO_ENV_RESOLUTION, base_seed)
    init_states = suite.get_task_init_states(int(meta["task_id"]))

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from toolkits.standalone_eval_scripts.openpi import create_trained_policy

    cfg = get_openpi_config(
        args.config_name, model_path=args.pretrained_path, batch_size=1
    )
    policy = create_trained_policy(
        cfg, args.pretrained_path, sample_kwargs={"num_steps": args.num_steps}
    )
    print("[prim] policy loaded")

    base_chunk = {}
    for j in tqdm.tqdm(targets, desc="roll-to-C"):
        seed_r = base_seed * 10000 + j
        _, bc = _roll_to_C(env, policy, init_states[j], C, args.num_steps_wait,
                           args.action_chunk, seed_r, prompt)
        base_chunk[j] = bc

    M = {}
    for j in tqdm.tqdm(targets, desc="targets"):
        M[j] = {}
        seed_r = base_seed * 10000 + j
        for name, d in prims:
            ok = _roll_intervene(
                env, policy, init_states[j], C, base_chunk[j], d, alpha,
                seed_r, prompt, max_steps, args.num_steps_wait,
                args.action_chunk,
            )
            M[j][name] = bool(ok)
    env.close()

    # ---- metrics ----------------------------------------------------------
    prr = {m: float(np.mean([M[j][m] for j in targets])) for m in prim_names}
    cover = {j: float(np.max([M[j][m] for m in prim_names])) for j in targets}
    best_any = float(np.mean(list(cover.values())))
    zero_col = {j: bool(M[j]["zero"]) for j in targets}
    zero_rate = float(np.mean(list(zero_col.values())))

    print("\n[prim] matrix rows=target, cols=primitive ('#'=R=1)")
    print("        " + " ".join(f"{m:>9}" for m in prim_names))
    for j in targets:
        row = " ".join("#" if M[j][m] else "." for m in prim_names)
        print(f"  t{j:3d}   {row}   any={int(cover[j] > 0)}")
    print("  PRR    " + " ".join(f"{prr[m]:9.3f}" for m in prim_names))

    print(f"\n[prim] zero (replan-trigger) rate = {zero_rate:.3f} "
          f"({sum(zero_col.values())}/{len(targets)})")
    print(f"[prim] RR(0.1) collection = {rr01_rate:.3f} | "
          f"BestOf-15(any primitive) = {best_any:.3f}")
    general = [m for m in prim_names if prr[m] >= max(0.30, zero_rate + 0.10)]
    print(f"[prim] general primitives (PRR >= {max(0.30, zero_rate + 0.10):.2f}): "
          + (", ".join(f"{m}:{prr[m]:.3f}" for m in general) or "none"))
    # per-target primitive lift over zero (direction beyond replanning)
    n_need_direction = sum(1 for j in targets
                           if not zero_col[j] and cover[j] > 0)
    print(f"[prim] targets rescued by a NON-zero primitive but not by "
          f"replanning: {n_need_direction}/{len(targets)}")

    if len(general) >= 2:
        decision = ("A2: general direction primitives exist -> primitive-SIR "
                    "(fixed bank, no selector) is viable")
    elif best_any >= 0.5:
        decision = ("B2: no general primitive but every recoverable target has "
                    "one -> primitive SELECTION needed (rejection / light "
                    "retrieval)")
    else:
        decision = ("D: primitives also fail -> state-specific intervention "
                    "search; large data / vision becomes justified")
    print(f"[prim] decision signal: {decision}")

    report = {
        "task": meta["task"],
        "task_suite": meta["task_suite"],
        "task_id": meta["task_id"],
        "C": C, "alpha": alpha, "max_steps": max_steps,
        "scale": args.scale,
        "targets": targets,
        "primitive_names": prim_names,
        "matrix": {str(j): {m: bool(M[j][m]) for m in prim_names}
                   for j in targets},
        "prr": {m: float(prr[m]) for m in prim_names},
        "target_coverability": {str(j): float(cover[j]) for j in targets},
        "best_any": float(best_any),
        "zero_rate": float(zero_rate),
        "zero_column": {str(j): bool(zero_col[j]) for j in targets},
        "n_need_direction": n_need_direction,
        "general_primitives": general,
        "rr_sigma0_1": [rr01[0], rr01[1]],
        "decision_signal": decision,
    }
    print(json.dumps(report, indent=2))
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[prim] wrote {args.out_json}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-npz", required=True,
                    help="sir_critic_select.py checkpoint (tuples + meta)")
    ap.add_argument("--pretrained-path",
                    default="/workspace/models/RLinf-Pi05-LIBERO-SFT")
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--action-chunk", type=int, default=5)
    ap.add_argument("--num-steps", type=int, default=5)
    ap.add_argument("--num-steps-wait", type=int, default=10)
    ap.add_argument("--scale", type=float, default=0.3,
                    help="single-axis primitive magnitude (delta = +/-scale*e_i)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-json", required=True)
    main(ap.parse_args())
