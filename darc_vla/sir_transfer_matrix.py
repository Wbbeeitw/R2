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

"""SIR basin-geometry B0: verified-patch transfer matrix.

GPT review (2026-08-12): the question is NOT "can a successful delta
transfer?" but "does environment-verification enrich the proposal
distribution?"  A raw transfer rate of ~30% proves nothing on its own --
the matched random baseline (sigma=0.1) already recovers ~25% per proposal.
So B0 compares verified patches against that baseline.

Design
------
  * delta_i+ = min-norm successful patch of source trial i
    (argmin ||delta|| over the executed candidates with R=1).  Avoids dragging
    rollout order into the experiment (first-positive would) and matches SIR's
    minimum-behavioral-edit prior.  A source with a single positive uses that
    one.
  * T[i,j] = R(s_j, delta_i+): roll target j deterministically to the fixed
    intervention step C, execute  A~_C(j) = base_chunk_j + alpha*tanh(delta_i+)
    (only the 7-D residual transfers; the base chunk is the TARGET's own),
    then hand control back to base.  Protocol identical to
    sir_critic_select.py collection, so pre-C states match bit-for-bit.
  * targets = ALL failed trials (13), including the 6 hard failures (0/8).
    A rescue of a hard target is the strongest evidence of transferable
    corrective structure; if hard targets stay at 0 we get the clean
    decomposition recoverable-vs-irrecoverable that bounds SIR's applicability.
  * diagonal cells T[i,i] get --diag-repeats extra runs (3 total) as
    reproducibility sanity.  The protocol is seed-deterministic, so a 0 means
    the "verified patch" was a lucky sample, not a patch.

Metrics (all on the executed matrix, zero new rollout beyond the matrix)
-----------------------------------------------------------------------
  PTR  = off-diagonal transfer rate          (compare vs RR(sigma=0.1) ~ 25%)
  G_i  = mean_{j!=i} T[i,j]                  (source generality)
  C_j  = max_i T[i,j]                        (target coverability = BestOf7
                                              from the verified library)
  D    = mean_i T[i,i]                       (diagonal reproducibility)
plus (optional, --with-matched-random) per-target K random deltas
      N(0, 0.1^2 I) for BestOf7(verified) vs BestOf7(random, sigma=0.1).
      Off by default per GPT staging -- run only if the verified matrix shows
      structure.

Decision signals (mechanical, for the A/B/C/D rules)
----------------------------------------------------
  A  transfer >> random and several G_i >> 25%  -> Verified Patch Bank /
     Retrieval SIR (no neural selector needed).
  B  no overall lift but block structure in T   -> cluster / retrieval-key
     search (pose / object-relative geometry).
  C  transfer ~ random, no structure            -> drop retrieval, go to the
     intervention-primitive map.
  D  transfer ~ 0 and diagonals fragile         -> state-specific search
     (large data / vision becomes justified).
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
    """Roll deterministically up to the intervention step and return
    (state_C, base_chunk).  Seed-identical to collection, so the base chunk
    passed into _roll_intervene matches what collection executed."""
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
    Returns R in {0,1}."""
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
            # npz stores sigma as float32 (0.1 -> 0.10000000149...), so a raw
            # `== 0.1` (float64) comparison would silently count zero matches.
            "sigma": float(round(float(d["sigma"][i]), 4)),
            "R": bool(d["R"][i]),
        })
    with open(path.replace(".npz", "_meta.json")) as f:
        meta = json.load(f)
    return rows, meta


def _min_norm_positive(rows, trial):
    """delta_i+ = argmin_{delta: R=1} ||delta|| for a source trial."""
    pos = [r for r in rows if r["trial"] == trial and r["R"]]
    if not pos:
        return None
    return min(pos, key=lambda r: float(np.linalg.norm(r["delta"])))


def main(args):
    rows, meta = _load_data(args.data_npz)
    trials_fail = [int(t) for t in meta["trials_fail"]]
    sources = [t for t in trials_fail
               if any(r["trial"] == t and r["R"] for r in rows)]
    targets = trials_fail
    patches = {t: _min_norm_positive(rows, t) for t in sources}
    C = int(meta["C"])
    alpha = float(meta["alpha"])
    max_steps = int(meta["max_steps"])
    sigmas = [float(s) for s in meta["sigmas"]]
    base_seed = int(meta["seed"])

    print(f"[transfer] {meta['task']!r} | C={C} alpha={alpha} "
          f"max_steps={max_steps}")
    print(f"[transfer] failed trials {targets} (n={len(targets)})")
    for t in sources:
        p = patches[t]
        n_pos = sum(1 for r in rows if r["trial"] == t and r["R"])
        print(f"[transfer] source trial {t}: positive {n_pos} | min-norm "
              f"patch ||d||={np.linalg.norm(p['delta']):.4f} "
              f"sigma={p['sigma']} delta={np.round(p['delta'], 3).tolist()}")

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
    print("[transfer] policy loaded")

    # base_chunk per target: cached once (seed-identical to collection)
    base_chunk = {}
    for j in tqdm.tqdm(targets, desc="roll-to-C"):
        seed_r = base_seed * 10000 + j
        _, bc = _roll_to_C(env, policy, init_states[j], C, args.num_steps_wait,
                           args.action_chunk, seed_r, prompt)
        base_chunk[j] = bc

    # main verified-transfer matrix + diagonal repeats
    T = {}
    diag = {}
    for i in tqdm.tqdm(sources, desc="sources"):
        d = patches[i]["delta"]
        T[i] = {}
        for j in targets:
            seed_r = base_seed * 10000 + j
            ok = _roll_intervene(
                env, policy, init_states[j], C, base_chunk[j], d, alpha,
                seed_r, prompt, max_steps, args.num_steps_wait,
                args.action_chunk,
            )
            T[i][j] = bool(ok)
        diag[i] = [bool(T[i][i])]
        for _rep in range(args.diag_repeats):
            seed_r = base_seed * 10000 + i
            ok = _roll_intervene(
                env, policy, init_states[i], C, base_chunk[i], d, alpha,
                seed_r, prompt, max_steps, args.num_steps_wait,
                args.action_chunk,
            )
            diag[i].append(bool(ok))

    # optional matched-random library per target (sigma=0.1, GPT staging)
    matched = None
    if args.with_matched_random:
        rng = np.random.default_rng(base_seed + 1000)
        matched = {}
        for j in tqdm.tqdm(targets, desc="matched random"):
            matched[j] = []
            for _ in range(args.matched_k):
                delta = (0.1 * rng.standard_normal(7)).astype(np.float32)
                seed_r = base_seed * 10000 + j
                ok = _roll_intervene(
                    env, policy, init_states[j], C, base_chunk[j], delta,
                    alpha, seed_r, prompt, max_steps, args.num_steps_wait,
                    args.action_chunk,
                )
                matched[j].append(bool(ok))

    env.close()

    # ---- metrics ----------------------------------------------------------
    off = [(i, j) for i in sources for j in targets if j != i]
    ptr = sum(T[i][j] for i, j in off) / len(off)
    G = {i: float(np.mean([T[i][j] for j in targets if j != i]))
         for i in sources}
    Cj = {j: float(np.max([T[i][j] for i in sources])) for j in targets}
    best7_verified = float(np.mean(list(Cj.values())))
    D = {i: diag[i] for i in sources}
    d_ok = {i: diag[i][0] for i in sources}
    diag_rate = float(np.mean([diag[i][0] for i in sources]))
    # pooled RR at sigma=0.1 from the collection data
    rr01 = (sum(1 for r in rows if r["sigma"] == 0.1 and r["R"]),
            sum(1 for r in rows if r["sigma"] == 0.1))
    rr01_rate = rr01[0] / rr01[1] if rr01[1] else 0.0

    n_pos_sources = sum(1 for t in sources if diag[t][0])
    print("\n[transfer] matrix rows=source, cols=target ('#'=R=1)")
    header = "        " + " ".join(f"{j:3d}" for j in targets)
    print(header)
    for i in sources:
        row = " ".join("#" if T[i][j] else "." for j in targets)
        print(f"  d{i:3d}   {row}   G={G[i]:.3f}")

    print(f"\n[transfer] off-diagonal PTR = {ptr:.3f} "
          f"({sum(T[i][j] for i, j in off)}/{len(off)})")
    print(f"[transfer] RR(sigma=0.1) from collection = "
          f"{rr01[0]}/{rr01[1]} = {rr01_rate:.3f}")
    print(f"[transfer] PTR - RR_0.1 = {ptr - rr01_rate:+.3f}")
    print(f"[transfer] source generality G = "
          + ", ".join(f"trial {i}: {G[i]:.3f}" for i in sources))
    print(f"[transfer] target coverability C_j (BestOf7 verified): "
          + ", ".join(f"{j}:{'#' if Cj[j] > 0 else '.'}" for j in targets))
    print(f"[transfer] BestOf7(verified) = {best7_verified:.3f}")
    print(f"[transfer] diagonal: " + "; ".join(
        f"trial {i}: {diag[i]}" for i in sources))
    print(f"[transfer] diagonal-first-run rate = {diag_rate:.3f} "
          f"({n_pos_sources}/{len(sources)})")
    if matched:
        best7_rand = float(np.mean([any(matched[j]) for j in targets]))
        rr_match = float(np.mean([np.mean(matched[j]) for j in targets]))
        print(f"[transfer] BestOf7(random,sigma=.1) = {best7_rand:.3f}")
        print(f"[transfer] matched-random per-proposal rate = {rr_match:.3f}")
        print(f"[transfer] BestOf7(verified) - BestOf7(random) = "
              f"{best7_verified - best7_rand:+.3f}")

    gap = ptr - rr01_rate
    n_generous = sum(1 for i in sources if G[i] >= 0.40)
    if gap >= 0.15 and n_generous >= 2:
        decision = ("A: transfer >> random with multiple general sources -> "
                    "Verified Patch Bank / Retrieval SIR (no selector)")
    elif gap <= -0.05 and diag_rate >= 0.8:
        decision = ("C: transfer ~ random, no lift -> drop retrieval, run the "
                    "intervention-primitive map")
    else:
        decision = ("B/C boundary: look at block structure in T before "
                    "choosing retrieval-key search vs primitive map")
    print(f"[transfer] decision signal: {decision}")

    report = {
        "task": meta["task"],
        "task_suite": meta["task_suite"],
        "task_id": meta["task_id"],
        "C": C, "alpha": alpha, "max_steps": max_steps,
        "sources": sources,
        "patch_norms": {str(t): float(np.linalg.norm(patches[t]["delta"]))
                        for t in sources},
        "targets": targets,
        "matrix": {str(i): {str(j): bool(T[i][j]) for j in targets}
                   for i in sources},
        "diagonal_repeats": {str(i): [bool(x) for x in diag[i]]
                             for i in sources},
        "ptr": float(ptr),
        "n_off_diag": len(off),
        "rr_sigma0_1": [rr01[0], rr01[1]],
        "ptr_minus_rr01": float(gap),
        "source_generality": {str(i): float(G[i]) for i in sources},
        "target_coverability": {str(j): float(Cj[j]) for j in targets},
        "best7_verified": float(best7_verified),
        "diag_rate": float(diag_rate),
        "matched_random": ({"per_trial": {str(j): [bool(x) for x in matched[j]]
                                          for j in targets},
                            "best7_random": float(best7_rand),
                            "per_proposal_rate": float(rr_match)}
                           if matched else None),
        "decision_signal": decision,
        "recovery_by_sigma": {
            str(s): (
                int(sum(1 for r in rows if r["sigma"] == s and r["R"])),
                int(sum(1 for r in rows if r["sigma"] == s)),
            )
            for s in sigmas
        },
    }
    print(json.dumps(report, indent=2))
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[transfer] wrote {args.out_json}")


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
    ap.add_argument("--diag-repeats", type=int, default=2,
                    help="extra re-runs of each diagonal cell (3 total)")
    ap.add_argument("--with-matched-random", action="store_true",
                    help="also execute K random N(0,0.1^2 I) deltas per target "
                         "for the BestOf7 verified-vs-random contrast")
    ap.add_argument("--matched-k", type=int, default=7)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-json", required=True)
    main(ap.parse_args())
