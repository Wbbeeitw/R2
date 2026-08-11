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

"""DARC-VLA Behavioral Divergence Anchor detector (locked v0.4 spec).

Finds divergence anchors on paired success/failure rollouts:

  * Pairing: first-frame state KDTree (< 0.02, raw space) over all success
    episodes' first frames -- reproduces the v0.4 2097/2097 pairing rate.
  * W=5 local time alignment: for fail frame i, j* = argmin over the window
    j in [i-W, i+W] of D_s(o_i^- , o_j^+)  (per-dim standardized L2).
  * H=5 persistence window: the paired drift after the anchor must stay apart,
    median_{k=1..H} D_s(o_{i+k}^-, o_{j*+k}^+) > D_s(i) + eps.
  * Gates (defaults locked): D_s < P_s-th percentile, D_a > P_a-th percentile,
    on per-dim standardized state/action; arm/gripper split available.

The visual gate (D_init = D_proprio + beta*D_visual) is a real-robot (variable
layout) filter; in fixed-layout LIBERO it is a near no-op (Yellow 4), so this
detector is proprio-only for Stage 0.

Outputs (parquet):
  anchors_train.parquet   -- per surviving fail episode: earliest anchor
                             (fail_ep, k, success_ep, j*, D_s, D_a,
                              A_c_minus, A_c_plus).  Used to build corrected
                             batches for L_corr.
  anchors_heldout.parquet -- same, for held-out fail episodes (CPR eval, never
                             seen in training).
  report.json             -- survival rate, anchor position median, action
                             divergence breakdown (full/arm/gripper).
"""

from __future__ import annotations

import argparse
import json
import os
import glob

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------

def load_episodes(data_dir: str):
    """Load all episodes. Returns dict: ep_idx -> {state, actions, is_success}."""
    files = sorted(glob.glob(os.path.join(data_dir, "data", "chunk-*", "episode_*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet files under {data_dir}/data/chunk-*")
    episodes = {}
    for f in files:
        df = pd.read_parquet(f)
        ep = int(df["episode_index"].iloc[0])
        episodes[ep] = {
            "state": np.asarray(df["state"].tolist(), dtype=np.float32),
            "actions": np.asarray(df["actions"].tolist(), dtype=np.float32),
            "is_success": bool(df["is_success"].iloc[0]),
        }
    return episodes


def episode_obs_refs(ep: int, frame: int) -> dict:
    """LeRobot v2.0 path reconstruction for an (episode, frame) observation."""
    c = ep // 1000
    return {
        "episode_index": int(ep),
        "frame": int(frame),
        "video_image": f"videos/chunk-{c:03d}/image/episode_{ep:06d}.mp4",
        "video_wrist_image": f"videos/chunk-{c:03d}/wrist_image/episode_{ep:06d}.mp4",
    }


# ---------------------------------------------------------------------------
# detector
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="LeRobot v2.0 dataset dir")
    ap.add_argument("--out-dir", required=True, help="output dir for anchor parquet")
    ap.add_argument("--W", type=int, default=5, help="local alignment half-window")
    ap.add_argument("--H", type=int, default=5, help="persistence window")
    ap.add_argument("--P-s", type=float, default=30.0, help="state-close percentile (D_s below this)")
    ap.add_argument("--P-a", type=float, default=60.0, help="action-diff percentile (D_a above this)")
    ap.add_argument("--eps", type=float, default=0.005, help="divergence margin")
    ap.add_argument("--pair-dist", type=float, default=0.02, help="first-frame pairing threshold")
    ap.add_argument("--heldout-frac", type=float, default=0.20, help="fraction of fail episodes held out")
    ap.add_argument(
        "--persist-mode",
        choices=["paired", "sameidx"],
        default="sameidx",
        help="persistence drift: 'paired' keeps the aligned offset (D_s(o_{i+k}^-, o_{j*+k}^+)); "
        "'sameidx' compares the same index (D_s(o_{i+k}^-, o_{i+k}^+), the v0.4 wording).",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[detector] loading episodes from {args.data_dir}")
    episodes = load_episodes(args.data_dir)
    eps_idx = sorted(episodes.keys())
    n_succ = sum(1 for e in eps_idx if episodes[e]["is_success"])
    n_fail = sum(1 for e in eps_idx if not episodes[e]["is_success"])
    print(f"[detector] {len(eps_idx)} episodes: {n_succ} success / {n_fail} fail")

    # global per-dim std of state and actions (full data)
    all_state = np.concatenate([episodes[e]["state"] for e in eps_idx], axis=0)
    all_act = np.concatenate([episodes[e]["actions"] for e in eps_idx], axis=0)
    state_sigma = all_state.std(axis=0) + 1e-6
    act_sigma = all_act.std(axis=0) + 1e-6
    del all_state, all_act
    print(f"[detector] state sigma={np.round(state_sigma, 4)}")
    print(f"[detector] act   sigma={np.round(act_sigma, 4)}")

    success_eps = [e for e in eps_idx if episodes[e]["is_success"]]
    fail_eps = [e for e in eps_idx if not episodes[e]["is_success"]]

    # pairing: first-frame state KDTree over success episodes (raw space)
    first_states = np.stack([episodes[e]["state"][0] for e in success_eps])
    tree = cKDTree(first_states)
    pairs = {}  # fail_ep -> success_ep
    for e in fail_eps:
        d, idx = tree.query(episodes[e]["state"][0], k=1)
        if d < args.pair_dist:
            pairs[e] = success_eps[int(idx)]
    n_paired = len(pairs)
    print(f"[detector] paired {n_paired}/{n_fail} fail episodes (threshold {args.pair_dist})")

    # held-out split on fail episodes (before anchor detection so CPR anchors
    # come from episodes whose corrected data never enters training)
    fail_eps_arr = sorted(pairs.keys())
    rng.shuffle(fail_eps_arr)
    n_held = max(1, int(len(fail_eps_arr) * args.heldout_frac))
    heldout_fails = set(fail_eps_arr[:n_held])
    train_fails = fail_eps_arr[n_held:]

    W, H = args.W, args.H
    d_s = np.zeros(0)  # all aligned state distances (for P_s percentile)
    d_a = np.zeros(0)  # all action distances (for P_a percentile)
    # per-pair aligned info, stored as dict lists to avoid a giant matrix
    pair_aligned = []  # list of (fail_ep, success_ep, d_s_aligned, d_a_arr, jstar)

    print("[detector] computing aligned D_s / D_a for all paired fail episodes ...")
    for e in train_fails + list(heldout_fails):
        suc = pairs[e]
        sf = episodes[e]["state"] / state_sigma[None, :]   # standardized fail state
        sa = episodes[suc]["state"] / state_sigma[None, :]  # standardized success state
        af = episodes[e]["actions"] / act_sigma[None, :]
        aa = episodes[suc]["actions"] / act_sigma[None, :]
        Lf, Ls = sf.shape[0], sa.shape[0]

        # pairwise squared distances fail x success (standardized L2)
        # M[i, j] = ||sf[i] - sa[j]||^2
        M2 = (
            (sf[:, None, :] - sa[None, :, :]) ** 2
        ).sum(axis=2)  # (Lf, Ls)
        A2 = (
            (af[:, None, :] - aa[None, :, :]) ** 2
        ).sum(axis=2)

        Lmin = min(Lf, Ls)
        n_cand = Lmin - 2 * W - H + 1  # candidate offsets i = W .. Lmin-W-H
        if n_cand <= 0:
            pair_aligned.append((e, suc, np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)))
            continue
        i0, i1 = W, Lmin - W - H  # inclusive candidate range

        # local alignment over window: j* = argmin_{j in [max(0,i-W), min(Ls-1,i+W)]} M2[i,j]
        # vectorized: build shifted index windows
        rows = np.arange(i0, i1 + 1)
        js = np.clip(rows[None, :] + np.arange(-W, W + 1)[:, None], 0, Ls - 1)
        win_M = M2[rows[None, :], js]  # (2W+1, n_cand)
        jstar = js[np.argmin(win_M, axis=0), np.arange(win_M.shape[1])]  # (n_cand,)
        d_s_al = np.sqrt(win_M[np.argmin(win_M, axis=0), np.arange(win_M.shape[1])])
        d_a_arr = np.sqrt(A2[rows, jstar])

        pair_aligned.append((e, suc, d_s_al, d_a_arr, jstar))
        d_s = np.concatenate([d_s, d_s_al])
        d_a = np.concatenate([d_a, d_a_arr])

    p_s = np.percentile(d_s, args.P_s)
    p_a = np.percentile(d_a, args.P_a)
    print(f"[detector] P_s={args.P_s} -> D_s threshold {p_s:.4f} (n={len(d_s)})")
    print(f"[detector] P_a={args.P_a} -> D_a threshold {p_a:.4f} (n={len(d_a)})")

    # persistence + gating, per pair -> earliest anchor per episode
    def detect_anchors(fail_eps_list, persist_mode):
        anchors = []
        for e in fail_eps_list:
            rec = next((r for r in pair_aligned if r[0] == e), None)
            if rec is None:
                continue
            suc, d_s_al, d_a_arr, jstar = rec[1], rec[2], rec[3], rec[4]
            if d_s_al.size == 0:
                continue
            sf = episodes[e]["state"] / state_sigma[None, :]
            sa = episodes[suc]["state"] / state_sigma[None, :]
            Lf, Ls = sf.shape[0], sa.shape[0]
            i0 = W
            i1 = d_s_al.size - 1  # last candidate index offset

            for off in range(d_s_al.size):
                i = i0 + off
                j = jstar[off]
                if d_s_al[off] >= p_s or d_a_arr[off] <= p_a:
                    continue
                # persistence: median drift over next H frames
                if i + H >= Lf:
                    continue
                if persist_mode == "paired":
                    if j + H >= Ls:
                        continue
                    drifts = np.sqrt(
                        ((sf[i + 1:i + H + 1] - sa[j + 1:j + H + 1]) ** 2).sum(axis=1)
                    )
                else:  # sameidx: fail i+k vs success i+k
                    if i + H >= Ls:
                        continue
                    drifts = np.sqrt(
                        ((sf[i + 1:i + H + 1] - sa[i + 1:i + H + 1]) ** 2).sum(axis=1)
                    )
                if np.median(drifts) > d_s_al[off] + args.eps:
                    row = {
                        "fail_episode": int(e),
                        "k": int(i),
                        "success_episode": int(suc),
                        "j_star": int(j),
                        "D_s": float(d_s_al[off]),
                        "D_a": float(d_a_arr[off]),
                        "A_c_minus": episodes[e]["actions"][i].tolist(),
                        "A_c_plus": episodes[suc]["actions"][j].tolist(),
                    }
                    row.update(episode_obs_refs(e, i))
                    row.update(
                        {"obs_success_episode": int(suc), "obs_success_frame": int(j)}
                    )
                    anchors.append(row)
                    break  # earliest anchor per episode
        df = pd.DataFrame(anchors)
        return df

    df_train = detect_anchors(train_fails, args.persist_mode)
    df_held = detect_anchors(list(heldout_fails), args.persist_mode)

    # report
    surv = len(df_train) + len(df_held)
    pos_med = None
    if surv:
        all_df = pd.concat([df_train, df_held])
        # trajectory-relative anchor position: frac of fail episode length
        rel = [
            r["k"] / max(1, len(episodes[r["fail_episode"]]["state"]))
            for _, r in all_df.iterrows()
        ]
        pos_med = float(np.median(rel))
        a_full = np.array(all_df["A_c_minus"].tolist()) - np.array(all_df["A_c_plus"].tolist())
        a_full_l2 = np.linalg.norm(a_full, axis=1)
        a_arm = np.linalg.norm(a_full[:, :6], axis=1)
        a_gr = np.abs(a_full[:, 6])
        rep = {
            "n_paired": n_paired,
            "n_fail_total": n_fail,
            "n_anchors_total": surv,
            "survival_rate": surv / max(1, n_paired),
            "anchor_pos_median_frac": pos_med,
            "action_divergence_full_med": float(np.median(a_full_l2)),
            "action_divergence_arm_med": float(np.median(a_arm)),
            "action_divergence_gripper_med": float(np.median(a_gr)),
            "gripper_flip_rate": float(np.mean(np.abs(a_full[:, 6]) > 0.5)),
            "P_s_threshold": float(p_s),
            "P_a_threshold": float(p_a),
        }
    else:
        rep = {"n_paired": n_paired, "n_fail_total": n_fail, "n_anchors_total": 0}
    print("[detector] report:")
    print(json.dumps(rep, indent=2))

    out_train = os.path.join(args.out_dir, "anchors_train.parquet")
    out_held = os.path.join(args.out_dir, "anchors_heldout.parquet")
    df_train.to_parquet(out_train, index=False)
    df_held.to_parquet(out_held, index=False)
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(f"[detector] wrote {out_train} ({len(df_train)} rows), {out_held} ({len(df_held)} rows)")


if __name__ == "__main__":
    main()
