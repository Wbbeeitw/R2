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

"""CPR (Correction-Pull Rate) evaluation for DARC-VLA L_corr.

For every held-out Behavioral Divergence Anchor -- fail observation at frame k,
success repair action A_c^+ = success[j*], fail action A_c^- = fail[k] --
sample the policy at o_c^- and measure whether the first predicted action
lands closer to the success action than to the fail action:

    CPR_full = P[ d(A_hat[0], A_c^+) < d(A_hat[0], A_c^-) ]

split into arm (dims 0:6) and gripper (dim 6). Distances are raw-space L2
over the 7-dim LIBERO action. The Policy wrapper decodes the model output back
to raw space (Normalize -> model -> Unnormalize), so no normalization math is
needed on our side -- A_c^+ / A_c^- are the raw values stored in the anchor.

The policy is loaded exactly like RLinf's standalone eval
(``toolkits/standalone_eval_scripts/openpi``): ``create_trained_policy`` ->
Policy wrapper -> ``policy.infer({"observation/...": ...})["actions"]``.

The corrected observation is built from the *real dataset frames* -- no 180 deg
rotation (unlike libero_eval those frames are already in the exact orientation
the model was trained on).

Usage (base model, establishes CPR_base):
  python darc_vla/cpr_eval.py \\
      --checkpoint-dir /workspace/models/RLinf-Pi05-LIBERO-SFT \\
      --anchors /workspace/workspcae/darc_vla_anchors/anchors_heldout.parquet \\
      --data-dir /workspace/datasets/recap_libero10_task0/libero10_task0_train \\
      --num-samples 3 --out-json /workspace/workspcae/cpr_base.json

Usage (post-trained RLinf SFT ckpt -- auto-converts to openpi deploy format):
  python darc_vla/cpr_eval.py \\
      --sft-ckpt /workspace/workspcae/<run>/global_step_<N> \\
      --reference-model /workspace/models/RLinf-Pi05-LIBERO-SFT \\
      --deploy-out /workspace/workspcae/cpr_deploy \\
      --anchors /workspace/workspcae/darc_vla_anchors/anchors_heldout.parquet \\
      --data-dir /workspace/datasets/recap_libero10_task0/libero10_task0_train \\
      --out-json /workspace/workspcae/cpr_post.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

from darc_vla.corrected_sft_data_loader import _load_task_prompt


# ---------------------------------------------------------------------------
# policy loading (mirrors toolkits/standalone_eval_scripts/openpi/__init__.py)
# ---------------------------------------------------------------------------


def load_policy(checkpoint_dir: str, num_steps: int = 5):
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from toolkits.standalone_eval_scripts.openpi import create_trained_policy

    train_cfg = get_openpi_config(
        "pi05_libero",
        model_path=checkpoint_dir,
    )
    return create_trained_policy(
        train_cfg,
        checkpoint_dir,
        sample_kwargs={"num_steps": num_steps},
    )


# ---------------------------------------------------------------------------
# raw corrected observation assembly (real dataset frames, no rotation)
# ---------------------------------------------------------------------------


def _read_episode_state(data_dir: str, ep: int) -> np.ndarray:
    p = os.path.join(
        data_dir, "data", f"chunk-{ep // 1000:03d}", f"episode_{ep:06d}.parquet"
    )
    df = pd.read_parquet(p)
    return np.asarray(df["state"].tolist(), dtype=np.float32)


def _read_frame(data_dir: str, ep: int, cam: str, idx: int) -> np.ndarray:
    """uint8 [H,W,3] RGB frame at idx (cv2 mp4 seek)."""
    import cv2

    p = os.path.join(
        data_dir,
        "videos",
        f"chunk-{ep // 1000:03d}",
        cam,
        f"episode_{ep:06d}.mp4",
    )
    cap = cv2.VideoCapture(p)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError(f"failed to read frame {idx} of {p}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def build_observation(
    data_dir: str, fail_ep: int, k: int, prompt: str
) -> dict:
    """Raw observation dict exactly as libero_eval feeds the Policy wrapper."""
    state = _read_episode_state(data_dir, fail_ep)[k]
    img = _read_frame(data_dir, fail_ep, "image", k)
    wrist = _read_frame(data_dir, fail_ep, "wrist_image", k)
    return {
        "observation/state": np.asarray(state, dtype=np.float32),
        "observation/image": img,
        "observation/wrist_image": wrist,
        "prompt": prompt,
    }


# ---------------------------------------------------------------------------
# CPR computation
# ---------------------------------------------------------------------------


def evaluate(policy, anchors: pd.DataFrame, data_dir: str, prompt: str,
             num_samples: int, seed: int) -> dict:
    del seed  # policy sampling is fresh-noise i.i.d. per call; no seeding needed
    # per (anchor, sample) distances: [n_anchors, num_samples]
    d_full_p = np.zeros((len(anchors), num_samples))
    d_full_m = np.zeros((len(anchors), num_samples))
    d_arm_p = np.zeros((len(anchors), num_samples))
    d_arm_m = np.zeros((len(anchors), num_samples))
    d_gr_p = np.zeros((len(anchors), num_samples))
    d_gr_m = np.zeros((len(anchors), num_samples))

    for i, row in anchors.iterrows():
        fail_ep, k = int(row["fail_episode"]), int(row["k"])
        a_plus = np.asarray(row["A_c_plus"], dtype=np.float32)    # 7 raw
        a_minus = np.asarray(row["A_c_minus"], dtype=np.float32)  # 7 raw
        obs = build_observation(data_dir, fail_ep, k, prompt)

        for s in range(num_samples):
            policy.reset()
            out = policy.infer(obs)
            a_hat = np.asarray(out["actions"][0], dtype=np.float32)  # [7] raw
            if a_hat.shape[0] < 7:
                raise RuntimeError(
                    f"policy returned {a_hat.shape} actions; expected >= 7"
                )
            a_hat = a_hat[:7]
            diff = a_hat - a_plus
            difm = a_hat - a_minus
            d_full_p[i, s] = np.linalg.norm(diff)
            d_full_m[i, s] = np.linalg.norm(difm)
            d_arm_p[i, s] = np.linalg.norm(diff[:6])
            d_arm_m[i, s] = np.linalg.norm(difm[:6])
            d_gr_p[i, s] = abs(diff[6])
            d_gr_m[i, s] = abs(difm[6])

    # pooled over all (anchor, sample) pairs
    return {
        "n_anchors": len(anchors),
        "n_samples": num_samples,
        "n_pairs": int(len(anchors) * num_samples),
        "cpr_full": float(np.mean(d_full_p < d_full_m)),
        "cpr_arm": float(np.mean(d_arm_p < d_arm_m)),
        "cpr_grip": float(np.mean(d_gr_p < d_gr_m)),
        "mean_d_plus_full": float(np.mean(d_full_p)),
        "mean_d_minus_full": float(np.mean(d_full_m)),
        "mean_d_plus_arm": float(np.mean(d_arm_p)),
        "mean_d_minus_arm": float(np.mean(d_arm_m)),
        "mean_d_plus_grip": float(np.mean(d_gr_p)),
        "mean_d_minus_grip": float(np.mean(d_gr_m)),
    }


# ---------------------------------------------------------------------------
# RLinf SFT ckpt -> openpi deploy dir (model.safetensors) for create_trained_policy
# ---------------------------------------------------------------------------


def convert_sft_to_deploy(
    sft_ckpt: str,
    reference_model: str,
    deploy_out: str,
) -> str:
    """Strip FSDP wrappers + convert to the openpi PyTorch layout.

    Mirrors sft2deploy steps 1+2 but produces a ``model.safetensors`` deploy
    dir (what ``create_trained_policy`` loads) instead of a deploy .pt.
    """
    import tempfile

    from rlinf.utils.ckpt_convertor.openpi import (
        openpi_rlinf_to_openpi_pytorch as new2old,
    )
    from rlinf.utils.ckpt_convertor.openpi.sft2deploy import (
        sft_to_new_safetensors,
    )

    os.makedirs(deploy_out, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cpr_convert_", dir=deploy_out) as tmp:
        bare = sft_to_new_safetensors(sft_ckpt, os.path.join(tmp, "new"))
        new2old.convert_trained_ckpt(
            input_ckpt=str(bare),
            output_dir=deploy_out,
            reference_model=reference_model,
            norm_stats=None,
        )
    # copy norm_stats so create_trained_policy resolves asset "physical-intelligence/libero"
    src = os.path.join(
        reference_model, "physical-intelligence", "libero", "norm_stats.json"
    )
    dst_dir = os.path.join(
        deploy_out, "physical-intelligence", "libero"
    )
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dst_dir, "norm_stats.json"))
    print(f"[cpr] wrote deploy model -> {deploy_out}/model.safetensors")
    return deploy_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", default=None,
                    help="openpi deploy dir with model.safetensors (base model)")
    ap.add_argument("--sft-ckpt", default=None,
                    help="RLinf SFT full_weights.pt / global_step dir; auto-converted")
    ap.add_argument("--reference-model",
                    default="/workspace/models/RLinf-Pi05-LIBERO-SFT")
    ap.add_argument("--deploy-out", default="/workspace/workspcae/cpr_deploy")
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--num-steps", type=int, default=5)
    ap.add_argument("--num-samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    if args.checkpoint_dir:
        ckpt = args.checkpoint_dir
    elif args.sft_ckpt:
        ckpt = convert_sft_to_deploy(
            args.sft_ckpt, args.reference_model, args.deploy_out
        )
    else:
        ap.error("need --checkpoint-dir or --sft-ckpt")

    anchors = pd.read_parquet(args.anchors)
    print(f"[cpr] {len(anchors)} anchors; checkpoint dir = {ckpt}")

    prompt = _load_task_prompt(args.data_dir, None)
    policy = load_policy(ckpt, num_steps=args.num_steps)
    print("[cpr] policy loaded")

    report = evaluate(
        policy, anchors, args.data_dir, prompt,
        num_samples=args.num_samples, seed=args.seed,
    )
    report["checkpoint_dir"] = ckpt
    report["anchors_path"] = args.anchors
    report["num_steps"] = args.num_steps
    print(json.dumps(report, indent=2))
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[cpr] wrote {args.out_json}")


if __name__ == "__main__":
    main()
