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

"""Corrected-batch dataloader for DARC-VLA L_corr.

L_corr = L_FM(o_c^-, A_c^+) is the *same* flow-matching loss as the normal policy
loss -- only the (observation, actions) pair differs. So no model change is
needed; the correction enters training through the batch stream.

The corrected batch is assembled from a Behavioral Divergence Anchor:
  * observation = the FAIL episode's observation at the anchor frame k
    (base camera + wrist camera video frames, proprio state),
  * actions    = the SUCCESS episode's action chunk starting at the aligned
    success frame j* (the "repair" target).

To guarantee byte-identical batch format with the normal openpi SFT loader, the
corrected sample is produced in the exact raw format the LeRobotDataset yields
(image/wrist_image as float32 [3,H,W] in [0,1], state [8], actions [10,7],
prompt string + metadata) and pushed through the *same* openpi transform chain
(repack -> LiberoInputs -> Normalize -> model transforms). The output is then
wrapped exactly like openpi's DataLoaderImpl: (Observation.from_dict(batch),
batch["actions"]).

Mixing: MixedSftDataLoader interleaves the policy loader and the corrected
loader 1:1, so the effective objective is L = L_policy + L_corr (equal weight).
Delegates ``_data_loader`` / ``data_config()`` so the RLinf worker's batch-count
and attribute probes keep working.

Run standalone to smoke-test one corrected batch:
  EMBODIED_PATH=... LIBERO_REPO_PATH=... PYTHONPATH=... \
  python darc_vla/corrected_sft_data_loader.py --anchors <path> --data-dir <path>
"""

from __future__ import annotations

import os
import json
import pathlib

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import openpi.training.data_loader as _odl
from openpi.models import model as _model

# ---------------------------------------------------------------------------
# raw-sample assembly (mirrors the LeRobotDataset item format)
# ---------------------------------------------------------------------------


def _chw_float(hwc_uint8: np.ndarray) -> np.ndarray:
    """uint8 [H,W,3] -> float32 [3,H,W] in [0,1], matching lerobot decoding."""
    return (hwc_uint8.astype(np.float32) / 255.0).transpose(2, 0, 1)


def _load_task_prompt(data_dir: str, fallback: str | None) -> str:
    """Read the single task instruction from meta/tasks.jsonl (or fallback)."""
    meta = os.path.join(data_dir, "meta", "tasks.jsonl")
    try:
        with open(meta, "r", encoding="utf-8") as f:
            first = json.loads(f.readline())
            return str(first.get("task_index", first.get("task", fallback)))
    except (OSError, json.JSONDecodeError):
        pass
    if fallback:
        return fallback
    raise FileNotFoundError(
        f"cannot read task prompt from {meta}; pass fallback prompt explicitly"
    )


class CorrectedSftDataset(Dataset):
    """Raw corrected samples (fail obs at anchor k, success actions at j*).

    ``__getitem__`` returns a dict whose keys exactly match the LeRobotDataset
    raw item, so the standard openpi transform chain produces an identical batch.
    """

    def __init__(
        self,
        anchors_path: str,
        data_dir: str,
        action_horizon: int = 10,
        prompt: str | None = None,
    ):
        self.anchors = pd.read_parquet(anchors_path)
        self.data_dir = data_dir
        self.action_horizon = action_horizon
        self.prompt = prompt or _load_task_prompt(data_dir, None)
        self._episodes: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    # -- lazy episode (state, actions) from the flat parquet -----------------
    def _episode_data(self, ep: int) -> tuple[np.ndarray, np.ndarray]:
        if ep not in self._episodes:
            p = os.path.join(
                self.data_dir,
                "data",
                f"chunk-{ep // 1000:03d}",
                f"episode_{ep:06d}.parquet",
            )
            df = pd.read_parquet(p)
            self._episodes[ep] = (
                np.asarray(df["state"].tolist(), dtype=np.float32),
                np.asarray(df["actions"].tolist(), dtype=np.float32),
            )
        return self._episodes[ep]

    # -- video frame (uint8 [H,W,3] BGR) ------------------------------------
    def _frame(self, ep: int, cam: str, idx: int) -> np.ndarray:
        import cv2

        p = os.path.join(
            self.data_dir,
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

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, i: int) -> dict:
        row = self.anchors.iloc[i]
        fail_ep = int(row["fail_episode"])
        k = int(row["k"])
        suc_ep = int(row["success_episode"])
        j = int(row["j_star"])

        state_f, _ = self._episode_data(fail_ep)
        _, acts_s = self._episode_data(suc_ep)

        img = self._frame(fail_ep, "image", k)
        wrist = self._frame(fail_ep, "wrist_image", k)

        acts = np.asarray(acts_s[j : j + self.action_horizon], dtype=np.float32)
        if len(acts) < self.action_horizon:  # clamp like lerobot at episode end
            acts = np.pad(
                acts,
                ((0, self.action_horizon - len(acts)), (0, 0)),
                mode="edge",
            )

        # Raw LeRobotDataset-style item. Metadata keys are dropped by the repack
        # transform; they are filled only to keep the schema identical.
        return {
            "image": _chw_float(img),
            "wrist_image": _chw_float(wrist),
            "state": np.asarray(state_f[k], dtype=np.float32),
            "actions": acts,
            "timestamp": np.float32(k / 10.0),
            "frame_index": np.int64(k),
            "episode_index": np.int64(fail_ep),
            "index": np.int64(i),
            "task_index": np.int64(0),
            "done": np.bool_(False),
            "is_success": np.bool_(False),
            "return": np.float32(0.0),
            "reward": np.float32(0.0),
            "prompt": self.prompt,
            "actions_is_pad": np.zeros(self.action_horizon, dtype=bool),
            "task": self.prompt,
        }


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


class _TorchLoaderHolder:
    """Duck-typing shim so RLinf's batch-count probe
    (``get_official_openpi_sft_num_batches`` -> ``loader._data_loader._data_loader``)
    can reach the inner torch DataLoader of a pure-correction loader."""

    def __init__(self, torch_loader):
        self._data_loader = torch_loader


class CorrectedLoader:
    """Infinite corrected-batch iterator, yielding (Observation, actions).

    Duck-types the openpi ``DataLoaderImpl``: ``_data_loader`` and
    ``data_config()`` let the RLinf worker's batch-count / attribute probes
    work in pure-correction mode (L = L_corr only, no policy mixing).
    """

    def __init__(self, torch_loader, seed: int = 0, data_config=None):
        self._torch_loader = torch_loader
        self._seed = seed
        self._data_config = data_config
        self._data_loader = _TorchLoaderHolder(torch_loader)

    def __iter__(self):
        while True:
            data_iter = iter(self._torch_loader)
            for batch in data_iter:
                import jax

                batch = jax.tree.map(torch.as_tensor, batch)
                yield _model.Observation.from_dict(batch), batch["actions"]

    def data_config(self):
        return self._data_config


class MixedSftDataLoader:
    """1:1 interleave of the policy loader and the corrected loader.

    Duck-types the openpi ``DataLoaderImpl`` the RLinf worker expects:
    ``_data_loader`` and ``data_config()`` delegate to the policy loader so
    batch-count / attribute probes behave identically to the unmixed run.
    """

    def __init__(self, policy_loader, corrected_loader):
        self._policy = policy_loader
        self._corrected = corrected_loader

    def __iter__(self):
        pol = iter(self._policy)
        cor = iter(self._corrected)
        while True:
            yield next(pol)  # L_policy batch
            yield next(cor)  # L_corr batch

    @property
    def _data_loader(self):
        return self._policy._data_loader

    def data_config(self):
        return self._policy.data_config()


def build_corrected_sft_loader(
    cfg,
    data_dir: str,
    anchors_path: str,
    *,
    action_horizon: int = 10,
    seed: int = 0,
):
    """Build the infinite corrected loader for the anchor file.

    Uses the *same* openpi config (norm_stats, transforms) as the policy loader.
    """
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    model_cfg = cfg.actor.model
    train_cfg = get_openpi_config(
        model_cfg.openpi.config_name,
        model_path=model_cfg.model_path,
        batch_size=cfg.actor.micro_batch_size,
        repo_id=data_dir,
        data_kwargs=getattr(model_cfg, "openpi_data", None),
    )
    built_data = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)

    ds = CorrectedSftDataset(
        anchors_path, data_dir, action_horizon=action_horizon
    )
    tds = _odl.transform_dataset(ds, built_data)

    torch_loader = torch.utils.data.DataLoader(
        tds,
        batch_size=int(cfg.actor.micro_batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=_odl._collate_fn,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
    )
    return CorrectedLoader(torch_loader, seed=seed, data_config=train_cfg.data)


def build_correction_only_sft_loader(cfg, data_paths):
    """Pure-correction loader: L = L_corr only, no policy batches.

    Opted in via ``cfg.actor.model.openpi.correction.mode == "only"``. The VLM
    stays frozen (the Action-only config), so this isolates "frozen-VLM
    capacity" from "supervision conflict" (DARC E0).
    """
    corr = getattr(cfg.actor.model.openpi, "correction", None)
    if corr is None:
        raise ValueError("correction-only mode requires actor.model.openpi.correction")
    anchors_path = os.path.expanduser(str(corr["anchors_path"]))
    data_dir = str(corr.get("data_dir") or data_paths)
    if not os.path.isfile(anchors_path):
        raise FileNotFoundError(f"correction.anchors_path not found: {anchors_path}")
    loader = build_corrected_sft_loader(
        cfg,
        data_dir,
        anchors_path,
        action_horizon=int(getattr(cfg.actor.model, "num_action_chunks", 10)),
        seed=int(cfg.actor.get("seed", 0)),
    )
    import logging

    logging.info(
        f"[DARC] correction-only (L = L_corr): {anchors_path} "
        f"({len(pd.read_parquet(anchors_path))} anchors, no policy batches)"
    )
    return loader


def build_fixed_correction_batch(
    cfg, data_dir, anchors_path, n: int = 32, *, action_horizon: int = 10
):
    """Collate the first ``n`` anchors (fixed order, no shuffle) into a list of
    single-sample ``(Observation, actions)`` batches for the fixed-noise eval
    monitor. Same set every call -> apples-to-apples L_corr curve when the
    worker also reseeds the torch global RNG before each forward."""
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    model_cfg = cfg.actor.model
    train_cfg = get_openpi_config(
        model_cfg.openpi.config_name,
        model_path=model_cfg.model_path,
        batch_size=1,
        repo_id=data_dir,
        data_kwargs=getattr(model_cfg, "openpi_data", None),
    )
    built_data = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    ds = CorrectedSftDataset(
        anchors_path, data_dir, action_horizon=action_horizon
    )
    tds = _odl.transform_dataset(ds, built_data)
    n = min(n, len(tds))
    return [
        (_model.Observation.from_dict(b), b["actions"])
        for i in range(n)
        for b in [_odl._collate_fn([tds[i]])]
    ]


def build_mixed_sft_dataloader(policy_loader, cfg, data_paths):
    """Wrap the policy loader with 1:1 corrected batches (opt-in via config).

    Active only when ``cfg.actor.model.openpi.correction`` is set:
      actor.model.openpi.correction:
        anchors_path: <anchors_train.parquet>
        data_dir:     <same LeRobot dataset>
    """
    corr = getattr(cfg.actor.model.openpi, "correction", None)
    if corr is None:
        return policy_loader
    anchors_path = os.path.expanduser(str(corr["anchors_path"]))
    data_dir = str(corr.get("data_dir") or data_paths)
    if not os.path.isfile(anchors_path):
        raise FileNotFoundError(f"correction.anchors_path not found: {anchors_path}")
    corrected = build_corrected_sft_loader(
        cfg,
        data_dir,
        anchors_path,
        action_horizon=int(getattr(cfg.actor.model, "num_action_chunks", 10)),
        seed=int(cfg.actor.get("seed", 0)),
    )
    import logging

    logging.info(
        f"[DARC] L_corr mixing enabled: {anchors_path} "
        f"(policy + corrected batches 1:1)"
    )
    return MixedSftDataLoader(policy_loader, corrected)


# ---------------------------------------------------------------------------
# standalone smoke test
# ---------------------------------------------------------------------------


def _smoke(anchors_path: str, data_dir: str) -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "actor": {
                "micro_batch_size": 1,
                "seed": 0,
                "model": {
                    "model_path": "/workspace/models/RLinf-Pi05-LIBERO-SFT",
                    "num_action_chunks": 10,
                    "openpi": {
                        "config_name": "pi05_libero",
                        "openpi_data": {
                            "norm_stats_path": "/workspace/models/RLinf-Pi05-LIBERO-SFT/"
                            "physical-intelligence/libero/norm_stats.json"
                        },
                    },
                },
            }
        }
    )
    loader = build_corrected_sft_loader(cfg, data_dir, anchors_path)
    obs, actions = next(iter(loader))
    print("actions:", tuple(actions.shape), actions.dtype)
    for cam, img in obs.images.items():
        print(f"images[{cam}]:", tuple(img.shape), img.dtype,
              "min=", float(img.min()), "max=", float(img.max()))
    for cam, m in obs.image_masks.items():
        print(f"image_masks[{cam}]:", tuple(m.shape), m.dtype)
    print("state:", tuple(obs.state.shape), obs.state.dtype)
    print("tokenized_prompt:", tuple(obs.tokenized_prompt.shape),
          obs.tokenized_prompt.dtype)
    print("tokenized_prompt_mask:", tuple(obs.tokenized_prompt_mask.shape),
          obs.tokenized_prompt_mask.dtype)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--data-dir", required=True)
    _smoke(ap.parse_args().anchors, ap.parse_args().data_dir)
