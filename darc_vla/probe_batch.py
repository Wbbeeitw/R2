# Copyright 2026 The DARC-VLA Authors.
"""Probe: dump ONE real openpi SFT batch structure from the RLinf loader.

Runs on the server container. Goal: pin the exact (observation, actions)
contract that sft_forward consumes, so corrected batches (o_c^-, A_c^+) can be
assembled to match it bit-for-bit.

  cd /workspace/RLinf && \
  EMBODIED_PATH=/workspace/RLinf/examples/sft \
  LIBERO_REPO_PATH=/opt/venv/openpi/libero \
  PYTHONPATH=/workspace/RLinf:/opt/venv/openpi/libero \
  /opt/venv/openpi/bin/python /workspace/workspcae/probe_batch.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/workspace/RLinf")

import hydra
import numpy as np
from omegaconf import DictConfig


def shape_of(x):
    if isinstance(x, np.ndarray):
        return {"type": "np.ndarray", "shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, (list, tuple)):
        return {"type": type(x).__name__, "items": [shape_of(v) for v in x]}
    if isinstance(x, dict):
        return {k: shape_of(v) for k, v in x.items()}
    if isinstance(x, str):
        return {"type": "str", "value": x[:120]}
    if x is None:
        return {"type": "None"}
    return {"type": type(x).__name__, "repr": str(x)[:200]}


def leaf_sample(x, prefix, out):
    if isinstance(x, dict):
        for k, v in x.items():
            leaf_sample(v, f"{prefix}/{k}", out)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            leaf_sample(v, f"{prefix}[{i}]", out)
    elif isinstance(x, np.ndarray):
        flat = x.ravel()
        out.append(
            {
                "key": prefix,
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "min": float(np.min(flat)) if flat.size and flat.dtype.kind in "fc" else None,
                "max": float(np.max(flat)) if flat.size and flat.dtype.kind in "fc" else None,
                "first": x.ravel()[0].tolist() if x.size else None,
            }
        )
    else:
        out.append({"key": prefix, "type": type(x).__name__, "repr": str(x)[:200]})


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="libero10_task0_sft_openpi_pi05",
)
def run(cfg: DictConfig) -> None:
    _main(cfg)


def _main(cfg) -> None:
    os.chdir("/workspace/RLinf")
    from rlinf.data.datasets.openpi_rlinf.official_sft_data_loader import (
        build_official_openpi_sft_dataloader,
    )

    loader, data_config = build_official_openpi_sft_dataloader(
        cfg,
        world_size=1,
        rank=0,
        data_paths=cfg.data.train_data_paths,
        eval_dataset=False,
    )
    print(f"[probe] action_sequence_keys={data_config.action_sequence_keys}")
    print(f"[probe] use_quantile_norm={data_config.use_quantile_norm}")
    print(f"[probe] prompt_from_task={data_config.prompt_from_task}")

    # raw LeRobotDataset item (BEFORE any transform) -- the exact format a
    # corrected-batch dataset must mimic to run through the same pipeline.
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    import openpi.training.data_loader as _odl

    train_cfg = get_openpi_config(
        "pi05_libero",
        model_path=cfg.actor.model.model_path,
        batch_size=cfg.actor.micro_batch_size,
        repo_id=cfg.data.train_data_paths,
        data_kwargs=dict(cfg.actor.model.openpi_data),
    )
    built_data = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    raw_ds = _odl.create_torch_dataset(
        built_data, train_cfg.model.action_horizon, train_cfg.model
    )
    raw_item = raw_ds[0]
    print("\n===== RAW LeRobotDataset ITEM (pre-transform) =====")
    print("keys:", list(raw_item.keys()))
    for k, v in raw_item.items():
        if isinstance(v, dict):
            print(f"  {k}: dict -> { {kk: (str(type(vv)), getattr(vv, 'shape', None)) for kk, vv in v.items()} }")
        else:
            print(f"  {k}: {type(v).__name__} shape={getattr(v, 'shape', None)} dtype={getattr(v, 'dtype', None)}")
            if isinstance(v, str):
                print(f"      prompt: {v[:120]}")
            if isinstance(v, np.ndarray) and v.size and v.dtype.kind in "fc":
                print(f"      min={v.ravel().min()} max={v.ravel().max()}")

    # raw batch dict from the inner torch loader
    raw = next(iter(loader._data_loader))
    print("\n===== RAW BATCH DICT KEYS =====")
    print(json.dumps(shape_of(raw), indent=2, default=str)[:4000])

    print("\n===== RAW BATCH LEAF SUMMARY =====")
    leaves = []
    leaf_sample(raw, "", leaves)
    for leaf in leaves:
        print(json.dumps(leaf, default=str))

    # the (Observation, actions) tuple that sft_forward actually receives
    obs, actions = next(iter(loader))
    print("\n===== Observation.from_dict pytree =====")
    print("actions:", type(actions), getattr(actions, "shape", None), getattr(actions, "dtype", None))
    fields = {}
    for fname in obs.__dataclass_fields__:
        val = getattr(obs, fname)
        fields[fname] = shape_of(val) if fname not in ("tokenized_prompt", "tokenized_prompt_mask") else {
            "shape": list(np.asarray(val).shape),
            "dtype": str(np.asarray(val).dtype),
            "sample": np.asarray(val).ravel()[:12].tolist(),
        }
    print(json.dumps(fields, indent=2, default=str)[:6000])


if __name__ == "__main__":
    run()
