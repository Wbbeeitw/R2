import json

import pytest
from omegaconf import OmegaConf

from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


def _fake_actor(tmp_path, *, enabled=True, strict=False, rank=1, world_size=4):
    actor = object.__new__(EmbodiedFSDPActor)
    actor.cfg = OmegaConf.create(
        {
            "algorithm": {
                "adv_type": "grpo_degen",
                "degen_state_checkpoint_enable": enabled,
                "degen_state_checkpoint_strict": strict,
            }
        }
    )
    actor._rank = rank
    actor._world_size = world_size
    actor._degen_beta = 0.9
    actor._degen_p_hat = 0.37
    actor._degen_s = 3.7
    actor._degen_n = 10.0
    actor._degen_rescue = True
    actor.logger = type("Logger", (), {"warning": lambda *_args: None})()
    return actor


def test_degen_state_round_trip(tmp_path):
    actor = _fake_actor(tmp_path)
    actor._save_degen_state(str(tmp_path), step=25)

    path = tmp_path / "grpo_degen_state_rank_1.json"
    assert path.exists()
    assert json.loads(path.read_text()) == {
        "beta": 0.9,
        "count_ema": 10.0,
        "p_hat": 0.37,
        "rank": 1,
        "rescue_active": True,
        "schema_version": 1,
        "step": 25,
        "success_ema": 3.7,
        "world_size": 4,
    }

    actor._degen_p_hat = 0.31
    actor._degen_s = 2.79
    actor._degen_n = 10.0
    actor._degen_rescue = False
    actor._load_degen_state(str(tmp_path))

    assert actor._degen_p_hat == pytest.approx(0.37)
    assert actor._degen_s == pytest.approx(3.7)
    assert actor._degen_n == pytest.approx(10.0)
    assert actor._degen_rescue is True


def test_degen_state_disabled_does_not_write(tmp_path):
    actor = _fake_actor(tmp_path, enabled=False)
    actor._save_degen_state(str(tmp_path), step=1)
    assert not list(tmp_path.iterdir())


def test_degen_state_strict_missing_fails(tmp_path):
    actor = _fake_actor(tmp_path, strict=True)
    with pytest.raises(FileNotFoundError, match="state sidecar not found"):
        actor._load_degen_state(str(tmp_path))


def test_degen_state_rejects_wrong_rank(tmp_path):
    actor = _fake_actor(tmp_path)
    actor._save_degen_state(str(tmp_path), step=1)
    state_path = tmp_path / "grpo_degen_state_rank_1.json"
    state = json.loads(state_path.read_text())
    state["rank"] = 0
    state_path.write_text(json.dumps(state))

    with pytest.raises(ValueError, match="rank"):
        actor._load_degen_state(str(tmp_path))
