#!/bin/bash
# Server-2: LIBERO spatial suite JOINT (all 10 tasks, no task_id_filter) A/B.
#   Arm A: vanilla GRPO (adv_type=grpo)   500 steps
#   Arm B: grpo_degen v1 (adv_type=grpo_degen + degen knobs) 500 steps
# Each arm saves checkpoints every 100 steps (100/200/300/400/500) and runs a
# 32-env eval per checkpoint. Run inside the rlinf container (GPUs 4-7 only).
# NOTE: GRPO cannot use runner.use_training_pipeline=True (config.py asserts
# adv_type==gae for pipeline); we run serial. ~50s/step observed on server 2.
# Stage marker: /data/grpo/result/spatial_joint_ab500_stage.txt (monitor this)
set -u

RES=/data/grpo/result
MARK=$RES/spatial_joint_ab500_stage.txt
LOG=$RES/spatial_joint_ab500_orch.log
mkdir -p $RES/spatial_joint_ab500_vanilla $RES/spatial_joint_ab500_degen

log()  { echo "[$(date -u)] $*" >> "$LOG"; }
mark() { echo "$(date -u) $*" > "$MARK"; }

export EMBODIED_PATH=/workspace/RLinf/examples/embodiment
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard
export PYTHONPATH=/workspace/RLinf
export HF_HOME=/data/grpo/model
export HF_ENDPOINT=https://hf-mirror.com
# Container was launched with --gpus '"device=4,5,6,7"', so it only sees 4 GPUs
# (renumbered 0-3). Do NOT set CUDA_VISIBLE_DEVICES here.

cd /workspace/RLinf

mark "scheduled: spatial joint10 vanilla->degen, 500 steps each, eval@100/200/300/400/500 (32 env)"
log "=== orchestrator started $(date -u) ==="

PY=/opt/venv/openpi/bin/python
TRAIN=/workspace/RLinf/examples/embodiment/train_embodied_agent.py
SFT=/data/grpo/model/RLinf-Pi05-LIBERO-SFT
STEPS=500

# --- Train Arm A: vanilla GRPO, 500 steps, ckpt every 100 ---
mark "arma-vanilla"
log "=== ArmA vanilla training START ==="
$PY $TRAIN \
  --config-path /workspace/RLinf/examples/embodiment/config/ \
  --config-name libero_spatial_grpo_openpi_pi05 \
  runner.logger.log_path=$RES/spatial_joint_ab500_vanilla \
  runner.max_steps=$STEPS runner.save_interval=100 \
  actor.model.model_path=$SFT \
  rollout.model.model_path=$SFT \
  actor.fsdp_config.use_orig_params=True \
  actor.micro_batch_size=8 actor.global_batch_size=64 \
  actor.seed=42 \
  env.train.total_num_envs=16 env.train.rollout_epoch=1 \
  algorithm.group_size=4 algorithm.filter_rewards=False \
  algorithm.adv_type=grpo \
  > $RES/spatial_joint_ab500_vanilla/train.log 2>&1
ARMA_EXIT=$?
log "ArmA vanilla training EXIT=$ARMA_EXIT"
if [ "$ARMA_EXIT" -ne 0 ]; then mark "FAIL-arma($ARMA_EXIT)"; exit 1; fi

# --- Train Arm B: grpo_degen v1, 500 steps, ckpt every 100 ---
mark "armb-degen"
log "=== ArmB degen training START ==="
$PY $TRAIN \
  --config-path /workspace/RLinf/examples/embodiment/config/ \
  --config-name libero_spatial_grpo_openpi_pi05 \
  runner.logger.log_path=$RES/spatial_joint_ab500_degen \
  runner.max_steps=$STEPS runner.save_interval=100 \
  actor.model.model_path=$SFT \
  rollout.model.model_path=$SFT \
  actor.fsdp_config.use_orig_params=True \
  actor.micro_batch_size=8 actor.global_batch_size=64 \
  actor.seed=42 \
  env.train.total_num_envs=16 env.train.rollout_epoch=1 \
  algorithm.group_size=4 algorithm.filter_rewards=False \
  algorithm.adv_type=grpo_degen \
  algorithm.degen_p_init=0.31 algorithm.degen_p_decay=0.9 \
  algorithm.degen_lambda_minus=0.25 algorithm.degen_alpha=0.5 \
  algorithm.degen_rescue_group_size=8 algorithm.degen_rescue_noise_scale=1.15 \
  algorithm.degen_rescue_enable=True \
  algorithm.degen_rescue_file=$RES/spatial_joint_ab500_degen/rescue_state.json \
  > $RES/spatial_joint_ab500_degen/train.log 2>&1
ARMB_EXIT=$?
log "ArmB degen training EXIT=$ARMB_EXIT"
if [ "$ARMB_EXIT" -ne 0 ]; then mark "FAIL-armb($ARMB_EXIT)"; exit 1; fi

# --- Eval both arms at step100/200/300/400/500 (32 envs each) ---
PYS="timeout -k 60 2h $PY"
EVAL=/workspace/RLinf/evaluations/eval_embodied_agent.py
EVALCFG="--config-path /workspace/RLinf/evaluations/libero/ --config-name libero_spatial_openpi_pi05_eval"
for ARM in vanilla degen; do
  for STEP in 100 200 300 400 500; do
    CKPT=$RES/spatial_joint_ab500_${ARM}/libero_spatial_grpo_openpi_pi05/checkpoints/global_step_${STEP}/actor/model_state_dict/full_weights.pt
    if [ ! -f "$CKPT" ]; then mark "FAIL-eval-missing-${ARM}-step${STEP}-ckpt"; exit 1; fi
    mark "eval-${ARM}-step${STEP}"
    log "=== EVAL ${ARM} step${STEP} START ==="
    $PYS $EVAL $EVALCFG \
      env.eval.video_cfg.save_video=False env.eval.total_num_envs=32 \
      rollout.model.model_path=$SFT \
      runner.logger.log_path=$RES/eval_spatial_joint_ab500_${ARM}_step${STEP} \
      runner.ckpt_path=$CKPT > $RES/eval_spatial_joint_ab500_${ARM}_step${STEP}.log 2>&1
    E=$?
    log "EVAL ${ARM} step${STEP} EXIT=$E"
    if [ "$E" -ne 0 ]; then mark "FAIL-eval-${ARM}-step${STEP}($E)"; exit 1; fi
  done
done

mark "DONE arma=$ARMA_EXIT armb=$ARMB_EXIT"
echo "spatial_joint_ab500_DONE" > $RES/spatial_joint_ab500_done.txt
