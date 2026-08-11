# DARC-VLA Stage 0 工程路线图（四组消融）

> 状态：2026-08-11 · 数据管线已通（Gate 0.5 冒烟 ✅）
> 方法 v0.4 已锁：Behavioral Divergence Anchor → { L_corr = FM(o_c⁻, A_c⁺) 借成功分支动作纠正；L_state = VLM-LoRA outcome ranking }。
> 指标：**SR** + **CPR**（held-out anchors 上 `P(‖Â−A_c⁺‖ < ‖Â−A_c⁻‖)`）。

## 1. 四组 Loss 组成（锁死）

| 组 | 配置 | Loss | VLM | Action Expert | 今天可跑 |
|---|---|---|---|---|---|
| Base | `libero10_task0_sft_openpi_pi05.yaml` | L_policy | 训 | 训 | ✅ 已验证 |
| Action-only | `..._action_only_openpi_pi05.yaml` | L_policy + L_corr | 冻结 | 训 | 骨架 ✅（L_corr 数据侧待接） |
| State-only | `..._state_only_openpi_pi05.yaml` | L_policy + L_state | 训 | 训 | ❌ 待 L_state |
| Full | `..._full_openpi_pi05.yaml` | L_policy + L_corr + L_state | 训 | 训 | ❌ 待 L_state + L_corr |

统一：1 seed，2k–3k steps，micro_batch=1 / global=2，lr 2.5e-5 cosine。loss 加权求和单次 backward（Stage 0 不需要真正的三流独立 backward；那是后补优化，不影响消融结论）。

## 2. 数据侧：离线 anchor 标注 + L_corr 校正 batch（Track A）

**核心洞见：L_corr 与 L_policy 是同一个 FM loss，只是喂不同的 (obs, actions)。** 官方 openpi 路径的 SFT loss 由模型前向从 batch 内部算（`model(forward_type=SFT, data=batch)`，target 即 `batch["actions"]`）。所以 **L_corr 不需要任何模型改动**——只要构造 observation=o_c⁻、actions=A_c⁺ 的 batch。

1. **检测器定稿**（`darc_vla/detector.py`，从 v0.4 升级版移植）：
   - `D_s`/`D_a` 用逐维标准化 state；arm/gripper 拆分门控；W=5 局部对齐 + H=5 持续性窗口。
   - 配对：首帧 state KDTree（0.02）+ 真机视觉门控 `D_init = D_proprio + β·D_visual`。
   - 输出：每 fail episode 的 anchor 列表 + 匹配成功分支 `(ep, j*, step k)`。
   - ε 按存活率曲线选（Yellow 2 表可复现；默认 P_s30/P_a60/ε_Δ0.005）。

2. **校正 batch 数据集**（`darc_vla/build_corrected_dataset.py`）：
   - 对每个 anchor `(i⁻, j*, k)`：取失败轨迹 o_c⁻（anchor 附近窗口帧）+ 成功分支 A_c⁺（delta 动作），拼成合成 episode（obs 用失败帧，actions 换成 A_c⁺）。
   - 按 openpi LeRobot 格式落盘（或生成索引供 loader 动态组装）。
   - 同时产出一份**held-out anchors**（CPR 评测用，训练不见）。

3. **dataloader 挂接**（`rlinf/data/datasets/openpi_rlinf/official_sft_data_loader.py`）：
   - `build_official_openpi_sft_dataloader` 加参数 `corrected_data_paths`，把校正 batch 与正常 batch 按比例混入（如 1:1）。
   - 复用同一 `create_data_loader`；数据 key 对齐 repack（`observation/image→image` 等，见 `dataconfig/libero_dataconfig.py`）。

## 3. 模型侧：L_state ranking head（Track B）

1. **Ranking head**（`rlinf/models/embodiment/openpi/openpi_action_model.py` 的 `OpenPi0ForRLActionPrediction`）：
   - 在 `paligemma_with_expert` 的 VLM 特征上做 pooling（mean/last/first_token，参照 RLinf-native 的 `value_vlm_mode` 机制：`openpi_rlinf/rl_action_model.py:44`）→ 线性层 → scalar score → softplus。
   - L_state = pairwise margin / binary CE：anchor 对 `(o_c⁻, A⁺分支) > (o_c⁻, A⁻分支)`，即"同一 context 下成功分支的 consequence 表征分更高"。
2. **VLM 训练**：第一版直接全量训 VLM（ranking objective）；LoRA 是后补的参数高效化（RLinf-native gemma 内建 `lora_configs`，official openpi 路径要另加）。
3. **loss 汇合**：L_state 与 L_policy（+L_corr）加权求和进单次 backward（`rlinf/workers/sft/fsdp_vla_sft_worker.py:82 get_train_model_output` 返回 dict loss）。

## 4. CPR 评测（`darc_vla/eval_cpr.py`）

- 输入：held-out anchors `(o_c⁻, A_c⁺, A_c⁻)` + 训练后的 policy。
- 对每个 anchor：`Â = policy(o_c⁻)`（rollout/eval forward），`CPR = P(‖Â−A_c⁺‖ < ‖Â−A_c⁻‖)`。
- 即使 SR 只涨 3pp，CPR 也能告诉我们"动作纠正学进去没有"。

## 5. 建议实施顺序

1. ~~Gate 0.5 冒烟~~ ✅（已完成，log 在 `/workspace/workspcae/gate0_5/`）
2. 检测器定稿 + 离线 anchor 标注（`darc_vla/detector.py`）→ 落 anchor parquet
3. L_corr 校正 batch 数据集 + dataloader 混入 → **Action-only 真跑**（此时四组中 Base/Action-only 可跑）
4. L_state ranking head + worker 汇合 → **State-only / Full**（四组齐）
5. CPR eval + SR 评测 → Stage 0 结论

## 6. 关键文件锚点

| 文件 | 位置 |
|---|---|
| 官方 openpi SFT loader | `rlinf/data/datasets/openpi_rlinf/official_sft_data_loader.py` |
| SFT worker（loss 入口） | `rlinf/workers/sft/fsdp_vla_sft_worker.py:82` |
| FSDP 梯度累积 | `rlinf/workers/sft/fsdp_sft_worker.py:47-55` |
| 模型构建/冻结 | `rlinf/models/embodiment/openpi/__init__.py:67`（train_expert_only → freeze_vlm） |
| freeze_vlm | `rlinf/models/embodiment/openpi/openpi_action_model.py:1404` |
| VLM pooling 参照 | `rlinf/models/embodiment/openpi_rlinf/rl_action_model.py:44`（value_vlm_mode） |
| 数据 repack/key 对齐 | `rlinf/models/embodiment/openpi/dataconfig/libero_dataconfig.py` |
| checkpoint 路径 | `rlinf/runners/sft_runner.py:179`（`{log_path}/{experiment_name}/checkpoints/global_step_{N}/actor`） |
| 启动方式（禁 run_vla_sft.sh） | `examples/sft/config/libero10_task0_sft_openpi_pi05.yaml` 头注释 |
