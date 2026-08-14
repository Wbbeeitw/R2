# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch

from rlinf.algorithms.registry import register_advantage
from rlinf.algorithms.utils import kl_penalty, safe_normalize
from rlinf.utils.utils import masked_mean


@register_advantage("gae")
def compute_gae_advantages_and_returns(
    rewards: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    values: Optional[torch.Tensor] = None,
    normalize_advantages: bool = True,
    normalize_returns: bool = False,
    loss_mask: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate advantages and returns for Proximal Policy Optimization (PPO).
    NOTE: currently this function does not support auto-reset.

    This function implements Generalized Advantage Estimation (GAE) to compute
    advantages and returns for PPO training. The advantages are normalized
    using mean and standard deviation for stable training.

    Args:
        rewards (torch.Tensor): Rewards per timestep. Shape: [seq_len, bsz].
        values (torch.Tensor): Value function estimates. Shape: [seq_len, bsz].
        dones (torch.Tensor): Done flags (1 if episode ended, else 0).
        gamma (float, optional): Discount factor. Defaults to 1.0.
        gae_lambda (float, optional): GAE smoothing factor. Defaults to 1.0.
        normalize_advantages (bool, optional): Whether to normalize advantages. Defaults to True.
        normalize_returns (bool, optional): Whether to normalize returns. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (advantages, returns)
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0

    critic_free = values is None
    if critic_free:
        gae_lambda = 1
        gamma = 1

    for step in reversed(range(T)):
        if critic_free:
            delta = rewards[step]
        else:
            delta = (
                rewards[step]
                + gamma * values[step + 1] * (~dones[step + 1])
                - values[step]
            )

        gae = delta + gamma * gae_lambda * (~dones[step + 1]) * gae
        returns[step] = gae if critic_free else gae + values[step]

    advantages = returns - values[:-1] if not critic_free else returns

    if normalize_advantages:
        advantages = safe_normalize(advantages, loss_mask=loss_mask)
    if normalize_returns:
        returns = safe_normalize(returns, loss_mask=loss_mask)

    return advantages, returns


@register_advantage("grpo")
def compute_grpo_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    **kwargs,
):
    """
    Compute GRPO advantages.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups, group_size]
        loss_mask (torch.Tensor): Loss mask for valid entries. Shape: [num_groups, group_size]
        group_size (int): Group size for advantage computation.

    Returns:
        torch.Tensor: advantages
    """
    grouped_rewards = rewards.view(-1, group_size)

    grouped_reward_mean = grouped_rewards.mean(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )
    grouped_reward_std = grouped_rewards.std(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )

    advantages = grouped_rewards - grouped_reward_mean
    advantages = advantages / (grouped_reward_std + 1e-6)

    advantages = (torch.zeros_like(loss_mask) + advantages.view(1, -1)) * loss_mask

    return advantages, None


@register_advantage("grpo_degen")
def compute_grpo_degen_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    degen_mode: str = "prefix",
    **kwargs,
):
    """Compute GRPO advantages with degeneracy-triggered reward shaping.

    Vanilla GRPO collapses to zero advantage whenever a group's reward std is
    zero (every trajectory scored identically). For sparse-reward VLA tasks this
    is the common case: an all-success group (every score == max) and an
    all-failure group (every score == 0) both have std == 0, so the model gets
    no gradient even though "which success finished sooner" and "which failure
    got further" are both informative.

    This variant leaves mixed groups (std > 0) untouched, and only recovers a
    relative ranking from trajectory structure when a group degenerates:

    - all-success: rank by completion efficiency (earlier completion wins),
      derived from the first `done` step in the available `dones` tensor.
    - all-failure: rank by conservative valid-prefix length — the earliest
      suspicious action chunk (from the per-step `suspicious` boundary flags)
      minus a one-chunk safety margin. Unlocalized trajectories (no boundary
      detected) get zero credit (宁缺毋滥) rather than being forced to a full
      prefix.

    The all-failure treatment has two modes (`degen_mode`):

    - "prefix" (default, main method): temporal cut. The min-max failure ranking
      is broadcast only to the clean prefix `t < f_i`; actions after the
      boundary get advantage 0 and are zeroed out of the returned loss mask, so
      they never enter the policy-gradient denominator. Ranking is min-max so
      the worst localized failure gets r=0 and the best gets r=1 (no negative
      advantage for short prefixes).
    - "ranking": ablation. The same min-max failure ranking, but broadcast over
      the whole trajectory with no temporal cut (loss mask returned unchanged).

    Args:
        rewards: Per-trajectory scores. Shape [num_groups, group_size].
        loss_mask: Loss mask for valid entries. Shape [n_steps, bsz].
        group_size: Group size for advantage computation.
        degen_mode: "prefix" (default) or "ranking".
        dones: (in kwargs) Done flags. Shape [n_steps + 1, bsz].
        suspicious: (in kwargs) Per-chunk boundary flags. Shape [n_steps, bsz].

    Returns:
        Tuple[advantages, None, loss_mask]: the third element is the (possibly
        prefix-masked) loss mask, so the caller can propagate the temporal cut
        into the policy-gradient denominator.
    """
    eps = 1e-6
    n_steps, bsz = loss_mask.shape
    num_groups = bsz // group_size
    device = loss_mask.device

    # Per-trajectory valid length (loss mask is 1 up to the first done step).
    T_i = loss_mask.sum(dim=0).clamp(min=1).float()  # [bsz]

    grouped_rewards = rewards.view(num_groups, group_size)  # [num_groups, group_size]
    group_mean = grouped_rewards.mean(dim=-1, keepdim=True)
    group_std = grouped_rewards.std(dim=-1, keepdim=True)
    vanilla = (grouped_rewards - group_mean) / (group_std + eps)

    # Degenerate groups (std == 0): the vanilla advantage is identically zero.
    degenerate = group_std.squeeze(-1) < eps  # [num_groups]
    all_success = degenerate & (group_mean.squeeze(-1) > 0)
    all_failure = degenerate & (group_mean.squeeze(-1) <= 0)

    # Per-trajectory scalar advantage (mixed keeps vanilla) and clean-prefix
    # length (full valid length unless a failure boundary is cut).
    traj_adv = vanilla.view(-1)  # [bsz]
    prefix_len = T_i.clone()  # [bsz]

    # --- all-success: completion-efficiency ranking ---
    # A success step sets `dones` to 1, so the first done index is the
    # completion step; earlier is better. The score carries no timing signal
    # (all successes score the same), which is exactly why vanilla degenerates.
    dones = kwargs.get("dones", None)
    if all_success.any() and dones is not None:
        n_steps_d = dones.shape[0] - 1  # dones is [n_steps + 1, bsz]
        dones_bool = dones.bool()
        first_done = dones_bool.float().argmax(dim=0)  # [bsz]
        any_done = dones_bool.any(dim=0)
        first_done = torch.where(
            any_done, first_done, torch.full_like(first_done, n_steps_d)
        )
        completion_step = first_done.float().view(num_groups, group_size)
        max_step = completion_step.max(dim=-1, keepdim=True).values
        efficiency = max_step - completion_step  # earlier completion -> larger
        eff_mean = efficiency.mean(dim=-1, keepdim=True)
        eff_std = efficiency.std(dim=-1, keepdim=True)
        completion = (efficiency - eff_mean) / (eff_std + eps)
        success_mask = all_success.repeat_interleave(group_size)  # [bsz]
        traj_adv = torch.where(success_mask, completion.view(-1), traj_adv)

    # --- all-failure: min-max boundary ranking (+ temporal cut in prefix mode) ---
    suspicious = kwargs.get("suspicious", None)
    if all_failure.any() and suspicious is not None:
        susp_bool = suspicious.bool()  # [n_steps, bsz]
        first_susp = susp_bool.float().argmax(dim=0)  # [bsz]; 0 if never suspicious
        localized = susp_bool.any(dim=0)  # [bsz]
        # Conservative boundary: one chunk safety margin before the first
        # suspicious action chunk.
        f_i = (first_susp.float() - 1).clamp(min=1)  # [bsz]

        # Quality = clean-prefix fraction of the trajectory; unlocalized -> 0.
        q_i = torch.where(localized, f_i / T_i, torch.zeros_like(f_i))  # [bsz]
        q_g = q_i.view(num_groups, group_size)
        loc_g = localized.view(num_groups, group_size)

        # Min-max over localized members only (unlocalized excluded from the
        # normalization, 宁缺毋滥).
        q_min = torch.where(
            loc_g, q_g, torch.full_like(q_g, float("inf"))
        ).min(dim=-1, keepdim=True).values
        q_max = torch.where(
            loc_g, q_g, torch.full_like(q_g, float("-inf"))
        ).max(dim=-1, keepdim=True).values

        r_g = (q_g - q_min) / (q_max - q_min + eps)
        # No spread among localized members -> reinforce the common boundary.
        no_spread = (q_max - q_min) < eps
        r_g = torch.where(no_spread, torch.ones_like(r_g), r_g)
        # Unlocalized get zero credit.
        r_g = torch.where(loc_g, r_g, torch.zeros_like(r_g))

        r_i = r_g.view(-1)  # [bsz]
        failure_mask = all_failure.repeat_interleave(group_size)  # [bsz]

        traj_adv = torch.where(failure_mask, r_i, traj_adv)
        if degen_mode == "prefix":
            # Cut the loss mask at the failure boundary for localized failures.
            prefix_len = torch.where(failure_mask & localized, f_i, prefix_len)

    # Broadcast the per-trajectory advantage over time, then apply the temporal
    # cut (prefix_len == T_i everywhere except localized failures in prefix
    # mode, so mixed/success/unlocalized-failure keep the full valid region).
    t_idx = torch.arange(n_steps, device=device, dtype=prefix_len.dtype).unsqueeze(-1)
    prefix_mask = t_idx < prefix_len.unsqueeze(0)  # [n_steps, bsz]

    advantages = (
        traj_adv.unsqueeze(0).expand(n_steps, bsz)
        * loss_mask.float()
        * prefix_mask.float()
    )
    ff_loss_mask = loss_mask.bool() & prefix_mask

    return advantages, None, ff_loss_mask


@register_advantage("grpo_dynamic")
def compute_grpo_dynamic_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    idx_to_traj: list[int],
    advantage_mode: str = "turn",  # "trajectory" or "turn"
    **kwargs,
):
    """
    Compute GRPO advantages for multi-turn multi-agent scenarios.

    IMPORTANT: This function computes advantages PER QUESTION, not globally.
    - idx_to_traj maps turn_idx -> global_traj_idx (e.g., [0,0,1,1,2,2,3,3,4,4,...,15,15])
    - Trajectories 0-3 belong to question 0, 4-7 to question 1, etc.
    - We must compute GRPO separately for each question's group_size trajectories

    Two advantage computation modes:
    1. "trajectory": Trajectory-level GRPO (Method 1)
       - Compute mean/std over group_size trajectory rewards per question
       - Broadcast same advantage to all turns in a trajectory
       - Example: Q0 has 4 trajs with 1,2,3,4 turns. Compute GRPO over 4 traj rewards,
                  then assign traj0_adv to its 1 turn, traj1_adv to its 2 turns, etc.

    2. "turn": Turn-level GRPO (Method 2)
       - Compute mean/std over all turns within each question
       - Example: Q0 has 4 trajs with 1,2,3,4 turns = 10 turns total.
                  Compute GRPO over these 10 turn rewards (currently all same within traj).
       - Future-proof: works when turns have different rewards within same trajectory

    Args:
        rewards: Shape [num_sequence, 1] after preprocessing (num_sequence = total turns)
        loss_mask: Shape [seq_len, num_sequence] after preprocessing
        group_size: Number of trajectories per question (e.g., 4)
        idx_to_traj: List mapping turn_idx -> global_traj_idx
        advantage_mode: "trajectory" or "turn"

    Returns:
        advantages: Shape [seq_len, num_sequence]
    """
    num_sequence = len(idx_to_traj)

    rewards_flat = rewards.squeeze(-1)

    assert rewards_flat.numel() == num_sequence, (
        f"Rewards size mismatch: {rewards_flat.numel()} != {num_sequence}"
    )

    num_trajectories = max(idx_to_traj) + 1
    num_questions = num_trajectories // group_size
    assert num_trajectories % group_size == 0, (
        f"num_trajectories {num_trajectories} not divisible by group_size {group_size}"
    )

    turn_advantages = torch.zeros(
        num_sequence, dtype=rewards.dtype, device=rewards.device
    )

    if advantage_mode == "trajectory":
        # Aggregate turn rewards into per-trajectory rewards first.
        trajectory_rewards = torch.zeros(
            num_trajectories, dtype=rewards.dtype, device=rewards.device
        )
        trajectory_counts = torch.zeros(
            num_trajectories, dtype=torch.long, device=rewards.device
        )

        for turn_idx, traj_idx in enumerate(idx_to_traj):
            trajectory_rewards[traj_idx] += rewards_flat[turn_idx]
            trajectory_counts[traj_idx] += 1

        # Step 1: Average rewards per trajectory.
        trajectory_rewards = trajectory_rewards / trajectory_counts.clamp(min=1).float()

        # Step 2: reshape to [num_questions, group_size] for per-question GRPO.
        trajectory_rewards_grouped = trajectory_rewards.view(num_questions, group_size)

        # Step 3: compute per-question mean and std.
        per_question_mean = trajectory_rewards_grouped.mean(
            dim=-1, keepdim=True
        )  # [num_questions, 1]
        per_question_std = trajectory_rewards_grouped.std(
            dim=-1, keepdim=True
        )  # [num_questions, 1]

        # Step 4: normalize within each question group.
        normalized_trajectory_rewards = (
            trajectory_rewards_grouped - per_question_mean
        ) / (per_question_std + 1e-6)  # [num_questions, group_size]

        # Step 5: flatten back to [num_trajectories].
        normalized_trajectory_rewards = normalized_trajectory_rewards.view(-1)

        # Step 6: broadcast trajectory advantages to all turns in that trajectory.
        for turn_idx, traj_idx in enumerate(idx_to_traj):
            turn_advantages[turn_idx] = normalized_trajectory_rewards[traj_idx]

    elif advantage_mode == "turn":
        # Step 1: map each turn to its owning question.
        turn_to_question = torch.tensor(
            [idx_to_traj[i] // group_size for i in range(num_sequence)],
            dtype=torch.long,
            device=rewards.device,
        )

        # Step 2: normalize turn rewards within each question group.
        for question_idx in range(num_questions):
            question_mask = turn_to_question == question_idx
            question_turn_rewards = rewards_flat[question_mask]

            # Step 3: compute mean and std for all turns in this question.
            question_mean = question_turn_rewards.mean()
            question_std = question_turn_rewards.std()

            # Step 4: normalize turn rewards within the question.
            normalized_question_rewards = (question_turn_rewards - question_mean) / (
                question_std + 1e-6
            )

            # Step 5: write normalized turn-level advantages back.
            turn_advantages[question_mask] = normalized_question_rewards

    else:
        raise ValueError(
            f"Invalid advantage_mode: {advantage_mode}. Must be 'trajectory' or 'turn'"
        )

    advantages = torch.zeros_like(
        loss_mask, dtype=rewards.dtype
    ) + turn_advantages.view(1, -1)
    advantages = advantages * loss_mask

    return advantages, None


@register_advantage("reinpp")
def compute_reinpp_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    use_reinpp_baseline: bool = False,
    kl_beta: float = 0.0,
    logprob=None,
    ref_logprob=None,
    kl_penalty_type: str = "",
    **kwargs,
):
    """
    Compute advantages for reinforce++ and reinforce++ baseline.

    Args:
        rewards (torch.Tensor): The reward or score values.
        loss_mask (torch.Tensor): The loss mask for valid entries.
        group_size (int): The group size for advantage computation.
        use_reinpp_baseline (bool, optional): Whether to use reinforce++ baseline.
        kl_beta (float, optional): KL penalty coefficient.
        logprob (optional): Log probability of current policy.
        ref_logprob (optional): Log probability of reference policy.
        kl_penalty_type (str, optional): Type of KL penalty.

    Returns:
        torch.Tensor: advantages
    """
    # first group baseline for reinforce++ baseline
    if use_reinpp_baseline:
        grouped_rewards = rewards.view(-1, group_size)  # [num_prompt, group_size]
        grouped_rewards -= grouped_rewards.mean(dim=1, keepdims=True)
        rewards = grouped_rewards.view(-1)  # [B]

    # build the reward matrix
    r_matrix = torch.zeros_like(loss_mask).float()  # [L, B]
    seq_length = loss_mask.size(0)
    mask_flipped = loss_mask.long().fliplr()
    eos_positions = mask_flipped.argmax(
        dim=0, keepdim=True
    )  # position of last True in original mask
    eos_indices = seq_length - 1 - eos_positions  # [1, B]

    r_matrix = r_matrix.scatter_(dim=0, index=eos_indices, src=rewards)  # [L, B]

    # add kl penalty
    if kl_beta > 0:
        kld = kl_penalty(logprob, ref_logprob, kl_penalty=kl_penalty_type)  # [L, B]
        r_matrix -= kl_beta * kld

    # compute return
    ret_matrix = torch.cumsum(r_matrix.flip(dims=[0]), dim=0).flip(dims=[0])

    # normalize
    advantages = ret_matrix.clone()

    mean = masked_mean(advantages, loss_mask)
    var = masked_mean((advantages - mean).pow(2), loss_mask)
    rstd = var.clamp(min=1e-8).rsqrt()

    advantages = (advantages - mean) * rstd

    return advantages, None


@register_advantage("opd")
def compute_opd_advantages(
    prev_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    normalize_advantages: bool = False,
    **kwargs,
):
    """Compute OPD advantages from frozen teacher token log-probabilities."""
    assert teacher_logprobs is not None, (
        "OPD advantage computation requires post-rollout teacher_logprobs."
    )
    assert prev_logprobs is not None, (
        "OPD advantage computation requires prev_logprobs from student rollout."
    )
    assert teacher_logprobs.shape == prev_logprobs.shape, (
        f"teacher_logprobs shape {teacher_logprobs.shape} must match "
        f"prev_logprobs shape {prev_logprobs.shape}."
    )
    assert not normalize_advantages, (
        "VLA-OPD uses raw reverse-KL rewards; set normalize_advantages to False."
    )
    num_action_chunks = kwargs.get("num_action_chunks", None)
    assert num_action_chunks is not None, (
        "OPD advantage computation requires num_action_chunks."
    )
    advantages = teacher_logprobs.float() - prev_logprobs.float()
    assert advantages.shape[-1] % num_action_chunks == 0, (
        f"OPD token count {advantages.shape[-1]} must be divisible by "
        f"num_action_chunks {num_action_chunks}."
    )
    advantages = advantages.reshape(*advantages.shape[:-1], num_action_chunks, -1)
    if loss_mask is not None:
        target_steps = loss_mask.shape[0]
        assert advantages.shape[0] in {target_steps, target_steps + 1}, (
            f"OPD advantages time dimension {advantages.shape[0]} must match "
            f"loss_mask time dimension {target_steps} or include one bootstrap step."
        )
        advantages = advantages[:target_steps]

    return advantages, None


@register_advantage("raw")
def compute_raw_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    normalize_advantages: bool = False,
    **kwargs,
):
    """
    Return raw rewards or normalized rewards.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups, group_size]
        loss_mask (torch.Tensor): Loss mask for valid entries. Shape: [num_groups, group_size]
        normalize_advantages (bool): Whether to normalize advantages.

    Returns:
        torch.Tensor: advantages
    """
    if rewards.ndim == 2:
        rewards = rewards.reshape(-1)
    advantages = rewards.unsqueeze(0).expand_as(loss_mask) * loss_mask

    # Simple baseline subtraction (mean of valid advantages)
    if normalize_advantages:
        valid = advantages[loss_mask.bool()]
        if valid.numel() > 0:
            advantages = (advantages - valid.mean()) / (valid.std() + 1e-5)

    return advantages, None
