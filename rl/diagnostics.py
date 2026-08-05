"""Reusable training-curve sanity checks (Step 10), so "confirm reward
curves are sane before trusting any comparison in Step 11" is actual,
tested code rather than the ad hoc quartile-printing scripts used to
investigate Step 9's training runs. Any future retraining (e.g. revisiting
RL with the Step 10 FIFO-priority fix applied) should run its callback's
episode stats through this before treating the result as trustworthy.

Deliberately conservative: this flags concerns, it doesn't declare a
training run "good" -- Step 9's own results (reward improved and then
plateaued, inventory std dropped substantially, no instability, and the
policy *still* underperformed hand-tuned baselines) are the reminder that
a clean-looking training curve doesn't guarantee a competitive policy.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np


@dataclasses.dataclass
class TrainingDiagnosis:
    n_episodes: int
    first_quartile_reward_mean: float
    last_quartile_reward_mean: float
    first_quartile_reward_std: float
    last_quartile_reward_std: float
    first_quartile_inventory_std_mean: float
    last_quartile_inventory_std_mean: float
    learning_detected: bool  # last-quartile mean reward clearly better than first
    variance_exploding: bool  # last-quartile reward std substantially larger than first
    inventory_control_improved: bool
    concerns: list[str]

    @property
    def is_sane(self) -> bool:
        return len(self.concerns) == 0


def diagnose_training(
    episode_rewards: list[float],
    episode_inventory_stds: list[float],
    min_episodes: int = 20,
    improvement_threshold: float = 0.2,
    variance_explosion_ratio: float = 1.5,
) -> TrainingDiagnosis:
    """Compute sanity diagnostics from an EpisodeStatsCallback's logs.

    `improvement_threshold`: fraction the last quartile's mean reward must
    exceed the first quartile's by (relative to the first quartile's
    spread) to count as "learning detected." `variance_explosion_ratio`:
    how much larger the last quartile's reward std can be than the
    first's before it's flagged as instability rather than normal
    exploration noise.
    """
    concerns: list[str] = []
    n = len(episode_rewards)

    if n == 0:
        raise ValueError("episode_rewards is empty -- nothing to diagnose")
    if len(episode_inventory_stds) != n:
        raise ValueError("episode_rewards and episode_inventory_stds must be the same length")

    rewards = np.array(episode_rewards, dtype=np.float64)
    inv_stds = np.array(episode_inventory_stds, dtype=np.float64)

    if not np.all(np.isfinite(rewards)):
        concerns.append("non-finite (NaN/Inf) values found in episode_rewards")

    if n < min_episodes:
        concerns.append(f"only {n} episodes logged (< {min_episodes}) -- quartile stats will be noisy")

    q = max(1, n // 4)
    first_q, last_q = rewards[:q], rewards[-q:]
    first_inv_q, last_inv_q = inv_stds[:q], inv_stds[-q:]

    first_mean, last_mean = float(first_q.mean()), float(last_q.mean())
    first_std, last_std = float(first_q.std()), float(last_q.std())
    first_inv_mean, last_inv_mean = float(first_inv_q.mean()), float(last_inv_q.mean())

    spread = first_std if first_std > 0 else max(abs(first_mean), 1.0)
    learning_detected = (last_mean - first_mean) > improvement_threshold * spread
    if not learning_detected:
        concerns.append(
            f"no clear improvement from first to last quartile "
            f"(mean {first_mean:.3g} -> {last_mean:.3g})"
        )

    variance_exploding = first_std > 0 and last_std > variance_explosion_ratio * first_std
    if variance_exploding:
        concerns.append(f"reward variance grew substantially (std {first_std:.3g} -> {last_std:.3g})")

    inventory_control_improved = last_inv_mean <= first_inv_mean
    if not inventory_control_improved:
        concerns.append(
            f"inventory std did not improve (mean {first_inv_mean:.3g} -> {last_inv_mean:.3g})"
        )

    return TrainingDiagnosis(
        n_episodes=n,
        first_quartile_reward_mean=first_mean,
        last_quartile_reward_mean=last_mean,
        first_quartile_reward_std=first_std,
        last_quartile_reward_std=last_std,
        first_quartile_inventory_std_mean=first_inv_mean,
        last_quartile_inventory_std_mean=last_inv_mean,
        learning_detected=learning_detected,
        variance_exploding=variance_exploding,
        inventory_control_improved=inventory_control_improved,
        concerns=concerns,
    )
