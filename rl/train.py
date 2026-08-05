"""SB3 DQN training for the market-making environment.

Trains against one fixed, reproducible synthetic dataset (see rl.env's
docstring for why a fixed dataset per training run, not fresh-generated
per episode). Two variants are trained from the same dataset and the same
initial weights seed, differing only in whether `strategy_latency_model`
is active during training -- "latency-naive" (trained at zero latency) vs
"latency-aware" (trained with the same stochastic latency the environment
will eventually be evaluated under). Comparing how much each degrades when
latency is introduced (or increased) at *evaluation* time, not training
time, is the self-contained latency-robustness finding this project wants;
rl/evaluate.py is where that comparison actually happens.

Hyperparameters below are adapted from SB3's DQN defaults for this
environment's actual scale (7-dim observation, 10 discrete actions,
~600-step episodes at the 10-minute training dataset length) rather than
SB3's out-of-the-box Atari-scale defaults (1M-sample replay buffer,
50k-step warmup, 10k-step target update) -- kept simple and disclosed as
a reasonable starting point given the project's "minutes to low hours on
CPU" time budget, not an extensively tuned configuration.

Also enabled by default: an inventory-penalty curriculum
(InventoryPenaltyCurriculumCallback below). Two prior full training runs
(reward reshaping to a Huber-style penalty, then a fill-count bonus, then
a FIFO-priority execution fix -- see rl/reward.py and rl/env.py's
docstrings) each fixed a real, diagnosed problem but the resulting policy
still ended up trading essentially never on held-out evaluation. The
remaining suspected cause: early in training, fills are rare under a
random policy, so there's rarely enough signal to learn "getting filled
is good" before the inventory penalty teaches "holding inventory is bad"
-- the agent can satisfy the second lesson perfectly by never doing
anything that risks the first. The curriculum starts inventory_penalty_lambda
at 0 (pure spread-capture incentive, no risk aversion) and ramps it
linearly up to its target value over the first `curriculum_warmup_fraction`
of training, so the agent has a chance to discover that quoting
competitively is profitable before it's taught to fear the consequences.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from lob.engine import LatencyModel
from rl.env import MarketMakingEnv

# Revised after an initial run (learning_rate=5e-4, exploration_fraction=0.2,
# default max_grad_norm=10, quadratic inventory penalty) collapsed to a
# near-total "never quote" policy: reward variance *increased* over
# training instead of shrinking, driven by rare catastrophic-inventory
# episodes during exploration. Alongside switching to a capped
# (Huber-style) inventory penalty in rl/reward.py, this lowers the
# learning rate and tightens gradient clipping so a handful of extreme
# transitions can't dominate a single update, and extends the exploration
# schedule so the agent spends more of the budget discovering profitable
# quoting before committing to greedy exploitation. Still a disclosed,
# reasonable-effort configuration, not an exhaustively tuned one.
DQN_HYPERPARAMS = dict(
    learning_rate=1e-4,
    buffer_size=50_000,
    learning_starts=2_000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1_000,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    max_grad_norm=1.0,
    policy_kwargs=dict(net_arch=[64, 64]),
    verbose=0,
)


class EpisodeStatsCallback(BaseCallback):
    """Tracks per-episode total reward and inventory std -- SB3's built-in
    Monitor only logs reward/length, not the custom `info["inventory"]`
    this environment reports every step, so training-curve diagnostics
    (this project explicitly wants inventory variance logged, not just
    reward) need this extra bookkeeping.
    """

    def __init__(self) -> None:
        super().__init__()
        self.episode_rewards: list[float] = []
        self.episode_inventory_stds: list[float] = []
        self.episode_final_inventory: list[int] = []
        self._current_rewards: list[float] = []
        self._current_inventories: list[float] = []

    def _on_step(self) -> bool:
        reward = float(self.locals["rewards"][0])
        info = self.locals["infos"][0]
        done = bool(self.locals["dones"][0])

        self._current_rewards.append(reward)
        self._current_inventories.append(info["inventory"])

        if done:
            self.episode_rewards.append(sum(self._current_rewards))
            self.episode_inventory_stds.append(float(np.std(self._current_inventories)))
            self.episode_final_inventory.append(int(self._current_inventories[-1]))
            self._current_rewards = []
            self._current_inventories = []
        return True


class InventoryPenaltyCurriculumCallback(BaseCallback):
    """Ramps the env's inventory_penalty_lambda linearly from 0 up to
    `target_lambda` over the first `warmup_fraction` of training, then
    holds it constant. Uses VecEnv.env_method rather than set_attr:
    set_attr would set a new attribute directly on the Monitor wrapper
    object (shadowing, not reaching, the underlying env), since Python
    attribute *assignment* doesn't forward through gym.Wrapper the way
    attribute *access* does. Calling a method by name does forward
    correctly (Monitor's __getattr__ resolves it to the wrapped env's
    bound method), which is why rl.env.MarketMakingEnv exposes
    set_inventory_penalty_lambda instead of just relying on direct
    attribute access from outside.
    """

    def __init__(self, target_lambda: float, total_timesteps: int, warmup_fraction: float = 0.3) -> None:
        super().__init__()
        self.target_lambda = target_lambda
        self.total_timesteps = total_timesteps
        self.warmup_fraction = warmup_fraction

    def _on_step(self) -> bool:
        warmup_steps = self.warmup_fraction * self.total_timesteps
        progress = min(1.0, self.num_timesteps / warmup_steps) if warmup_steps > 0 else 1.0
        current_lambda = self.target_lambda * progress
        self.training_env.env_method("set_inventory_penalty_lambda", current_lambda)
        return True


@dataclasses.dataclass
class TrainingRun:
    model: DQN
    callback: EpisodeStatsCallback
    label: str


def train_dqn(
    events: pd.DataFrame,
    label: str,
    strategy_latency_model: LatencyModel | None = None,
    total_timesteps: int = 60_000,
    seed: int = 0,
    inventory_penalty_lambda: float = 1e-3,
    inventory_penalty_cap: float = 50.0,
    fill_bonus_per_share: float = 0.01,
    decision_interval_seconds: float = 1.0,
    use_curriculum: bool = True,
    curriculum_warmup_fraction: float = 0.3,
) -> TrainingRun:
    # Curriculum starts the env's own penalty at 0 regardless of the
    # target -- the callback ramps it up from there every step.
    initial_lambda = 0.0 if use_curriculum else inventory_penalty_lambda
    env = Monitor(
        MarketMakingEnv(
            events,
            decision_interval_seconds=decision_interval_seconds,
            inventory_penalty_lambda=initial_lambda,
            inventory_penalty_cap=inventory_penalty_cap,
            fill_bonus_per_share=fill_bonus_per_share,
            strategy_latency_model=strategy_latency_model,
        )
    )
    model = DQN("MlpPolicy", env, seed=seed, **DQN_HYPERPARAMS)

    stats_callback = EpisodeStatsCallback()
    callback = stats_callback
    if use_curriculum:
        curriculum_callback = InventoryPenaltyCurriculumCallback(
            target_lambda=inventory_penalty_lambda,
            total_timesteps=total_timesteps,
            warmup_fraction=curriculum_warmup_fraction,
        )
        callback = CallbackList([stats_callback, curriculum_callback])

    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    return TrainingRun(model=model, callback=stats_callback, label=label)
