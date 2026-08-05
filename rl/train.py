"""SB3 DQN training for the market-making environment (Step 9).

Trains against one fixed, reproducible synthetic dataset (see rl.env's
docstring for why a fixed dataset per training run, not fresh-generated
per episode). Two variants are trained from the same dataset and the same
initial weights seed, differing only in whether `strategy_latency_model`
is active during training -- "latency-naive" (trained at zero latency) vs
"latency-aware" (trained with the same stochastic latency the environment
will eventually be evaluated under). Comparing how much each degrades when
latency is introduced (or increased) at *evaluation* time, not training
time, is Step 9's self-contained latency-robustness finding; rl/evaluate.py
is where that comparison actually happens.

Hyperparameters below are adapted from SB3's DQN defaults for this
environment's actual scale (7-dim observation, 16 discrete actions,
~600-step episodes at the 10-minute training dataset length) rather than
SB3's out-of-the-box Atari-scale defaults (1M-sample replay buffer,
50k-step warmup, 10k-step target update) -- kept simple and disclosed as
a reasonable starting point given this step's "minutes to low hours on
CPU" time budget, not an extensively tuned configuration.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
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
    (Step 9 explicitly wants inventory variance logged, not just reward)
    need this extra bookkeeping.
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
    decision_interval_seconds: float = 1.0,
) -> TrainingRun:
    env = Monitor(
        MarketMakingEnv(
            events,
            decision_interval_seconds=decision_interval_seconds,
            inventory_penalty_lambda=inventory_penalty_lambda,
            inventory_penalty_cap=inventory_penalty_cap,
            strategy_latency_model=strategy_latency_model,
        )
    )
    model = DQN("MlpPolicy", env, seed=seed, **DQN_HYPERPARAMS)
    callback = EpisodeStatsCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
    return TrainingRun(model=model, callback=callback, label=label)
