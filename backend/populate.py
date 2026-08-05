"""One-time population script for the tables /strategies/compare and
/rl/training serve -- both report precomputed results, not something
computed live on request (a full sweep takes minutes; training takes
longer), so this script is what actually produces the numbers those
endpoints read back out.

Run directly: `python3 -m backend.populate`. Idempotent in the sense that
re-running adds fresh rows rather than erroring, but does not deduplicate
-- clear the tables first (or point DATABASE_URL at a fresh database) for
a clean rebuild.
"""

from __future__ import annotations

import numpy as np

from backend.db import SessionLocal, init_db
from backend.db_models import ComparisonResult, TrainingEpisode, TrainingRunRecord
from backtest.market_maker_sim import run_backtest
from backtest.metrics import summarize
from data.synthetic_lob import SyntheticLOBConfig, generate_session
from lob.latency import make_latency_model
from rl.evaluate import evaluate_policy
from rl.train import train_dqn
from strategies.adverse_selection_aware import AdverseSelectionAwareStrategy
from strategies.inventory_aware import InventoryAwareStrategy
from strategies.naive import NaiveSymmetricStrategy

TRAIN_SESSION_SECONDS = 600.0
TRAIN_SEED = 100
EVAL_SESSION_SECONDS = 600.0
EVAL_SEED = 500
TICK_SIZE = 0.01
LATENCY_PRESETS = ("0ms", "5ms", "20ms", "50ms")


def make_baselines():
    return {
        "naive": NaiveSymmetricStrategy(half_spread=0.02, quote_size=10, tick_size=TICK_SIZE),
        "inventory_aware": InventoryAwareStrategy(
            half_spread=0.02, quote_size=10, inventory_penalty=0.0005, tick_size=TICK_SIZE
        ),
        "adverse_selection_aware": AdverseSelectionAwareStrategy(
            half_spread=0.02, quote_size=10, imbalance_ema_alpha=0.05,
            imbalance_threshold=0.3, widen_multiplier=2.5, pull_threshold=0.7, tick_size=TICK_SIZE,
        ),
    }


TRAIN_TOTAL_TIMESTEPS = 300_000


def train_final_policies():
    """Reproduces the evidence-based configuration decided in the research
    notebook: latency-naive trained without the fill-count bonus,
    latency-aware trained with it (see notebooks/research.ipynb section 5c
    for the ablation that justifies this per-variant choice).
    """
    train_events = generate_session(SyntheticLOBConfig(session_seconds=TRAIN_SESSION_SECONDS, seed=TRAIN_SEED))

    naive_run = train_dqn(
        train_events, label="latency_naive", strategy_latency_model=None,
        total_timesteps=TRAIN_TOTAL_TIMESTEPS, seed=0, fill_bonus_per_share=0.0,
    )
    rng = np.random.default_rng(7)
    aware_run = train_dqn(
        train_events, label="latency_aware", strategy_latency_model=make_latency_model("20ms", rng=rng),
        total_timesteps=TRAIN_TOTAL_TIMESTEPS, seed=0, fill_bonus_per_share=0.01,
    )
    return naive_run, aware_run


def populate_training_tables(db, naive_run, aware_run) -> None:
    for run in (naive_run, aware_run):
        record = TrainingRunRecord(label=run.label, total_timesteps=TRAIN_TOTAL_TIMESTEPS)
        db.add(record)
        db.flush()  # assign record.id
        for i, (reward, inv_std) in enumerate(
            zip(run.callback.episode_rewards, run.callback.episode_inventory_stds)
        ):
            db.add(TrainingEpisode(training_run_id=record.id, episode_index=i, reward=reward, inventory_std=inv_std))
    db.commit()


def populate_comparison_table(db, naive_run, aware_run) -> None:
    eval_events = generate_session(SyntheticLOBConfig(session_seconds=EVAL_SESSION_SECONDS, seed=EVAL_SEED))

    for preset in LATENCY_PRESETS:
        rng = np.random.default_rng(123)
        latency_model = None if preset == "0ms" else make_latency_model(preset, rng=rng)

        for name, strat in make_baselines().items():
            result = run_backtest(
                eval_events, strat, tick_size=TICK_SIZE, imbalance_levels=5, record_levels=5,
                strategy_latency_model=latency_model, decision_interval_seconds=1.0,
            )
            _add_comparison_row(db, name, preset, summarize(result))

        for name, run in (("rl_latency_naive", naive_run), ("rl_latency_aware", aware_run)):
            result = evaluate_policy(
                run.model, eval_events, session_seconds=EVAL_SESSION_SECONDS, decision_interval_seconds=1.0,
                tick_size=TICK_SIZE, imbalance_levels=5, record_levels=5, strategy_latency_model=latency_model,
            )
            _add_comparison_row(db, name, preset, summarize(result))

    db.commit()


def _add_comparison_row(db, strategy_name: str, latency_preset: str, stats: dict) -> None:
    db.add(
        ComparisonResult(
            strategy_name=strategy_name,
            latency_preset=latency_preset,
            data_source="synthetic",
            n_fills=stats["n_fills"],
            final_pnl=stats["final_pnl"],
            sharpe_ratio=stats["sharpe_ratio"],
            inventory_std=stats["inventory_std"],
            adverse_selection_cost=stats["adverse_selection_cost"],
        )
    )


def main() -> None:
    init_db()
    db = SessionLocal()
    try:
        print("Training final RL policies (this takes a few minutes)...")
        naive_run, aware_run = train_final_policies()

        print("Populating training curve tables...")
        populate_training_tables(db, naive_run, aware_run)

        print("Running strategy comparison sweep...")
        populate_comparison_table(db, naive_run, aware_run)

        print("Done.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
