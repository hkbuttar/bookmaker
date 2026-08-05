import pandas as pd
import pytest
from stable_baselines3 import DQN

from data.synthetic_lob import SyntheticLOBConfig, generate_session
from rl.env import ACTION_TABLE, NO_QUOTE_ACTION, MarketMakingEnv
from rl.evaluate import RLStrategyAdapter, evaluate_policy
from strategies.base import MarketState, Quote


class _FixedActionModel:
    """Stand-in for a trained SB3 model that always predicts one action --
    isolates the adapter's observation-building/action-decoding plumbing
    from needing an actual trained (or even real) model.
    """

    def __init__(self, action: int) -> None:
        self.action = action

    def predict(self, obs, deterministic=True):
        return self.action, None


def _state(**overrides):
    defaults = dict(
        time=0.0, best_bid=99.97, best_ask=100.03, mid_price=100.00,
        spread=0.06, imbalance=0.0, inventory=0, cash=0.0,
    )
    defaults.update(overrides)
    return MarketState(**defaults)


def test_adapter_decodes_fixed_action_to_expected_quote():
    model = _FixedActionModel(ACTION_TABLE.index((3, 10)))
    adapter = RLStrategyAdapter(model, session_seconds=600.0, tick_size=0.01)
    quote = adapter.quote(_state())
    assert quote.bid_price == pytest.approx(99.97)
    assert quote.ask_price == pytest.approx(100.03)
    assert quote.bid_size == 10
    assert quote.ask_size == 10


def test_adapter_no_quote_action_returns_quote_none():
    model = _FixedActionModel(NO_QUOTE_ACTION)
    adapter = RLStrategyAdapter(model, session_seconds=600.0)
    assert adapter.quote(_state()) == Quote.none()


def test_adapter_returns_quote_none_when_mid_unavailable():
    model = _FixedActionModel(ACTION_TABLE.index((3, 10)))
    adapter = RLStrategyAdapter(model, session_seconds=600.0)
    state = _state(best_bid=None, best_ask=None, mid_price=None, spread=None, imbalance=None)
    assert adapter.quote(state) == Quote.none()


def test_adapter_tracks_mid_history_across_calls():
    model = _FixedActionModel(NO_QUOTE_ACTION)
    adapter = RLStrategyAdapter(model, session_seconds=600.0)
    adapter.quote(_state(mid_price=100.00))
    adapter.quote(_state(mid_price=100.05))
    assert list(adapter._mid_history) == [100.00, 100.05]
    assert adapter._initial_mid_price == 100.00


def test_evaluate_policy_end_to_end_smoke():
    events = generate_session(SyntheticLOBConfig(session_seconds=60.0, seed=5))
    env = MarketMakingEnv(events, decision_interval_seconds=1.0)
    model = DQN("MlpPolicy", env, seed=0, verbose=0, buffer_size=1000, learning_starts=100)

    result = evaluate_policy(model, events, session_seconds=60.0, decision_interval_seconds=1.0)

    assert isinstance(result.portfolio_history, pd.DataFrame)
    assert len(result.portfolio_history) > 0
    assert len(result.book_snapshots) == len(result.portfolio_history)
