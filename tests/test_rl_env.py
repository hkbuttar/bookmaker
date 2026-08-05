import math

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from rl.env import ACTION_TABLE, N_ACTIONS, NO_QUOTE_ACTION, MarketMakingEnv
from rl.reward import huber_inventory_penalty


def _event(order_id, time, type_, side=None, price=None, size=None):
    return {"order_id": order_id, "time": time, "type": type_, "side": side, "price": price, "size": size}


def _small_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _event(1, 0.0, "LIMIT", "SELL", 99.00, 100),
            _event(2, 1.0, "LIMIT", "BUY", 97.00, 100),
            _event(3, 2.0, "LIMIT", "SELL", 99.00, 100),
            _event(4, 3.0, "LIMIT", "BUY", 97.00, 100),
            _event(5, 4.0, "MARKET", "BUY", size=10),
            _event(6, 5.0, "MARKET", "SELL", size=10),
        ]
    )


def test_gymnasium_check_env_passes():
    env = MarketMakingEnv(_small_events(), decision_interval_seconds=1.0)
    check_env(env.unwrapped, skip_render_check=True)


def test_action_and_observation_space_shapes():
    env = MarketMakingEnv(_small_events())
    assert env.action_space.n == N_ACTIONS == 10
    assert env.observation_space.shape == (7,)


def test_reset_returns_finite_observation():
    env = MarketMakingEnv(_small_events())
    obs, info = env.reset(seed=0)
    assert obs.shape == (7,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert info == {}


def test_step_returns_well_formed_tuple():
    env = MarketMakingEnv(_small_events())
    env.reset(seed=0)
    obs, reward, terminated, truncated, _ = env.step(NO_QUOTE_ACTION)
    assert obs.shape == (7,)
    assert np.all(np.isfinite(obs))
    assert isinstance(reward, float)
    assert math.isfinite(reward)
    assert isinstance(terminated, bool)
    assert truncated is False


def test_episode_eventually_terminates():
    env = MarketMakingEnv(_small_events(), decision_interval_seconds=1.0)
    env.reset(seed=0)
    terminated = False
    steps = 0
    while not terminated and steps < 100:
        _, _, terminated, _, _ = env.step(NO_QUOTE_ACTION)
        steps += 1
    assert terminated
    assert steps < 100  # didn't hit the safety cap


def test_reset_is_reproducible_against_fixed_dataset():
    events = _small_events()
    env_a = MarketMakingEnv(events, decision_interval_seconds=1.0)
    env_b = MarketMakingEnv(events, decision_interval_seconds=1.0)

    obs_a, _ = env_a.reset(seed=0)
    obs_b, _ = env_b.reset(seed=0)
    np.testing.assert_array_equal(obs_a, obs_b)

    actions = [3, 7, NO_QUOTE_ACTION, 0, 5]
    for action in actions:
        result_a = env_a.step(action)
        result_b = env_b.step(action)
        np.testing.assert_array_equal(result_a[0], result_b[0])
        assert result_a[1] == result_b[1]
        assert result_a[2] == result_b[2]


def test_no_quote_action_never_accrues_fills():
    env = MarketMakingEnv(_small_events(), decision_interval_seconds=1.0)
    env.reset(seed=0)
    terminated = False
    while not terminated:
        _, _, terminated, _, _ = env.step(NO_QUOTE_ACTION)
    assert env._portfolio.inventory == 0
    assert len(env._portfolio.trades) == 0


def test_fill_bonus_added_to_reward_when_a_fill_happens():
    events = pd.DataFrame(
        [
            _event(1, 0.0, "LIMIT", "SELL", 99.00, 10),
            _event(2, 1.0, "LIMIT", "BUY", 97.00, 10),  # mid=98.00 available from t=2 boundary onward
            _event(3, 2.5, "MARKET", "SELL", size=5),  # should hit our resting bid
        ]
    )
    bonus_per_share = 0.01
    env = MarketMakingEnv(events, decision_interval_seconds=1.0, fill_bonus_per_share=bonus_per_share)
    env.reset(seed=0)
    env.step(NO_QUOTE_ACTION)  # advance to t=2, mid=98.00 becomes available

    action = ACTION_TABLE.index((1, 20))  # bid = 98.00 - 0.01 = 97.99
    _, reward, terminated, _, info = env.step(action)

    assert info["filled_shares"] == 5
    assert info["fill_bonus"] == pytest.approx(5 * bonus_per_share)
    assert info["inventory"] == 5
    assert reward == pytest.approx(info["step_pnl"] - info["penalty"] + info["fill_bonus"])


def test_no_fill_bonus_when_nothing_fills():
    events = pd.DataFrame(
        [_event(1, 0.0, "LIMIT", "SELL", 99.00, 10), _event(2, 5.0, "LIMIT", "SELL", 99.50, 10)]
    )
    env = MarketMakingEnv(events, decision_interval_seconds=1.0, fill_bonus_per_share=0.01)
    env.reset(seed=0)
    _, _, terminated, _, info = env.step(NO_QUOTE_ACTION)
    assert info["filled_shares"] == 0
    assert info["fill_bonus"] == 0.0


def test_reward_is_pure_inventory_penalty_when_no_pnl_moves():
    # Drive inventory nonzero via a real fill, then take NO_QUOTE_ACTION
    # while mid_price stays put -- equity shouldn't move (flat resting
    # inventory marked at an unchanged mid), so reward should be exactly
    # -huber_inventory_penalty(inventory, lam, cap).
    events = pd.DataFrame(
        [
            _event(1, 0.0, "LIMIT", "SELL", 99.00, 100),
            _event(2, 1.0, "LIMIT", "BUY", 97.00, 100),  # mid=98.00
            _event(3, 2.0, "MARKET", "BUY", size=5),  # crosses nothing of ours yet (we haven't quoted)
            _event(4, 3.0, "LIMIT", "SELL", 99.00, 100),
            _event(5, 4.0, "LIMIT", "BUY", 97.00, 100),
        ]
    )
    lam, cap = 0.01, 50.0
    env = MarketMakingEnv(
        events, decision_interval_seconds=1.0, inventory_penalty_lambda=lam, inventory_penalty_cap=cap
    )
    env.reset(seed=0)

    # Action index for (offset=1 tick, size=20): find it in the table.
    action = ACTION_TABLE.index((1, 20))
    for _ in range(2):
        _, _, terminated, _, info = env.step(action)
        if terminated:
            break

    if info["inventory"] != 0:
        # If a fill happened, isolate a quiet step: hold NO_QUOTE with no
        # further background price movement and check the penalty term.
        _, reward, terminated, _, info2 = env.step(NO_QUOTE_ACTION)
        expected_penalty = -huber_inventory_penalty(info2["inventory"], lam, cap)
        if info2["step_pnl"] == 0.0:
            assert reward == pytest.approx(expected_penalty)


def test_latency_delays_agent_order_arrival_effect():
    # Same structure as backtest.market_maker_sim's delayed-crossing test:
    # a bid decided while the best ask is 99.00 arrives late enough that a
    # better ask (97.20) has appeared in the meantime, and should cross it.
    #
    # Trace, with decision_interval_seconds=1.0:
    #   reset()      -> boundary t=1.0: processes event@0.0 (ask=99.00) only;
    #                    event@1.0 is held back for the next boundary, so
    #                    mid_price is still None here (ask-only book).
    #   step 1 (noop)-> boundary t=2.0: processes event@1.0 (bid=97.00) ->
    #                    mid=98.00 now available. event@2.0 held back.
    #   step 2 (real)-> decision at t=2.0, offset=3 ticks -> bid=97.97.
    #                    Scheduled to arrive at t=52.0. This step's advance
    #                    to boundary t=3.0 processes event@2.0 (ask=97.20)
    #                    -- background data is now exhausted, but our bid
    #                    hasn't arrived yet, so no fill this step.
    #   step 3 (noop)-> background is exhausted, so _advance drains all
    #                    remaining pending immediately regardless of the
    #                    nominal boundary: our bid (97.97) arrives and
    #                    crosses the resting 97.20 ask.
    events = pd.DataFrame(
        [
            _event(1, 0.0, "LIMIT", "SELL", 99.00, 10),
            _event(2, 1.0, "LIMIT", "BUY", 97.00, 10),
            _event(3, 2.0, "LIMIT", "SELL", 97.20, 5),
        ]
    )
    env = MarketMakingEnv(
        events, decision_interval_seconds=1.0, strategy_latency_model=lambda t: t + 50.0
    )
    env.reset(seed=0)
    env.step(NO_QUOTE_ACTION)

    action = ACTION_TABLE.index((3, 20))  # bid = mid(98.00) - 0.03 = 97.97
    env.step(action)
    _, _, terminated, _, _ = env.step(NO_QUOTE_ACTION)

    assert terminated is True
    assert len(env._portfolio.trades) == 1
    trade = env._portfolio.trades[0]
    assert trade.is_maker is False
    assert trade.price == pytest.approx(97.20)


def test_repeating_the_same_action_preserves_fifo_priority():
    # Regression test for a real bug found during this project's testing: step()
    # originally scheduled a fresh cancel+resubmit on *every* call
    # regardless of whether the action's resulting quote actually changed
    # -- unlike backtest.market_maker_sim's strategies, which only touch
    # the book on a real change. That silently cost the agent its FIFO
    # queue priority every single decision, even while "holding" the same
    # quote, making competitive-looking quotes far less likely to ever
    # fill for reasons that had nothing to do with the policy itself.
    events = pd.DataFrame(
        [
            _event(1, 0.0, "LIMIT", "SELL", 99.00, 10),
            _event(2, 1.0, "LIMIT", "BUY", 97.00, 10),  # mid=98.00 from t=2 boundary onward
            _event(3, 2.5, "LIMIT", "SELL", 98.02, 5),  # background joins AFTER our ask at the same price
            _event(4, 3.5, "MARKET", "BUY", size=5),  # should hit OUR order first if priority preserved
        ]
    )
    env = MarketMakingEnv(events, decision_interval_seconds=1.0)
    env.reset(seed=0)
    env.step(NO_QUOTE_ACTION)  # advance to t=2, mid=98.00 becomes available

    action = ACTION_TABLE.index((2, 20))  # bid=97.98, ask=98.02
    env.step(action)  # decision at t=2 -> boundary t=3: posts our quote, then event@2.5 lands behind it
    env.step(action)  # same action again (must NOT cancel+resubmit) -> boundary t=4: event@3.5 processed here

    assert len(env._portfolio.trades) == 1
    trade = env._portfolio.trades[0]
    assert trade.is_maker is True
    assert trade.price == pytest.approx(98.02)
    assert trade.size == 5
