import math

import pytest

from risk.kill_switch import KillSwitch


def test_rejects_nonpositive_max_drawdown():
    with pytest.raises(ValueError):
        KillSwitch(max_drawdown=0.0)


def test_triggers_at_threshold_drawdown():
    switch = KillSwitch(max_drawdown=15.0)
    assert switch.check(100.0) is False
    assert switch.check(90.0) is False  # $10 drawdown, below threshold
    assert switch.check(84.0) is True  # $16 drawdown, breaches threshold


def test_stays_triggered_even_if_equity_recovers():
    switch = KillSwitch(max_drawdown=15.0)
    switch.check(100.0)
    switch.check(80.0)  # triggers
    assert switch.check(200.0) is True  # does not silently re-arm


def test_reset_clears_triggered_state():
    switch = KillSwitch(max_drawdown=15.0)
    switch.check(100.0)
    switch.check(80.0)
    switch.reset()
    assert switch.triggered is False
    assert switch.check(100.0) is False


def test_peak_tracks_new_highs():
    switch = KillSwitch(max_drawdown=15.0)
    switch.check(100.0)
    switch.check(120.0)
    assert switch.check(110.0) is False  # $10 off the new peak of 120, below threshold


def test_starts_from_zero_equity_without_dividing_by_a_capital_base():
    # bookmaker portfolios start flat at $0 -- this must not blow up the
    # way a fractional-drawdown definition would (dividing by ~0).
    switch = KillSwitch(max_drawdown=10.0)
    assert switch.check(0.0) is False
    assert switch.check(-5.0) is False  # $5 drawdown from the $0 peak
    assert switch.check(-11.0) is True  # $11 drawdown breaches the $10 threshold


def test_nan_equity_is_skipped_not_treated_as_a_loss():
    switch = KillSwitch(max_drawdown=15.0)
    switch.check(100.0)
    result = switch.check(float("nan"))
    assert result is False
    assert switch.triggered is False
    # Peak should be unaffected by the NaN reading.
    assert switch.check(90.0) is False  # still only $10 off the real peak of 100


def test_nan_equity_as_first_reading_does_not_corrupt_peak():
    switch = KillSwitch(max_drawdown=15.0)
    switch.check(float("nan"))
    assert switch.check(100.0) is False
    assert switch.check(84.0) is True  # peak correctly tracked as 100, not NaN
