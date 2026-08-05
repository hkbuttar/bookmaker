import pytest

from rl.reward import huber_inventory_penalty


def test_quadratic_below_cap():
    assert huber_inventory_penalty(30, lam=0.001, cap=50) == pytest.approx(0.001 * 0.5 * 900)


def test_matches_quadratic_exactly_at_cap():
    lam, cap = 0.001, 50
    quadratic_at_cap = lam * 0.5 * cap**2
    assert huber_inventory_penalty(cap, lam, cap) == pytest.approx(quadratic_at_cap)


def test_linear_beyond_cap_continuous_with_quadratic_branch():
    lam, cap = 0.001, 50
    at_cap = huber_inventory_penalty(cap, lam, cap)
    just_beyond = huber_inventory_penalty(cap + 1, lam, cap)
    # Should increase smoothly, not jump discontinuously.
    assert just_beyond > at_cap
    assert just_beyond - at_cap == pytest.approx(lam * cap, abs=1e-9)


def test_linear_growth_is_much_softer_than_quadratic_far_beyond_cap():
    lam, cap = 0.001, 50
    huber_penalty = huber_inventory_penalty(500, lam, cap)
    quadratic_penalty = lam * 500**2
    assert huber_penalty < quadratic_penalty / 5  # dramatically softer at extreme inventory


def test_symmetric_in_sign():
    assert huber_inventory_penalty(80, 0.001, 50) == huber_inventory_penalty(-80, 0.001, 50)


def test_zero_inventory_zero_penalty():
    assert huber_inventory_penalty(0, 0.001, 50) == 0.0
