import numpy as np
import pytest

from rl.diagnostics import diagnose_training


def _healthy_run(n=200, seed=0):
    rng = np.random.default_rng(seed)
    # Reward climbs from strongly negative toward ~0, variance shrinks;
    # inventory std drops as the (fabricated) policy learns control.
    progress = np.linspace(0, 1, n)
    rewards = -500 * (1 - progress) + rng.normal(0, 20 * (1 - progress) + 2, size=n)
    inventory_stds = 30 * (1 - progress) + rng.normal(0, 1, size=n).clip(min=0)
    return list(rewards), list(inventory_stds)


def _unstable_run(n=200, seed=0):
    rng = np.random.default_rng(seed)
    # Mimics Step 9's actual first failed run: variance grows over
    # training instead of shrinking, no sustained improvement.
    rewards = rng.normal(-300, 200 + 400 * np.linspace(0, 1, n), size=n)
    inventory_stds = rng.normal(20, 5, size=n).clip(min=0)
    return list(rewards), list(inventory_stds)


def _flat_run(n=200):
    return [-10.0] * n, [15.0] * n


def test_healthy_run_is_sane():
    rewards, inv_stds = _healthy_run()
    diagnosis = diagnose_training(rewards, inv_stds)
    assert diagnosis.learning_detected is True
    assert diagnosis.variance_exploding is False
    assert diagnosis.inventory_control_improved is True
    assert diagnosis.is_sane


def test_unstable_run_flags_concerns():
    rewards, inv_stds = _unstable_run()
    diagnosis = diagnose_training(rewards, inv_stds)
    assert diagnosis.variance_exploding is True
    assert not diagnosis.is_sane
    assert any("variance" in c for c in diagnosis.concerns)


def test_flat_run_flags_no_learning():
    rewards, inv_stds = _flat_run()
    diagnosis = diagnose_training(rewards, inv_stds)
    assert diagnosis.learning_detected is False
    assert not diagnosis.is_sane
    assert any("no clear improvement" in c for c in diagnosis.concerns)


def test_nan_in_rewards_is_flagged():
    rewards, inv_stds = _healthy_run()
    rewards[50] = float("nan")
    diagnosis = diagnose_training(rewards, inv_stds)
    assert not diagnosis.is_sane
    assert any("non-finite" in c for c in diagnosis.concerns)


def test_too_few_episodes_is_flagged():
    rewards, inv_stds = _healthy_run(n=5)
    diagnosis = diagnose_training(rewards, inv_stds, min_episodes=20)
    assert any("only 5 episodes" in c for c in diagnosis.concerns)


def test_empty_rewards_raises():
    with pytest.raises(ValueError):
        diagnose_training([], [])


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        diagnose_training([1.0, 2.0], [1.0])


def test_worsening_inventory_control_is_flagged():
    rewards, _ = _healthy_run()
    # Inventory std *increasing* over training despite reward improving --
    # a real, distinct failure mode from reward instability.
    worsening_inv_stds = list(np.linspace(5, 40, len(rewards)))
    diagnosis = diagnose_training(rewards, worsening_inv_stds)
    assert diagnosis.inventory_control_improved is False
    assert any("inventory std did not improve" in c for c in diagnosis.concerns)
