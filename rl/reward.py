"""Inventory-risk penalty for rl.env's reward.

First cut (see rl/env.py's git history) used a pure quadratic penalty,
`lambda * inventory^2`, straight from the plan's stated reward formula.
That produced rare but extreme penalties whenever exploration drifted into
large inventory (e.g. inventory=500 -> penalty=125 at lambda=5e-4, versus a
typical per-step P&L of a few hundredths of a dollar) -- and DQN's TD
bootstrapping lets a handful of such catastrophic transitions poison the
learned value of *any* state that risks holding inventory, not just the
extreme ones actually visited. Training collapsed to a near-total
"never quote" policy as a result.

This is a Huber-style penalty instead: quadratic (same shape as the
original, disclosed calibration) for |inventory| <= `cap`, transitioning
to linear growth beyond it. Deterrence against large inventory is still
present and still grows without bound -- it just no longer explodes
quadratically, so a single bad exploration episode can't dominate the
value function the way it did before. This is a disclosed reward-shaping
choice made in response to an observed training failure, not a derived
optimum.
"""

from __future__ import annotations


def huber_inventory_penalty(inventory: int, lam: float, cap: float) -> float:
    abs_inv = abs(inventory)
    if abs_inv <= cap:
        return lam * 0.5 * abs_inv**2
    return lam * cap * (abs_inv - 0.5 * cap)
