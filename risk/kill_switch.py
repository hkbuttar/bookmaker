"""Realized-loss kill-switch, consistent with alpha-signal-lab's design
(~/alpha-signal-lab/risk/kill_switch.py): a stateful, streaming monitor of
drawdown from the running equity peak. Once triggered it stays triggered
-- it does not auto-resume just because equity ticks back up -- and only
`reset()`, called deliberately, re-arms it. Treated the same way there and
here: the single most important safety control in the system, and one
that should never silently forgive itself.

One adaptation from alpha-signal-lab: that kill-switch measures *fractional*
drawdown (drawdown / peak), which is meaningful there because the portfolio
starts from a real capital base (e.g. $1M starting cash). bookmaker's
Portfolio starts flat at exactly $0 equity with no capital base to divide
by -- a fractional definition would be unstable (dividing by a peak near
zero) or outright undefined at the start of a session. This kill-switch
instead uses absolute-dollar drawdown from the peak. Same sticky-trigger,
manual-reset-only design; different unit, because the underlying quantity
being protected is structured differently here.
"""

from __future__ import annotations

import math


class KillSwitch:
    def __init__(self, max_drawdown: float) -> None:
        if max_drawdown <= 0:
            raise ValueError("max_drawdown must be positive")
        self.max_drawdown = max_drawdown
        self._peak_equity: float | None = None
        self.triggered = False

    def check(self, equity: float) -> bool:
        """Update with the latest equity value and report kill-switch state.

        Returns True if dollar drawdown from the running peak is at or
        beyond `max_drawdown`. Once triggered, stays triggered regardless
        of subsequent equity values until `reset()` is called.

        A NaN equity (inventory held but no mid-price to mark it against --
        see backtest.portfolio.equity_from) is skipped rather than treated
        as a loss: the peak isn't updated and the existing triggered state
        is returned unchanged, since there's no valid reading to act on.
        """
        if math.isnan(equity):
            return self.triggered

        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity

        drawdown = self._peak_equity - equity
        if drawdown >= self.max_drawdown:
            self.triggered = True
        return self.triggered

    def reset(self) -> None:
        """Manually re-arm the kill-switch after review."""
        self._peak_equity = None
        self.triggered = False
