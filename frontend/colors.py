"""Validated palette constants (dataviz skill's reference instance) --
color assignment follows the skill's rules: categorical hues assigned by
series *identity* in fixed order (never cycled, never repainted when a
filter changes which series are visible), sequential/diverging used only
for genuinely ordered/polar data, status colors reserved and never reused
for a series.
"""

from __future__ import annotations

# Categorical, fixed order -- validated (adjacent-pair CVD/normal-vision
# floors clear in both light and dark) via
# dataviz/scripts/validate_palette.js. Never reorder per-chart; a series
# always gets the same slot everywhere it appears.
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Fixed strategy -> slot assignment. Assigned once, referenced everywhere
# (comparison bars, latency lines, depth-chart highlight) so the same
# strategy is always the same color across every view.
STRATEGY_COLORS = {
    "naive": CATEGORICAL[0],
    "inventory_aware": CATEGORICAL[1],
    "adverse_selection_aware": CATEGORICAL[2],
    "rl_latency_naive": CATEGORICAL[3],
    "rl_latency_aware": CATEGORICAL[4],
}

# Order book depth chart: bid/ask read as two opposing sides of one
# center (mid-price), so this uses the diverging pair (blue<->red), not
# two arbitrary categorical slots.
BID_COLOR = "#2a78d6"
ASK_COLOR = "#e34948"

# Chart chrome.
CHART_SURFACE = "#fcfcfb"
PAGE_PLANE = "#f9f9f7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
