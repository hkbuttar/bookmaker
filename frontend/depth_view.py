"""Order book depth chart: cumulative bid/ask size vs. price at a chosen
snapshot, with a time slider to scrub through a simulation run's replay.

Bid/ask are two opposing sides of one center (the mid-price), so this uses
the diverging pair (blue<->red, frontend/colors.py) rather than two
arbitrary categorical slots -- color follows the *role* (buy-side vs.
sell-side liquidity), which is fixed and never repainted.
"""

from __future__ import annotations

import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, HoverTool, Select, Slider
from bokeh.plotting import figure

from frontend import colors, data_access


def _post_step_area(pairs: list[tuple[float, float]]) -> tuple[list[float], list[float]]:
    """pairs: [(x0, y0), (x1, y1), ...] ascending x. Returns (xs, ys) for a
    filled post-step area (flat after each point until the next jump),
    closed to a y=0 baseline at both ends -- the standard market-depth
    "staircase" shape, not a smoothly interpolated line.
    """
    if not pairs:
        return [], []
    xs = [pairs[0][0]]
    ys = [0.0]
    for i, (x, y) in enumerate(pairs):
        xs.append(x)
        ys.append(y)
        if i < len(pairs) - 1:
            xs.append(pairs[i + 1][0])
            ys.append(y)
    xs.append(pairs[-1][0])
    ys.append(0.0)
    return xs, ys


def _cumulative_best_first(levels: list[list[float]]) -> list[tuple[float, float]]:
    running = 0.0
    out = []
    for price, size in levels:
        running += size
        out.append((price, running))
    return out


def depth_polygon(levels: list[list[float]], side: str) -> tuple[list[float], list[float]]:
    """side: 'bid' or 'ask'. `levels` is best-first, matching
    OrderBook.depth()'s shape (bids descending, asks ascending).
    """
    pairs = _cumulative_best_first(levels)
    if side == "bid":
        pairs = list(reversed(pairs))  # best-first descending -> ascending price
    return _post_step_area(pairs)


def build_depth_view():
    runs_df = data_access.list_simulation_runs()

    bid_source = ColumnDataSource(data=dict(x=[], y=[]))
    ask_source = ColumnDataSource(data=dict(x=[], y=[]))

    fig = figure(
        title="Order book depth",
        x_axis_label="price",
        y_axis_label="cumulative size",
        height=420,
        width=760,
        background_fill_color=colors.CHART_SURFACE,
        border_fill_color=colors.PAGE_PLANE,
        tools="pan,wheel_zoom,reset,save",
    )
    fig.grid.grid_line_color = colors.GRIDLINE
    fig.axis.axis_line_color = colors.BASELINE
    fig.axis.major_label_text_color = colors.TEXT_MUTED
    fig.title.text_color = colors.TEXT_PRIMARY

    bid_patch = fig.patch("x", "y", source=bid_source, fill_color=colors.BID_COLOR, fill_alpha=0.35,
                           line_color=colors.BID_COLOR, line_width=2, legend_label="Bids")
    ask_patch = fig.patch("x", "y", source=ask_source, fill_color=colors.ASK_COLOR, fill_alpha=0.35,
                           line_color=colors.ASK_COLOR, line_width=2, legend_label="Asks")
    fig.legend.location = "top_left"
    fig.legend.label_text_color = colors.TEXT_SECONDARY
    fig.legend.background_fill_color = colors.CHART_SURFACE

    fig.add_tools(HoverTool(renderers=[bid_patch, ask_patch], tooltips=[("price", "@x{0.00}"), ("cumulative size", "@y{0.0}")]))

    run_select = Select(title="Simulation run", options=[(str(r["id"]), r["label"]) for _, r in runs_df.iterrows()])
    time_slider = Slider(start=0, end=1, value=0, step=1, title="Snapshot index")

    state = {"snapshots": pd.DataFrame()}

    def _load_run(run_id: int) -> None:
        state["snapshots"] = data_access.get_book_snapshots(run_id)
        n = len(state["snapshots"])
        time_slider.end = max(n - 1, 1)
        time_slider.value = 0
        _render(0)

    def _render(index: int) -> None:
        snapshots = state["snapshots"]
        if snapshots.empty or index >= len(snapshots):
            bid_source.data = dict(x=[], y=[])
            ask_source.data = dict(x=[], y=[])
            return
        row_ = snapshots.iloc[index]
        bx, by = depth_polygon(row_["bids"], "bid")
        ax, ay = depth_polygon(row_["asks"], "ask")
        bid_source.data = dict(x=bx, y=by)
        ask_source.data = dict(x=ax, y=ay)
        fig.title.text = f"Order book depth -- t={row_['time']:.3f}s"

    def _on_run_change(attr, old, new):
        if new:
            _load_run(int(new))

    def _on_slider_change(attr, old, new):
        _render(int(new))

    run_select.on_change("value", _on_run_change)
    time_slider.on_change("value", _on_slider_change)

    if len(runs_df):
        run_select.value = str(runs_df.iloc[0]["id"])
        _load_run(int(runs_df.iloc[0]["id"]))

    def refresh() -> None:
        """Re-pull the run list so newly completed runs appear without a
        page reload; leaves the current selection and slider position
        untouched since a completed run's own snapshots never change.
        """
        fresh = data_access.list_simulation_runs()
        current = run_select.value
        run_select.options = [(str(r["id"]), r["label"]) for _, r in fresh.iterrows()]
        if current in [opt for opt, _ in run_select.options]:
            run_select.value = current
        elif len(fresh):
            run_select.value = str(fresh.iloc[0]["id"])

    controls = row(run_select, time_slider)
    return column(controls, fig), refresh
