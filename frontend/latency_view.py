"""Latency-sensitivity chart: one line per strategy, metric vs. latency
preset. Latency presets are an ordinal sequence (0/5/20/50ms), so this is
a line chart on a categorical-but-ordered x-axis, not a bar chart --
lines make the *degradation trend* legible in a way separate bars don't.
"""

from __future__ import annotations

from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, HoverTool, Select
from bokeh.plotting import figure

from frontend import colors, data_access
from frontend.comparison_view import METRICS, STRATEGY_ORDER


def build_latency_view():
    fig = figure(
        title="Latency sensitivity",
        x_range=["0ms", "5ms", "20ms", "50ms"],
        height=420,
        width=820,
        background_fill_color=colors.CHART_SURFACE,
        border_fill_color=colors.PAGE_PLANE,
        tools="pan,wheel_zoom,reset,save",
    )
    fig.grid.grid_line_color = colors.GRIDLINE
    fig.axis.axis_line_color = colors.BASELINE
    fig.axis.major_label_text_color = colors.TEXT_MUTED
    fig.title.text_color = colors.TEXT_PRIMARY
    fig.xaxis.axis_label = "strategy latency"

    sources = {name: ColumnDataSource(data=dict(x=[], y=[])) for name in STRATEGY_ORDER}
    renderers = []
    for name in STRATEGY_ORDER:
        color = colors.STRATEGY_COLORS[name]
        line = fig.line("x", "y", source=sources[name], line_color=color, line_width=2, legend_label=name)
        fig.scatter("x", "y", source=sources[name], size=8, fill_color=color, line_color=colors.CHART_SURFACE, legend_label=name)
        renderers.append(line)

    fig.add_tools(HoverTool(renderers=renderers, tooltips=[("latency", "@x"), ("value", "@y{0.00}")]))
    fig.legend.location = "top_left"
    fig.legend.label_text_color = colors.TEXT_SECONDARY
    fig.legend.background_fill_color = colors.CHART_SURFACE
    fig.legend.click_policy = "hide"

    metric_select = Select(title="Metric", options=[(k, v) for k, v in METRICS], value=METRICS[0][0])

    state = {"df": data_access.get_comparison_table("synthetic")}

    def _render(metric: str) -> None:
        df = state["df"]
        fig.yaxis.axis_label = dict(METRICS)[metric]
        if df.empty:
            for name in STRATEGY_ORDER:
                sources[name].data = dict(x=[], y=[])
            fig.title.text = "Latency sensitivity -- no data yet (run: python3 -m backend.populate)"
            return
        fig.title.text = "Latency sensitivity"
        latency_order = sorted(df["latency_preset"].unique(), key=lambda p: int(p.replace("ms", "")))
        fig.x_range.factors = latency_order
        for name in STRATEGY_ORDER:
            sub = df[df["strategy_name"] == name].copy()
            sub["_order"] = sub["latency_preset"].map({p: i for i, p in enumerate(latency_order)})
            sub = sub.sort_values("_order")
            sources[name].data = dict(x=list(sub["latency_preset"]), y=list(sub[metric]))

    def _on_metric_change(attr, old, new):
        _render(new)

    metric_select.on_change("value", _on_metric_change)
    _render(METRICS[0][0])

    def refresh() -> None:
        state["df"] = data_access.get_comparison_table("synthetic")
        _render(metric_select.value)

    return column(row(metric_select), fig), refresh
