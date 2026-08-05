"""Strategy comparison: grouped bar chart of a chosen metric across
latency presets, plus the full comparison table (satisfies the palette's
relief rule for the sub-3:1 light-mode slots -- values are always visible
as text, not color-alone).
"""

from __future__ import annotations

from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, DataTable, FactorRange, HoverTool, Select, TableColumn
from bokeh.plotting import figure
from bokeh.transform import factor_cmap

from frontend import colors, data_access

METRICS = [
    ("final_pnl", "Final P&L ($)"),
    ("n_fills", "Number of fills"),
    ("sharpe_ratio", "Sharpe ratio"),
    ("inventory_std", "Inventory std. dev."),
    ("adverse_selection_cost", "Adverse-selection cost ($)"),
]

STRATEGY_ORDER = ["naive", "inventory_aware", "adverse_selection_aware", "rl_latency_naive", "rl_latency_aware"]


def build_comparison_view():
    strategy_palette = [colors.STRATEGY_COLORS[name] for name in STRATEGY_ORDER]

    source = ColumnDataSource(data=dict(x=[], value=[], strategy_name=[], latency_preset=[]))

    fig = figure(
        title="Strategy comparison",
        x_range=FactorRange(),
        height=420,
        width=820,
        background_fill_color=colors.CHART_SURFACE,
        border_fill_color=colors.PAGE_PLANE,
        tools="pan,wheel_zoom,reset,save",
    )
    fig.grid.grid_line_color = colors.GRIDLINE
    fig.xgrid.grid_line_color = None
    fig.axis.axis_line_color = colors.BASELINE
    fig.axis.major_label_text_color = colors.TEXT_MUTED
    fig.axis.major_label_orientation = 1.0
    fig.title.text_color = colors.TEXT_PRIMARY

    bars = fig.vbar(
        x="x", top="value", width=0.85, source=source,
        line_color=colors.CHART_SURFACE, line_width=2,
        fill_color=factor_cmap("strategy_name", palette=strategy_palette, factors=STRATEGY_ORDER),
    )
    fig.add_tools(HoverTool(renderers=[bars], tooltips=[
        ("strategy", "@strategy_name"), ("latency", "@latency_preset"), ("value", "@value{0.00}"),
    ]))

    metric_select = Select(title="Metric", options=[(k, v) for k, v in METRICS], value=METRICS[0][0])

    table_columns = [
        TableColumn(field="strategy_name", title="Strategy"),
        TableColumn(field="latency_preset", title="Latency"),
        TableColumn(field="n_fills", title="Fills"),
        TableColumn(field="final_pnl", title="Final P&L"),
        TableColumn(field="sharpe_ratio", title="Sharpe"),
        TableColumn(field="inventory_std", title="Inventory std"),
        TableColumn(field="adverse_selection_cost", title="Adverse-sel. cost"),
    ]
    table_source = ColumnDataSource(data=dict())
    data_table = DataTable(source=table_source, columns=table_columns, height=260, width=820)

    state = {"df": data_access.get_comparison_table("synthetic")}

    def _render(metric: str) -> None:
        df = state["df"]
        if df.empty:
            source.data = dict(x=[], value=[], strategy_name=[], latency_preset=[])
            table_source.data = dict()
            fig.title.text = "Strategy comparison -- no data yet (run: python3 -m backend.populate)"
            return
        fig.title.text = "Strategy comparison"
        df = df[df["strategy_name"].isin(STRATEGY_ORDER)].copy()
        df["x"] = list(zip(df["latency_preset"], df["strategy_name"]))
        latency_order = sorted(df["latency_preset"].unique(), key=lambda p: int(p.replace("ms", "")))
        factors = [(lat, strat) for lat in latency_order for strat in STRATEGY_ORDER if strat in set(df["strategy_name"])]
        fig.x_range.factors = factors
        fig.yaxis.axis_label = dict(METRICS)[metric]
        source.data = dict(
            x=list(df["x"]), value=list(df[metric]),
            strategy_name=list(df["strategy_name"]), latency_preset=list(df["latency_preset"]),
        )
        table_source.data = df[["strategy_name", "latency_preset", "n_fills", "final_pnl", "sharpe_ratio",
                                 "inventory_std", "adverse_selection_cost"]].to_dict("list")

    def _on_metric_change(attr, old, new):
        _render(new)

    metric_select.on_change("value", _on_metric_change)
    _render(METRICS[0][0])

    def refresh() -> None:
        state["df"] = data_access.get_comparison_table("synthetic")
        _render(metric_select.value)

    return column(row(metric_select), fig, data_table), refresh
