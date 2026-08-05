"""RL training diagnostics: episode reward and inventory std. dev. over
the course of training, one line per trained policy label.
"""

from __future__ import annotations

from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure

from frontend import colors, data_access

LABEL_COLORS = {
    "latency_naive": colors.STRATEGY_COLORS["rl_latency_naive"],
    "latency_aware": colors.STRATEGY_COLORS["rl_latency_aware"],
}


def _rolling_mean(values: list[float], window: int = 20) -> list[float]:
    out = []
    running = []
    for v in values:
        running.append(v)
        if len(running) > window:
            running.pop(0)
        out.append(sum(running) / len(running))
    return out


def _build_metric_figure(title: str, y_label: str):
    fig = figure(
        title=title,
        x_axis_label="episode",
        y_axis_label=y_label,
        height=340,
        width=820,
        background_fill_color=colors.CHART_SURFACE,
        border_fill_color=colors.PAGE_PLANE,
        tools="pan,wheel_zoom,reset,save",
    )
    fig.grid.grid_line_color = colors.GRIDLINE
    fig.axis.axis_line_color = colors.BASELINE
    fig.axis.major_label_text_color = colors.TEXT_MUTED
    fig.title.text_color = colors.TEXT_PRIMARY
    return fig


def build_training_view():
    reward_fig = _build_metric_figure("RL training reward (20-episode rolling mean)", "reward")
    inventory_fig = _build_metric_figure("RL training inventory std. dev. (20-episode rolling mean)", "inventory std")

    training_runs = data_access.get_training_runs()

    reward_renderers = []
    inventory_renderers = []
    sources: dict[str, ColumnDataSource] = {}
    for label, df in training_runs.items():
        color = LABEL_COLORS.get(label, colors.CATEGORICAL[-1])
        source = ColumnDataSource(data=dict(
            x=list(df["episode_index"]),
            reward=_rolling_mean(list(df["reward"])),
            inventory_std=_rolling_mean(list(df["inventory_std"])),
        ))
        sources[label] = source
        r1 = reward_fig.line("x", "reward", source=source, line_color=color, line_width=2, legend_label=label)
        r2 = inventory_fig.line("x", "inventory_std", source=source, line_color=color, line_width=2, legend_label=label)
        reward_renderers.append(r1)
        inventory_renderers.append(r2)

    for fig, renderers, field, label in (
        (reward_fig, reward_renderers, "reward", "reward"),
        (inventory_fig, inventory_renderers, "inventory_std", "inventory std"),
    ):
        if renderers:
            fig.add_tools(HoverTool(renderers=renderers, tooltips=[("episode", "@x"), (label, f"@{field}{{0.00}}")]))
        fig.legend.location = "top_left"
        fig.legend.label_text_color = colors.TEXT_SECONDARY
        fig.legend.background_fill_color = colors.CHART_SURFACE

    def refresh() -> None:
        """Updates data for labels already plotted at build time; a label
        trained for the first time after the server started won't appear
        until restart -- new episodes for existing labels do refresh.
        """
        fresh = data_access.get_training_runs()
        for label, source in sources.items():
            df = fresh.get(label)
            if df is None or df.empty:
                continue
            source.data = dict(
                x=list(df["episode_index"]),
                reward=_rolling_mean(list(df["reward"])),
                inventory_std=_rolling_mean(list(df["inventory_std"])),
            )

    return column(reward_fig, inventory_fig), refresh
