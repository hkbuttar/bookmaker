"""Bokeh server app entry point: tabbed dashboard over the backend's own
database. Run with `bokeh serve frontend` (needs a package __init__.py,
see frontend/__init__.py) from the repo root.
"""

from __future__ import annotations

import os
import sys

# bokeh serve execs this file with only frontend/ on sys.path, not the
# repo root -- add the root so `frontend.*` and `backend.*` both import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bokeh.io import curdoc
from bokeh.models import TabPanel, Tabs

from frontend import colors
from frontend.comparison_view import build_comparison_view
from frontend.depth_view import build_depth_view
from frontend.latency_view import build_latency_view
from frontend.training_view import build_training_view

REFRESH_INTERVAL_MS = 5_000


def build_app():
    depth_layout, depth_refresh = build_depth_view()
    comparison_layout, comparison_refresh = build_comparison_view()
    latency_layout, latency_refresh = build_latency_view()
    training_layout, training_refresh = build_training_view()

    depth_panel = TabPanel(child=depth_layout, title="Order book depth")
    comparison_panel = TabPanel(child=comparison_layout, title="Strategy comparison")
    latency_panel = TabPanel(child=latency_layout, title="Latency sensitivity")
    training_panel = TabPanel(child=training_layout, title="RL training")

    tabs = Tabs(tabs=[depth_panel, comparison_panel, latency_panel, training_panel])
    tabs.stylesheets = [f":host {{ background-color: {colors.PAGE_PLANE}; }}"]

    def refresh_all():
        depth_refresh()
        comparison_refresh()
        latency_refresh()
        training_refresh()

    curdoc().add_periodic_callback(refresh_all, REFRESH_INTERVAL_MS)
    return tabs


curdoc().add_root(build_app())
curdoc().title = "BookMaker"
