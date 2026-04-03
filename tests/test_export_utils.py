"""Tests for export helpers."""

from __future__ import annotations

import plotly.graph_objects as go

from src.export_utils import (
    PlotTextBlockLayout,
    add_plot_text_block,
    scale_figure_for_export,
    scaled_text_font_size_for_export,
)


def test_scale_figure_for_export_handles_bar_traces_without_marker_size() -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[1, 2, 3],
            y=[2, 4, 3],
            marker={"color": "#7aa6ff", "line": {"color": "#7aa6ff", "width": 1.0}},
        )
    )
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3], mode="lines+markers"))

    scaled = scale_figure_for_export(fig, visual_scale=1.2, target_text_pt=14.0, base_export_dpi=300)

    assert isinstance(scaled, go.Figure)
    assert len(scaled.data) == 2


def test_add_plot_text_block_never_overscales_requested_font() -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 2], mode="lines"))
    fig.update_layout(width=900, height=600)

    add_plot_text_block(
        fig,
        ["Short title", "Compact formula"],
        font_size=13,
    )

    annotation = fig.layout.annotations[-1]
    assert annotation.font.size <= 13


def test_add_plot_text_block_respects_manual_position_and_requested_font() -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 2], mode="lines"))
    fig.update_layout(width=900, height=600)

    add_plot_text_block(
        fig,
        ["Manual title", "Manual formula"],
        font_size=15,
        layout=PlotTextBlockLayout(manual=True, x=0.30, y=0.82, max_width=0.25, max_height=0.18),
    )

    annotation = fig.layout.annotations[-1]
    assert annotation.font.size <= 15
    assert annotation.x == 0.30
    assert annotation.y == 0.82


def test_scaled_text_font_size_for_export_scales_with_figure_font_ratio() -> None:
    scaled = scaled_text_font_size_for_export(
        requested_font_size=13,
        preview_base_font_size=14,
        export_base_font_size=56,
    )

    assert scaled == 52
