"""Tests for export helpers."""

from __future__ import annotations

import plotly.graph_objects as go
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from src.export_utils import (
    PlotTextBlockLayout,
    add_plot_text_block,
    autoscale_figure_to_data,
    scale_figure_for_export,
    scaled_text_font_size_for_export,
)
from src.mpl_export import (
    _apply_layout,
    _draw_annotations,
    _draw_legend,
    _draw_traces,
    _ensure_export_decorations_fit,
    _soften_grid_style,
)
from src.plotting import figure_to_image_bytes


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


def test_figure_to_image_bytes_exports_scatter_to_all_supported_formats() -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[1, 2, 3],
            y=[2, 4, 3],
            mode="lines+markers",
            error_y={"type": "data", "array": [0.2, 0.3, 0.2], "visible": True},
        )
    )
    fig.update_layout(width=800, height=500, xaxis_title="x", yaxis_title="y")

    png_bytes = figure_to_image_bytes(fig, "png", width=800, height=500, scale=1.0, base_dpi=300)
    svg_bytes = figure_to_image_bytes(fig, "svg", width=800, height=500, scale=1.0, base_dpi=300)
    pdf_bytes = figure_to_image_bytes(fig, "pdf", width=800, height=500, scale=1.0, base_dpi=300)

    assert png_bytes.startswith(bytes([137]) + b"PNG")
    assert svg_bytes.lstrip().startswith(b"<?xml")
    assert pdf_bytes.startswith(b"%PDF")


def test_figure_to_image_bytes_exports_histogram_style_bar_with_annotation() -> None:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[1, 2, 3], y=[10, 7, 4], marker={"color": "#7aa6ff"}))
    fig.add_annotation(
        x=0.98,
        y=0.95,
        xref="paper",
        yref="paper",
        text="&mu; = 3.2<br>&sigma; = 1.1",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.9)",
    )
    fig.update_layout(width=800, height=500, xaxis_title="x", yaxis_title="count")

    for fmt, signature in (("png", bytes([137]) + b"PNG"), ("svg", b"<?xml"), ("pdf", b"%PDF")):
        data = figure_to_image_bytes(fig, fmt, width=800, height=500, scale=1.0, base_dpi=300)
        assert data.lstrip().startswith(signature)


def test_scale_figure_for_export_treats_target_text_size_as_points_for_mpl_exports() -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 4, 3], mode="lines+markers"))
    fig.update_layout(font={"size": 14})

    scaled = scale_figure_for_export(
        fig,
        visual_scale=1.0,
        target_text_pt=14.0,
        base_export_dpi=300,
        text_unit="pt",
    )

    assert float(scaled.layout.font.size) == 14.0


def test_mpl_export_converts_plotly_px_fonts_to_reasonable_point_sizes() -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 4, 3], mode="lines+markers", name="Messwerte"))
    fig.add_annotation(x=2, y=3, text="?y = 2.43", font={"size": 30, "color": "red"}, showarrow=False)
    fig.update_layout(width=2480, height=1754, font={"size": 30}, xaxis_title="m [g]", yaxis_title="T^2 [s^2]")

    scaled = scale_figure_for_export(
        fig,
        visual_scale=1.0,
        target_text_pt=14.0,
        base_export_dpi=300,
        text_unit="pt",
    )

    mpl_figure = Figure(figsize=(2480 / 300, 1754 / 300), dpi=300)
    ax = mpl_figure.add_subplot(111)
    _apply_layout(scaled, mpl_figure, ax)
    _draw_traces(ax, scaled)
    _draw_annotations(mpl_figure, ax, scaled)
    _draw_legend(ax, scaled)

    legend = ax.get_legend()
    assert legend is not None
    legend_sizes = [text.get_fontsize() for text in legend.get_texts()]
    annotation_sizes = [text.get_fontsize() for text in ax.texts]

    assert max(legend_sizes) <= 12.0
    assert max(annotation_sizes) <= 11.0


def test_scale_figure_for_export_visual_scale_does_not_inflate_text_target() -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 4, 3], mode="lines+markers"))
    fig.update_layout(font={"size": 30})

    scaled = scale_figure_for_export(
        fig,
        visual_scale=3.0,
        target_text_pt=14.0,
        base_export_dpi=300,
        text_unit="pt",
    )

    assert float(scaled.layout.font.size) == 14.0


def test_mpl_apply_layout_preserves_major_and_minor_grid_locators() -> None:
    fig = go.Figure()
    fig.update_layout(width=800, height=500)
    fig.update_xaxes(
        showgrid=True,
        dtick=10,
        tick0=5,
        minor={"showgrid": True, "dtick": 2, "tick0": 1, "gridcolor": "rgba(120,120,120,0.16)"},
    )
    fig.update_yaxes(
        showgrid=True,
        dtick=20,
        tick0=10,
        minor={"showgrid": True, "dtick": 4, "tick0": 2, "gridcolor": "rgba(120,120,120,0.16)"},
    )

    mpl_figure = Figure(figsize=(8, 5), dpi=100)
    ax = mpl_figure.add_subplot(111)
    _apply_layout(fig, mpl_figure, ax)

    assert isinstance(ax.xaxis.get_major_locator(), MultipleLocator)
    assert isinstance(ax.xaxis.get_minor_locator(), MultipleLocator)
    assert isinstance(ax.yaxis.get_major_locator(), MultipleLocator)
    assert isinstance(ax.yaxis.get_minor_locator(), MultipleLocator)

    x_major = ax.xaxis.get_major_locator().tick_values(0, 30)
    x_minor = ax.xaxis.get_minor_locator().tick_values(0, 10)
    y_major = ax.yaxis.get_major_locator().tick_values(0, 50)
    y_minor = ax.yaxis.get_minor_locator().tick_values(0, 12)

    assert 5.0 in x_major
    assert 1.0 in x_minor
    assert 10.0 in y_major
    assert 2.0 in y_minor
    assert bool(ax.xaxis._minor_tick_kw.get("gridOn"))
    assert bool(ax.yaxis._minor_tick_kw.get("gridOn"))



def test_mpl_apply_layout_supports_log_x_axis() -> None:
    fig = go.Figure()
    fig.update_layout(width=800, height=500)
    fig.update_xaxes(type="log", range=[0, 2], title_text="x")
    fig.update_yaxes(title_text="y")

    mpl_figure = Figure(figsize=(8, 5), dpi=100)
    ax = mpl_figure.add_subplot(111)
    _apply_layout(fig, mpl_figure, ax)

    assert ax.get_xscale() == "log"
    x_limits = ax.get_xlim()
    assert x_limits[0] == 1.0
    assert x_limits[1] == 100.0


def test_mpl_draw_legend_respects_horizontal_layout_and_stays_inside_figure() -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2], mode="lines", name="A"))
    fig.add_trace(go.Scatter(x=[1, 2], y=[2, 3], mode="lines", name="B"))
    fig.update_layout(
        width=800,
        height=500,
        title={"text": "Export title", "font": {"size": 22}},
        legend={
            "orientation": "h",
            "x": 0.5,
            "y": 1.02,
            "xanchor": "center",
            "yanchor": "bottom",
        },
    )

    mpl_figure = Figure(figsize=(8, 5), dpi=100)
    ax = mpl_figure.add_subplot(111)
    _apply_layout(fig, mpl_figure, ax)
    _draw_traces(ax, fig)
    _draw_legend(ax, fig)

    legend = ax.get_legend()
    assert legend is not None
    assert getattr(legend, "_ncols", 1) == 2
    canvas = FigureCanvasAgg(mpl_figure)
    canvas.draw()
    bbox = legend.get_window_extent(renderer=canvas.get_renderer()).transformed(mpl_figure.transFigure.inverted())
    assert bbox.x0 >= -0.001
    assert bbox.x1 <= 1.001
    assert bbox.y0 >= -0.001
    assert bbox.y1 <= 1.001
    assert ax.get_title() == "Export title"


def test_figure_to_image_bytes_handles_shapes_and_both_error_axes() -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[1, 2, 3],
            y=[2, 4, 3],
            mode="markers",
            error_x={"type": "data", "array": [0.1, 0.1, 0.2], "visible": True},
            error_y={"type": "data", "array": [0.2, 0.3, 0.2], "visible": True},
            name="Messwerte",
        )
    )
    fig.add_shape(type="rect", x0=1.2, x1=2.6, y0=2.1, y1=3.8, line={"color": "royalblue"}, fillcolor="rgba(65,105,225,0.12)")
    fig.add_shape(type="circle", x0=2.4, x1=2.9, y0=2.6, y1=3.2, line={"color": "firebrick"}, fillcolor="rgba(178,34,34,0.10)")
    fig.update_layout(width=800, height=500)

    for fmt, signature in (("png", bytes([137]) + b"PNG"), ("svg", b"<?xml"), ("pdf", b"%PDF")):
        data = figure_to_image_bytes(fig, fmt, width=800, height=500, scale=1.0, base_dpi=300)
        assert data.lstrip().startswith(signature)



def test_export_fit_keeps_y_axis_label_inside_figure() -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[50, 100, 150], y=[0.84, 1.46, 2.16], mode="lines+markers", name="Messwerte"))
    fig.update_layout(width=1200, height=800, xaxis_title="m [g]", yaxis_title="T", margin={"l": 60, "r": 30, "t": 60, "b": 60})

    mpl_figure = Figure(figsize=(1200 / 300, 800 / 300), dpi=300)
    ax = mpl_figure.add_subplot(111)
    _apply_layout(fig, mpl_figure, ax)
    _draw_traces(ax, fig)
    _ensure_export_decorations_fit(mpl_figure, ax)

    canvas = FigureCanvasAgg(mpl_figure)
    canvas.draw()
    bbox = ax.yaxis.get_label().get_window_extent(renderer=canvas.get_renderer()).transformed(mpl_figure.transFigure.inverted())

    assert bbox.x0 >= -0.001
    assert bbox.x1 <= 1.001



def test_soften_grid_style_lightens_export_gridlines() -> None:
    color, width = _soften_grid_style((120 / 255.0, 120 / 255.0, 120 / 255.0, 0.32), 0.8, minor=False)
    minor_color, minor_width = _soften_grid_style((120 / 255.0, 120 / 255.0, 120 / 255.0, 0.16), 0.6, minor=True)

    assert color[0] > (120 / 255.0)
    assert color[1] > (120 / 255.0)
    assert color[2] > (120 / 255.0)
    assert color[3] < 0.32
    assert width < 0.8

    assert minor_color[3] < 0.16
    assert minor_width < 0.6


def test_autoscale_figure_to_data_includes_visible_trace_shape_and_annotation_extents() -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0.0, 1.0], y=[1.0, 2.0], mode="markers", name="data"))
    fig.add_trace(go.Scatter(x=[-5.0, 5.0], y=[0.5, 10.0], mode="lines", name="fit"))
    fig.add_shape(type="line", xref="x", yref="y", x0=-6.0, x1=6.0, y0=-1.0, y1=11.0)
    fig.add_annotation(x=6.5, y=11.5, xref="x", yref="y", text="edge", showarrow=False)

    autoscaled = autoscale_figure_to_data(
        fig,
        x_values=[0.0, 1.0],
        y_values=[1.0, 2.0],
        sigma_y_values=[0.0, 0.0],
        y_axis_type="linear",
    )

    x_range = list(autoscaled.layout.xaxis.range)
    y_range = list(autoscaled.layout.yaxis.range)
    assert x_range[0] <= -6.0
    assert x_range[1] >= 6.5
    assert y_range[0] <= -1.0
    assert y_range[1] >= 11.5


def test_autoscale_figure_to_data_snaps_linear_range_to_major_tick_grid() -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0.0, 10.0], y=[1.0, 2.0], mode="markers", name="data"))
    fig.update_xaxes(dtick=0.2, tick0=0.0)

    autoscaled = autoscale_figure_to_data(
        fig,
        x_values=[0.0, 10.0],
        y_values=[1.0, 2.0],
        sigma_y_values=[0.0, 0.0],
        y_axis_type="linear",
    )

    x_range = list(autoscaled.layout.xaxis.range)
    assert x_range[0] == pytest.approx(-0.6)
    assert x_range[1] == pytest.approx(10.6)


def test_autoscale_figure_to_data_keeps_linear_bar_charts_at_zero_baseline() -> None:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[1.0, 2.0, 3.0], y=[1.0, 2.0, 3.0], name="Histogram"))
    fig.add_trace(go.Scatter(x=[1.0, 2.0, 3.0], y=[0.4, 2.6, 0.5], mode="lines", name="Fit"))

    autoscaled = autoscale_figure_to_data(
        fig,
        x_values=[],
        y_values=[],
        sigma_y_values=[],
        y_axis_type="linear",
    )

    y_range = list(autoscaled.layout.yaxis.range)
    assert y_range[0] == pytest.approx(0.0)
    assert y_range[1] > 3.0


def test_autoscale_figure_to_data_preserves_explicit_x_and_y_ranges_when_requested() -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0.0, 10.0], y=[1.0, 2.0], mode="markers", name="data"))
    fig.update_xaxes(range=[2.0, 8.0])
    fig.update_yaxes(range=[0.5, 2.5])

    autoscaled = autoscale_figure_to_data(
        fig,
        x_values=[0.0, 10.0],
        y_values=[1.0, 2.0],
        sigma_y_values=[0.0, 0.0],
        y_axis_type="linear",
        preserve_x_range=True,
        preserve_y_range=True,
    )

    assert list(autoscaled.layout.xaxis.range) == [2.0, 8.0]
    assert list(autoscaled.layout.yaxis.range) == [0.5, 2.5]
