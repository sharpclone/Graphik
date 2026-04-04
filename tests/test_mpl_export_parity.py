from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from src.errors import ExportValidationError
from src.export_utils import scale_figure_for_export
from src.mpl_export import render_plotly_figure_to_matplotlib
from src.plotting import PlotStyle, create_base_figure, figure_to_image_bytes


def _assert_text_and_legend_inside_canvas(mpl_figure, ax) -> None:
    canvas = FigureCanvasAgg(mpl_figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    artists = [ax.xaxis.label, ax.yaxis.label, ax.title, *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.texts]
    legend = ax.get_legend()
    if legend is not None:
        artists.append(legend)
    for artist in artists:
        if hasattr(artist, "get_visible") and not artist.get_visible():
            continue
        if hasattr(artist, "get_text") and not str(artist.get_text() or "").strip():
            continue
        bbox = artist.get_window_extent(renderer=renderer).transformed(mpl_figure.transFigure.inverted())
        assert bbox.x0 >= -0.005
        assert bbox.x1 <= 1.005
        assert bbox.y0 >= -0.005
        assert bbox.y1 <= 1.005


def _assert_legend_texts_do_not_overlap(mpl_figure, ax) -> None:
    canvas = FigureCanvasAgg(mpl_figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    legend = ax.get_legend()
    assert legend is not None
    text_boxes = []
    for text in legend.get_texts():
        bbox = text.get_window_extent(renderer=renderer).transformed(mpl_figure.transFigure.inverted())
        text_boxes.append(bbox)
    for idx, first in enumerate(text_boxes):
        for second in text_boxes[idx + 1 :]:
            x_overlap = min(first.x1, second.x1) - max(first.x0, second.x0)
            y_overlap = min(first.y1, second.y1) - max(first.y0, second.y0)
            assert not (x_overlap > 0.002 and y_overlap > 0.002)


def _normal_preview(log_y: bool = False) -> go.Figure:
    df = pd.DataFrame(
        {
            "x": [50.0, 100.0, 150.0, 200.0],
            "y": [0.846, 1.4641, 2.1609, 2.7556],
            "sigma_y": [0.0129, 0.0242, 0.0176, 0.0332],
        }
    )
    style = PlotStyle(
        x_label="m [g]",
        y_label="T^2 [s^2]",
        show_grid=True,
        y_axis_type="log" if log_y else "linear",
        y_log_decades=1 if log_y else None,
        y_major_dtick="D1" if log_y else 0.5,
        y_minor_dtick="D1" if log_y else 0.1,
        show_minor_grid=True,
        connect_points=False,
        base_font_size=14,
        axis_title_font_size=16,
        tick_font_size=12,
    )
    fig = create_base_figure(df, style)
    fig.add_trace(go.Scatter(x=[50.0, 200.0], y=[0.82, 2.78], mode="lines", name="Fit"))
    fig.add_annotation(x=125.0, y=2.0, text="fit", showarrow=False)
    return fig


def test_preview_export_parity_invariants_for_normal_plot() -> None:
    fig = _normal_preview(log_y=False)

    mpl_figure, ax, summary = render_plotly_figure_to_matplotlib(fig, width=1000, height=640, base_dpi=200)

    assert summary.expected_trace_count == 2
    assert summary.rendered_trace_count == 2
    assert summary.expected_annotation_count == 1
    assert summary.rendered_annotation_count == 1
    assert summary.legend_present is True
    assert summary.xaxis_title == "m [g]"
    assert summary.yaxis_title == "T^2 [s^2]"
    assert summary.xscale == "linear"
    assert summary.yscale == "linear"
    _assert_text_and_legend_inside_canvas(mpl_figure, ax)


def test_horizontal_legend_is_present_and_inside_canvas() -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 3, 2], mode="lines", name="Series A"))
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 2.5, 4], mode="lines", name="Series B"))
    fig.update_layout(
        width=900,
        height=560,
        xaxis_title="x",
        yaxis_title="y",
        legend={"orientation": "h", "x": 0.0, "y": 1.02, "xanchor": "left", "yanchor": "bottom"},
        title={"text": "Legend test"},
    )

    mpl_figure, ax, summary = render_plotly_figure_to_matplotlib(fig, width=900, height=560, base_dpi=200)

    assert summary.legend_present is True
    legend = ax.get_legend()
    assert legend is not None
    assert getattr(legend, "_ncols", 1) == 2
    _assert_text_and_legend_inside_canvas(mpl_figure, ax)


def test_horizontal_legend_wraps_long_labels_without_overlap() -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 3, 2], mode="markers", name="Messwerte"))
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3], mode="lines", name="Ausgleichsgerade (a=0.006939)"))
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 1.8, 2.8], mode="lines", name="Fehlergerade mit kleinster Steigung (a=0.006964)"))
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1.2, 2.2, 3.1], mode="lines", name="Fehlergerade mit groesster Steigung (a=0.007202)"))
    fig.update_layout(
        width=1400,
        height=900,
        xaxis_title="x",
        yaxis_title="y",
        legend={"orientation": "h", "x": 0.0, "y": 1.02, "xanchor": "left", "yanchor": "bottom"},
    )

    mpl_figure, ax, summary = render_plotly_figure_to_matplotlib(fig, width=1400, height=900, base_dpi=200)

    assert summary.legend_present is True
    legend = ax.get_legend()
    assert legend is not None
    assert getattr(legend, "_ncols", 1) <= 2
    _assert_text_and_legend_inside_canvas(mpl_figure, ax)
    _assert_legend_texts_do_not_overlap(mpl_figure, ax)


@pytest.mark.parametrize("fmt,signature", [("svg", b"<?xml"), ("pdf", b"%PDF")])
def test_semilog_grid_and_ticks_are_coherent_in_vector_exports(fmt: str, signature: bytes) -> None:
    fig = _normal_preview(log_y=True)

    mpl_figure, ax, summary = render_plotly_figure_to_matplotlib(fig, width=1000, height=640, base_dpi=200)
    tick_labels = [label.get_text() for label in ax.get_yticklabels() if label.get_text().strip()]
    fallback_grid_lines = [line for line in ax.lines if line.get_zorder() <= 0.5]

    assert summary.yscale == "log"
    assert len(tick_labels) >= 3
    assert len(fallback_grid_lines) >= 3
    _assert_text_and_legend_inside_canvas(mpl_figure, ax)

    data = figure_to_image_bytes(fig, fmt, width=1000, height=640, scale=1.0, base_dpi=200)
    assert data.lstrip().startswith(signature)


def test_visual_scaling_changes_non_text_but_not_target_text_size() -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 4, 2], mode="lines+markers", line={"width": 1.5}))
    fig.update_layout(font={"size": 22})

    scaled_small = scale_figure_for_export(fig, visual_scale=1.0, target_text_pt=14.0, base_export_dpi=300, text_unit="pt")
    scaled_large = scale_figure_for_export(fig, visual_scale=3.0, target_text_pt=14.0, base_export_dpi=300, text_unit="pt")

    assert float(scaled_small.layout.font.size) == 14.0
    assert float(scaled_large.layout.font.size) == 14.0
    assert float(scaled_large.data[0].line.width) > float(scaled_small.data[0].line.width)


def test_unsupported_export_feature_fails_with_clear_message() -> None:
    fig = go.Figure()
    fig.add_trace(go.Pie(values=[2, 3, 4], labels=["a", "b", "c"]))

    with pytest.raises(ExportValidationError, match="Unsupported feature in export backend"):
        figure_to_image_bytes(fig, "png", width=800, height=500, scale=1.0, base_dpi=200)


def test_preview_export_parity_keeps_annotations_inside_canvas() -> None:
    fig = _normal_preview(log_y=False)
    fig.add_annotation(
        x=0.98,
        y=0.96,
        xref="paper",
        yref="paper",
        text="export box",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.9)",
    )

    mpl_figure, ax, _summary = render_plotly_figure_to_matplotlib(fig, width=1000, height=640, base_dpi=200)

    _assert_text_and_legend_inside_canvas(mpl_figure, ax)
