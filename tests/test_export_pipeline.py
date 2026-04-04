from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.errors import ExportValidationError
from src.export_pipeline import ExportConfig, ExportRequest, build_export_figure, prepare_export_bytes
from src.export_utils import (
    PLOT_INFO_ANNOTATION_NAME,
    PlotTextBlockLayout,
    autoscale_figure_to_data,
    place_plot_text_block,
)
from src.plotting import PlotStyle, create_base_figure

_EXPORT_SIGNATURES = {
    "png": bytes([137]) + b"PNG",
    "svg": b"<?xml",
    "pdf": b"%PDF",
}


def _normal_preview(y_axis_type: str) -> tuple[go.Figure, np.ndarray, np.ndarray, np.ndarray]:
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
        y_axis_type=y_axis_type,
        y_log_decades=1 if y_axis_type == "log" else None,
        y_major_dtick="D1" if y_axis_type == "log" else 0.5,
        y_minor_dtick="D1" if y_axis_type == "log" else 0.1,
        show_minor_grid=True,
        connect_points=False,
        base_font_size=14,
        axis_title_font_size=16,
        tick_font_size=12,
    )
    fig = create_base_figure(df, style)
    fig.add_annotation(x=125.0, y=2.05, text="fit", showarrow=False)
    return (
        fig,
        df["x"].to_numpy(dtype=float),
        df["y"].to_numpy(dtype=float),
        df["sigma_y"].to_numpy(dtype=float),
    )


def _statistics_preview(axis_mode: str) -> go.Figure:
    heights = [2.0, 4.0, 6.0, 5.0] if axis_mode == "linear" else [2.0, 4.0, 8.0, 6.0]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[1.0, 2.0, 3.0, 4.0], y=heights, name="Histogram"))
    fig.add_annotation(
        x=0.98,
        y=0.96,
        xref="paper",
        yref="paper",
        text="&mu; = 2.5<br>&sigma; = 1.1",
        showarrow=False,
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.85)",
    )
    fig.update_layout(
        template="plotly_white",
        width=900,
        height=560,
        xaxis_title="value",
        yaxis_title="count",
        font={"size": 14},
        legend={"orientation": "h", "x": 0.0, "y": 1.02},
        margin={"l": 60, "r": 30, "t": 60, "b": 60},
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(120,120,120,0.32)")
    if axis_mode == "log":
        fig.update_yaxes(type="log", range=[0, 1], showgrid=True, gridcolor="rgba(120,120,120,0.32)")
    else:
        fig.update_yaxes(showgrid=True, gridcolor="rgba(120,120,120,0.32)", range=[0.0, 7.0])
    return fig


def _manual_layout(enabled: bool) -> PlotTextBlockLayout:
    if not enabled:
        return PlotTextBlockLayout()
    return PlotTextBlockLayout(manual=True, x=0.08, y=0.94, max_width=0.26, max_height=0.18)


@pytest.mark.parametrize("mode", ["normal", "statistics"])
@pytest.mark.parametrize("fmt", ["png", "svg", "pdf"])
@pytest.mark.parametrize("axis_mode", ["linear", "log"])
@pytest.mark.parametrize("manual_info_box", [False, True])
@pytest.mark.parametrize("autoscale_axes", [False, True])
def test_prepare_export_bytes_regression_matrix(
    mode: str,
    fmt: str,
    axis_mode: str,
    manual_info_box: bool,
    autoscale_axes: bool,
) -> None:
    if mode == "normal":
        preview_figure, x_values, y_values, sigma_y_values = _normal_preview(axis_mode)

        def _build_base() -> go.Figure:
            return (
                autoscale_figure_to_data(preview_figure, x_values, y_values, sigma_y_values, axis_mode)
                if autoscale_axes
                else go.Figure(preview_figure)
            )

        signature_payload = {"fit_model": "linear", "axis_mode": axis_mode}
        info_lines = ("y = ax + b", "a = 0.0132")
    else:
        preview_figure = _statistics_preview(axis_mode)

        def _build_base() -> go.Figure:
            return go.Figure(preview_figure)

        signature_payload = {"stats_mode": True, "axis_mode": axis_mode}
        info_lines = ("Normal distribution", "mu = 2.5", "sigma = 1.1")

    request = ExportRequest(
        mode=mode,
        cache_prefix=f"{mode}_{fmt}",
        preview_figure=preview_figure,
        config=ExportConfig(
            base_name=f"{mode}_plot",
            paper="A5",
            orientation="landscape",
            dpi=72,
            raster_scale=1.0,
            visual_scale=1.0,
            target_text_pt=12.0,
            autoscale_axes=autoscale_axes,
        ),
        plot_info_lines=info_lines,
        plot_info_box_layout=_manual_layout(manual_info_box),
        plot_info_box_font_size=12,
        annotation_font_size=12,
        signature_payload=signature_payload,
    )

    export_figure = build_export_figure(request, _build_base)
    info_annotations = [
        ann for ann in (export_figure.layout.annotations or []) if getattr(ann, "name", None) == PLOT_INFO_ANNOTATION_NAME
    ]
    assert len(info_annotations) == 1

    export_bytes = prepare_export_bytes(request, fmt, _build_base)
    assert export_bytes.lstrip().startswith(_EXPORT_SIGNATURES[fmt])


def test_place_plot_text_block_is_idempotent() -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 3, 2], mode="lines"))
    fig.update_layout(width=900, height=560)

    place_plot_text_block(fig, ["y = ax + b", "a = 2.1"], font_size=13)
    place_plot_text_block(fig, ["y = ax + b", "a = 2.1"], font_size=13)

    info_annotations = [
        ann for ann in (fig.layout.annotations or []) if getattr(ann, "name", None) == PLOT_INFO_ANNOTATION_NAME
    ]
    assert len(info_annotations) == 1


def test_statistics_build_export_figure_applies_autoscale_when_enabled() -> None:
    preview_figure = _statistics_preview("linear")

    def _build_base() -> go.Figure:
        from src.export_utils import autoscale_figure_to_data

        return autoscale_figure_to_data(
            preview_figure,
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            "linear",
        )

    request = ExportRequest(
        mode="statistics",
        cache_prefix="statistics_autoscale",
        preview_figure=preview_figure,
        config=ExportConfig(
            base_name="statistics_plot",
            paper="A5",
            orientation="landscape",
            dpi=72,
            raster_scale=1.0,
            visual_scale=1.0,
            target_text_pt=12.0,
            autoscale_axes=True,
        ),
        plot_info_lines=("Normal distribution",),
        plot_info_box_layout=PlotTextBlockLayout(),
        plot_info_box_font_size=12,
        annotation_font_size=12,
        signature_payload={"stats_mode": True},
    )

    export_figure = build_export_figure(request, _build_base)
    x_range = list(export_figure.layout.xaxis.range)
    preview_x = np.asarray(preview_figure.data[0].x, dtype=float)

    assert x_range[0] < float(np.min(preview_x))
    assert x_range[1] > float(np.max(preview_x))


def test_prepare_export_bytes_writes_debug_report_for_validation_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("src.export_pipeline.EXPORT_DEBUG_DIR", tmp_path)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[1.0, 2.0],
            y=[1.0, 2.0],
            mode="markers",
            error_y={"type": "data", "array": [2.0, 0.1], "visible": True},
        )
    )
    fig.update_layout(yaxis={"type": "log"})

    request = ExportRequest(
        mode="normal",
        cache_prefix="invalid_log",
        preview_figure=fig,
        config=ExportConfig(
            base_name="invalid",
            paper="A5",
            orientation="landscape",
            dpi=72,
            raster_scale=1.0,
            visual_scale=1.0,
            target_text_pt=12.0,
            autoscale_axes=False,
        ),
        plot_info_lines=("invalid",),
    )

    with pytest.raises(ExportValidationError) as exc_info:
        prepare_export_bytes(request, "png", lambda: go.Figure(fig))

    assert exc_info.value.report_path is not None
    assert exc_info.value.report_path.exists()
    report_text = exc_info.value.report_path.read_text(encoding="utf-8")
    assert "invalid_log" in report_text
    assert "log-y export would place lower error bars at or below zero" in report_text


def test_normal_build_export_figure_respects_custom_x_range_even_with_autoscale_enabled() -> None:
    preview_figure, x_values, y_values, sigma_y_values = _normal_preview("linear")
    preview_figure.update_xaxes(range=[80.0, 180.0])

    def _build_base() -> go.Figure:
        return autoscale_figure_to_data(
            preview_figure,
            x_values,
            y_values,
            sigma_y_values,
            "linear",
            preserve_x_range=True,
            preserve_y_range=False,
        )

    request = ExportRequest(
        mode="normal",
        cache_prefix="normal_custom_range",
        preview_figure=preview_figure,
        config=ExportConfig(
            base_name="normal_plot",
            paper="A5",
            orientation="landscape",
            dpi=72,
            raster_scale=1.0,
            visual_scale=1.0,
            target_text_pt=12.0,
            autoscale_axes=True,
        ),
        plot_info_lines=("y = ax + b",),
        plot_info_box_layout=PlotTextBlockLayout(),
        plot_info_box_font_size=12,
        annotation_font_size=12,
        signature_payload={"fit_model": "linear"},
    )

    export_figure = build_export_figure(request, _build_base)
    x_range = list(export_figure.layout.xaxis.range)

    assert x_range == [80.0, 180.0]
