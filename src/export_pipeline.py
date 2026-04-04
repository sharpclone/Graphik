from __future__ import annotations

import hashlib
import json
import logging
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .errors import ExportRenderError, ExportValidationError
from .export_utils import (
    PlotTextBlockLayout,
    mm_to_px,
    paper_size_mm,
    place_plot_text_block,
    scale_figure_for_export,
    scaled_text_font_size_for_export,
)
from .logging_utils import log_event, runtime_mode
from .mpl_export import supported_export_contract_lines
from .plotting import figure_to_image_bytes

ExportFormat = Literal["png", "svg", "pdf"]
EXPORT_DEBUG_DIR = Path(".streamlit") / "export_debug"

LOGGER = logging.getLogger('graphik.export')


EXPORT_PRESETS: dict[str, dict[str, object]] = {
    "custom": {},
    "word_report": {
        "paper": "A4",
        "orientation": "portrait",
        "dpi": 220,
        "raster_scale": 1.0,
        "visual_scale": 1.0,
        "target_text_pt": 12.0,
        "autoscale_axes": True,
    },
    "a4_print": {
        "paper": "A4",
        "orientation": "landscape",
        "dpi": 300,
        "raster_scale": 1.0,
        "visual_scale": 1.0,
        "target_text_pt": 14.0,
        "autoscale_axes": True,
    },
    "svg_publication": {
        "paper": "A4",
        "orientation": "landscape",
        "dpi": 300,
        "raster_scale": 1.0,
        "visual_scale": 1.0,
        "target_text_pt": 11.0,
        "autoscale_axes": False,
    },
    "fast_preview": {
        "paper": "A5",
        "orientation": "landscape",
        "dpi": 144,
        "raster_scale": 1.0,
        "visual_scale": 1.0,
        "target_text_pt": 12.0,
        "autoscale_axes": False,
    },
}


def _export_preset_settings(name: str) -> dict[str, object]:
    return dict(EXPORT_PRESETS.get(name, {}))


def _resolved_export_config(config: "ExportConfig") -> dict[str, object]:
    base = {
        "base_name": config.base_name,
        "paper": config.paper,
        "orientation": config.orientation,
        "dpi": config.dpi,
        "raster_scale": config.raster_scale,
        "visual_scale": config.visual_scale,
        "target_text_pt": config.target_text_pt,
        "autoscale_axes": config.autoscale_axes,
        "preset_name": config.preset_name,
        "show_clean_export_preview": config.show_clean_export_preview,
    }
    return base


@dataclass(frozen=True)
class ExportConfig:
    """Shared export settings used by all plotting modes."""

    base_name: str
    paper: str
    orientation: str
    dpi: int
    raster_scale: float
    visual_scale: float
    target_text_pt: float | None
    autoscale_axes: bool = False
    preset_name: str = "custom"
    show_clean_export_preview: bool = False

    def canvas_size_px(self) -> tuple[int, int]:
        """Return target export canvas size in pixels."""
        paper_w_mm, paper_h_mm = paper_size_mm(self.paper)
        if self.orientation == "landscape":
            paper_w_mm, paper_h_mm = paper_h_mm, paper_w_mm
        return mm_to_px(paper_w_mm, self.dpi), mm_to_px(paper_h_mm, self.dpi)

    def scale_for_format(self, fmt: ExportFormat) -> float:
        """Return export scale used for the given format."""
        return float(max(0.2, self.raster_scale)) if fmt == "png" else 1.0


@dataclass(frozen=True)
class ExportRequest:
    """All information required to build and export a plot artifact."""

    mode: str
    cache_prefix: str
    preview_figure: go.Figure
    config: ExportConfig
    plot_info_lines: tuple[str, ...] = field(default_factory=tuple)
    plot_info_box_layout: PlotTextBlockLayout = field(default_factory=PlotTextBlockLayout)
    plot_info_box_font_size: int = 12
    annotation_font_size: int = 12
    signature_payload: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportArtifactSpec:
    """UI metadata for one export format."""

    fmt: ExportFormat
    mime: str
    download_label: str
    prepare_label: str
    spinner_label: str
    unavailable_message_key: str


FigureBuilder = Callable[[], go.Figure]
TranslateWithKwargsFn = Callable[..., str]


def render_export_settings(
    *,
    translate: TranslateWithKwargsFn,
    default_base_name: str,
    include_autoscale: bool,
    key_prefix: str = "",
) -> ExportConfig:
    """Render shared export controls and return the current export config."""

    def _key(name: str) -> str:
        return f"{key_prefix}{name}" if key_prefix else name

    st.header(translate("export.sidebar_header"))
    st.caption(translate("export.support_contract", contract="; ".join(supported_export_contract_lines())))
    preset_name = st.selectbox(
        translate("export.preset"),
        options=list(EXPORT_PRESETS.keys()),
        index=0,
        key=_key("export_preset"),
        format_func=lambda value: translate(f"export_preset.{value}"),
    )
    preset = _export_preset_settings(preset_name)
    manual_enabled = preset_name == "custom"
    if not manual_enabled:
        st.caption(translate("export.preset_caption", preset=translate(f"export_preset.{preset_name}")))

    base_name = st.text_input(translate("export.filename_prefix"), value=default_base_name, key=_key("export_base"))
    paper = st.selectbox(
        translate("export.paper_size"),
        options=["A4", "A5", "Letter"],
        index=0,
        key=_key("png_paper"),
        disabled=not manual_enabled,
    )
    orientation = st.selectbox(
        translate("export.orientation"),
        options=["portrait", "landscape"],
        index=1,
        key=_key("png_orientation"),
        format_func=lambda value: translate(f"orientation.{value}"),
        disabled=not manual_enabled,
    )
    dpi = int(
        st.number_input(
            translate("export.png_dpi"),
            min_value=72,
            max_value=600,
            value=300,
            step=1,
            key=_key("png_dpi"),
            disabled=not manual_enabled,
        )
    )
    raster_scale = float(
        st.number_input(
            translate("export.png_scale"),
            min_value=1.0,
            max_value=4.0,
            value=1.0,
            step=0.1,
            key=_key("png_scale"),
            disabled=not manual_enabled,
        )
    )
    word_like_text = bool(
        st.checkbox(
            translate("export.word_like_text"),
            value=True,
            key=_key("png_word_like"),
            disabled=not manual_enabled,
        )
    )
    target_text_pt = float(
        st.number_input(
            translate("export.target_text_size"),
            min_value=6.0,
            max_value=36.0,
            value=14.0,
            step=0.5,
            disabled=(not word_like_text) or (not manual_enabled),
            key=_key("png_text_pt"),
        )
    )
    visual_scale = float(
        st.number_input(
            translate("export.extra_visual_scale"),
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.1,
            key=_key("png_visual_scale"),
            disabled=not manual_enabled,
        )
    )
    autoscale_axes = False
    if include_autoscale:
        autoscale_axes = bool(
            st.checkbox(
                translate("export.autoscale"),
                value=True,
                key=_key("autoscale_export_axes"),
                help=translate("export.autoscale_help"),
                disabled=not manual_enabled,
            )
        )
    show_clean_export_preview = bool(
        st.checkbox(
            translate("export.clean_preview"),
            value=False,
            key=_key("show_clean_export_preview"),
            help=translate("export.clean_preview_help"),
        )
    )

    if not manual_enabled:
        paper = str(preset.get("paper", paper))
        orientation = str(preset.get("orientation", orientation))
        dpi = _coerce_int(preset.get("dpi", dpi), field_name="export preset dpi")
        raster_scale = _coerce_float(preset.get("raster_scale", raster_scale), field_name="export preset raster_scale")
        visual_scale = _coerce_float(preset.get("visual_scale", visual_scale), field_name="export preset visual_scale")
        target_text_pt = _coerce_float(preset.get("target_text_pt", target_text_pt), field_name="export preset target_text_pt")
        autoscale_axes = bool(preset.get("autoscale_axes", autoscale_axes))
        word_like_text = target_text_pt is not None

    config = ExportConfig(
        base_name=base_name,
        paper=paper,
        orientation=orientation,
        dpi=dpi,
        raster_scale=raster_scale,
        visual_scale=visual_scale,
        target_text_pt=target_text_pt if word_like_text else None,
        autoscale_axes=autoscale_axes,
        preset_name=preset_name,
        show_clean_export_preview=show_clean_export_preview,
    )
    preset_json = json.dumps(_resolved_export_config(config), indent=2, sort_keys=True)
    st.download_button(
        translate("export.download_preset"),
        data=preset_json.encode("utf-8"),
        file_name=f"{config.base_name or 'graphik'}_export_preset.json",
        mime="application/json",
        use_container_width=True,
        key=_key("download_export_preset"),
    )
    return config


def _coerce_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ExportValidationError(f"Export validation failed: {field_name} must be numeric, not boolean.")
    if isinstance(value, (int, float, str)):
        return int(value)
    raise ExportValidationError(f"Export validation failed: {field_name} is not numeric.")


def _coerce_float(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ExportValidationError(f"Export validation failed: {field_name} must be numeric, not boolean.")
    if isinstance(value, (int, float, str)):
        return float(value)
    raise ExportValidationError(f"Export validation failed: {field_name} is not numeric.")


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _as_float_array(values: Any, field_name: str) -> np.ndarray:
    if values is None:
        return np.array([], dtype=float)
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ExportValidationError(f"Export validation failed: {field_name} is not numeric.") from exc
    if arr.ndim == 0:
        arr = np.array([float(arr)], dtype=float)
    return arr.astype(float)


def export_signature(request: ExportRequest, fmt: ExportFormat) -> str:
    """Build a stable signature for export caching."""
    width, height = request.config.canvas_size_px()
    payload = {
        "mode": request.mode,
        "format": fmt,
        "width": width,
        "height": height,
        "scale": request.config.scale_for_format(fmt),
        "config": asdict(request.config),
        "plot_info_lines": list(request.plot_info_lines),
        "plot_info_box_layout": asdict(request.plot_info_box_layout),
        "plot_info_box_font_size": int(request.plot_info_box_font_size),
        "annotation_font_size": int(request.annotation_font_size),
        **dict(request.signature_payload),
    }
    digest = hashlib.sha256()
    digest.update(request.preview_figure.to_json().encode("utf-8"))
    digest.update(json.dumps(payload, sort_keys=True, default=_json_default).encode("utf-8"))
    return digest.hexdigest()


def _axis_range_is_valid(axis: Any, axis_name: str) -> None:
    axis_range = getattr(axis, "range", None)
    if axis_range is None:
        return
    if len(axis_range) != 2 or axis_range[0] is None or axis_range[1] is None:
        raise ExportValidationError(f"Export validation failed: {axis_name} range is incomplete.")
    try:
        start = float(axis_range[0])
        end = float(axis_range[1])
    except (TypeError, ValueError) as exc:
        raise ExportValidationError(f"Export validation failed: {axis_name} range is not finite.") from exc
    if not np.isfinite([start, end]).all():
        raise ExportValidationError(f"Export validation failed: {axis_name} range contains non-finite values.")
    if start == end:
        raise ExportValidationError(f"Export validation failed: {axis_name} range collapses to a single value.")


def _validate_axis_positive_for_log(fig: go.Figure, axis_name: str) -> None:
    for trace in fig.data:
        if getattr(trace, "visible", True) in (False, "legendonly"):
            continue
        trace_type = str(getattr(trace, "type", "scatter")).lower()
        values = getattr(trace, axis_name, None)
        if trace_type not in {"scatter", "bar"} or values is None:
            continue
        arr = _as_float_array(values, f"trace.{axis_name}")
        if arr.size and np.any(arr <= 0):
            raise ExportValidationError(
                f"Export validation failed: log-{axis_name} axis requires strictly positive visible data."
            )
        err = getattr(trace, f"error_{axis_name}", None)
        if err is None or not bool(getattr(err, "visible", False)):
            continue
        plus = _as_float_array(getattr(err, "array", None), f"trace.error_{axis_name}.array")
        if plus.size == 0:
            continue
        minus = _as_float_array(getattr(err, "arrayminus", None), f"trace.error_{axis_name}.arrayminus")
        if minus.size == 0:
            minus = plus
        if minus.size == 1 and arr.size > 1:
            minus = np.full(arr.size, float(minus[0]), dtype=float)
        lower = arr[: minus.size] - minus[: arr.size]
        if np.any(lower <= 0):
            raise ExportValidationError(
                f"Export validation failed: log-{axis_name} export would place lower error bars at or below zero."
            )


def _validate_figure(fig: go.Figure, *, require_dimensions: bool) -> None:
    if require_dimensions:
        width = float(fig.layout.width) if fig.layout.width is not None else 0.0
        height = float(fig.layout.height) if fig.layout.height is not None else 0.0
        if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
            raise ExportValidationError(
                "Export validation failed: figure width/height must be positive finite values."
            )

    layout_font = getattr(fig.layout, "font", None)
    if layout_font is not None and getattr(layout_font, "size", None) is not None:
        font_size = float(layout_font.size)
        if not np.isfinite(font_size) or font_size <= 0:
            raise ExportValidationError("Export validation failed: base font size is invalid.")

    xaxis = fig.layout.xaxis
    yaxis = fig.layout.yaxis
    _axis_range_is_valid(xaxis, "x-axis")
    _axis_range_is_valid(yaxis, "y-axis")

    if str(getattr(xaxis, "type", "linear")).lower() == "log":
        _validate_axis_positive_for_log(fig, "x")
    if str(getattr(yaxis, "type", "linear")).lower() == "log":
        _validate_axis_positive_for_log(fig, "y")

    for trace in fig.data:
        if getattr(trace, "visible", True) in (False, "legendonly"):
            continue
        trace_type = str(getattr(trace, "type", "scatter")).lower()
        if trace_type not in {"scatter", "bar"}:
            continue
        for field_name in ("x", "y"):
            values = getattr(trace, field_name, None)
            arr = _as_float_array(values, f"trace.{field_name}")
            if arr.size and not np.all(np.isfinite(arr)):
                raise ExportValidationError(f"Export validation failed: trace {field_name} contains NaN/inf.")
        for axis_name in ("x", "y"):
            err = getattr(trace, f"error_{axis_name}", None)
            if err is None or not bool(getattr(err, "visible", False)):
                continue
            for attr_name in ("array", "arrayminus"):
                values = getattr(err, attr_name, None)
                if values is None:
                    continue
                arr = _as_float_array(values, f"trace.error_{axis_name}.{attr_name}")
                if arr.size and not np.all(np.isfinite(arr)):
                    raise ExportValidationError(
                        f"Export validation failed: error_{axis_name}.{attr_name} contains NaN/inf."
                    )

    for annotation in fig.layout.annotations or []:
        try:
            annotation.to_plotly_json()
        except Exception as exc:  # pragma: no cover - defensive wrapper around Plotly internals
            raise ExportValidationError("Export validation failed: annotation is not serializable.") from exc
        if annotation.font is not None and getattr(annotation.font, "size", None) is not None:
            ann_font_size = float(annotation.font.size)
            if not np.isfinite(ann_font_size) or ann_font_size <= 0:
                raise ExportValidationError("Export validation failed: annotation font size is invalid.")
        for attr_name in ("x", "y"):
            value = getattr(annotation, attr_name, None)
            if value is None:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ExportValidationError(
                    f"Export validation failed: annotation {attr_name} is not numeric."
                ) from exc
            if not np.isfinite(numeric_value):
                raise ExportValidationError(
                    f"Export validation failed: annotation {attr_name} is not finite."
                )

    for shape in fig.layout.shapes or []:
        for attr_name in ("x0", "x1", "y0", "y1"):
            value = getattr(shape, attr_name, None)
            if value is None:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ExportValidationError(
                    f"Export validation failed: shape coordinate {attr_name} is not numeric."
                ) from exc
            if not np.isfinite(numeric_value):
                raise ExportValidationError(
                    f"Export validation failed: shape coordinate {attr_name} is not finite."
                )


def validate_export_figure(fig: go.Figure) -> None:
    """Validate the final figure before rendering an export artifact."""
    _validate_figure(fig, require_dimensions=True)


def validate_export_request(request: ExportRequest) -> None:
    """Validate export settings before building the export figure."""
    width, height = request.config.canvas_size_px()
    if width <= 0 or height <= 0:
        raise ExportValidationError("Export validation failed: paper size and DPI produce an invalid canvas size.")
    if not np.isfinite(float(request.config.dpi)) or request.config.dpi <= 0:
        raise ExportValidationError("Export validation failed: DPI must be a positive finite value.")
    if not np.isfinite(request.config.visual_scale) or request.config.visual_scale <= 0:
        raise ExportValidationError("Export validation failed: visual scale must be a positive finite value.")
    if not np.isfinite(request.config.raster_scale) or request.config.raster_scale <= 0:
        raise ExportValidationError("Export validation failed: raster scale must be a positive finite value.")
    if request.config.target_text_pt is not None:
        if not np.isfinite(request.config.target_text_pt) or request.config.target_text_pt <= 0:
            raise ExportValidationError("Export validation failed: target text size must be positive and finite.")
    if not np.isfinite(float(request.plot_info_box_font_size)) or request.plot_info_box_font_size <= 0:
        raise ExportValidationError("Export validation failed: info-box font size must be positive and finite.")
    if not np.isfinite(float(request.annotation_font_size)) or request.annotation_font_size <= 0:
        raise ExportValidationError("Export validation failed: annotation font size must be positive and finite.")
    layout = request.plot_info_box_layout
    for field_name in ("x", "y", "max_width", "max_height"):
        value = getattr(layout, field_name)
        if value is not None and not np.isfinite(float(value)):
            raise ExportValidationError(f"Export validation failed: info-box {field_name} is not finite.")
    _validate_figure(request.preview_figure, require_dimensions=False)


def _compute_export_plot_text_font_size(request: ExportRequest, export_figure: go.Figure) -> int:
    export_base_font_size = (
        float(export_figure.layout.font.size)
        if export_figure.layout.font is not None and export_figure.layout.font.size is not None
        else float(request.annotation_font_size)
    )
    if not request.plot_info_box_layout.manual:
        return int(round(export_base_font_size))

    preview_base_font_size = (
        float(request.preview_figure.layout.font.size)
        if request.preview_figure.layout.font is not None and request.preview_figure.layout.font.size is not None
        else float(request.annotation_font_size)
    )
    return scaled_text_font_size_for_export(
        requested_font_size=int(request.plot_info_box_font_size),
        preview_base_font_size=preview_base_font_size,
        export_base_font_size=export_base_font_size,
    )


def build_export_figure(request: ExportRequest, build_base_figure: FigureBuilder) -> go.Figure:
    """Build a validated export figure shared by all application modes."""
    validate_export_request(request)

    base_figure = build_base_figure()
    if not isinstance(base_figure, go.Figure):
        raise ExportValidationError("Export validation failed: export builder did not return a Plotly figure.")

    width, height = request.config.canvas_size_px()
    export_figure = go.Figure(base_figure)
    export_figure.update_layout(width=width, height=height)
    export_figure = scale_figure_for_export(
        export_figure,
        visual_scale=float(request.config.visual_scale),
        target_text_pt=float(request.config.target_text_pt) if request.config.target_text_pt is not None else None,
        base_export_dpi=int(request.config.dpi),
        text_unit="pt",
    )
    export_box_font_size = _compute_export_plot_text_font_size(request, export_figure)
    place_plot_text_block(
        export_figure,
        list(request.plot_info_lines),
        font_size=export_box_font_size,
        layout=request.plot_info_box_layout,
    )
    validate_export_figure(export_figure)
    return export_figure


def _debug_report_payload(
    request: ExportRequest,
    fmt: ExportFormat,
    error: BaseException,
    traceback_text: str,
    export_figure: go.Figure | None,
) -> dict[str, Any]:
    width, height = request.config.canvas_size_px()
    payload: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cache_prefix": request.cache_prefix,
        "mode": request.mode,
        "format": fmt,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback_text,
        "config": asdict(request.config),
        "canvas": {
            "width": width,
            "height": height,
            "scale": request.config.scale_for_format(fmt),
        },
        "plot_info_lines": list(request.plot_info_lines),
        "plot_info_box_layout": asdict(request.plot_info_box_layout),
        "plot_info_box_font_size": int(request.plot_info_box_font_size),
        "annotation_font_size": int(request.annotation_font_size),
        "signature_payload": dict(request.signature_payload),
        "preview_layout": request.preview_figure.layout.to_plotly_json(),
        "preview_axis_types": {
            "x": str(getattr(request.preview_figure.layout.xaxis, "type", "linear")),
            "y": str(getattr(request.preview_figure.layout.yaxis, "type", "linear")),
        },
    }
    if export_figure is not None:
        payload["export_layout"] = export_figure.layout.to_plotly_json()
        payload["export_axis_types"] = {
            "x": str(getattr(export_figure.layout.xaxis, "type", "linear")),
            "y": str(getattr(export_figure.layout.yaxis, "type", "linear")),
        }
    return payload


def write_export_debug_report(
    request: ExportRequest,
    fmt: ExportFormat,
    error: BaseException,
    traceback_text: str,
    export_figure: go.Figure | None = None,
) -> Path:
    """Persist a structured debug report for failed exports."""
    EXPORT_DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = EXPORT_DEBUG_DIR / f"{request.cache_prefix}_{fmt}_{timestamp}.json"
    payload = _debug_report_payload(request, fmt, error, traceback_text, export_figure)
    report_path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return report_path


def prepare_export_bytes(request: ExportRequest, fmt: ExportFormat, build_base_figure: FigureBuilder) -> bytes:
    """Build, validate, and render one export artifact."""
    export_figure: go.Figure | None = None
    width, height = request.config.canvas_size_px()
    log_context = {
        "mode": request.mode,
        "format": fmt,
        "figure_width": width,
        "figure_height": height,
        "dpi": int(request.config.dpi),
        "paper": request.config.paper,
        "orientation": request.config.orientation,
        "autoscale": bool(request.config.autoscale_axes),
        "trace_count": len(request.preview_figure.data),
        "y_axis_type": str(getattr(request.preview_figure.layout.yaxis, 'type', 'linear')),
        "packaged": runtime_mode() == 'packaged',
    }
    log_event(LOGGER, logging.INFO, 'export_prepare_started', **log_context)
    try:
        export_figure = build_export_figure(request, build_base_figure)
        export_bytes = figure_to_image_bytes(
            export_figure,
            fmt,
            width=width,
            height=height,
            scale=request.config.scale_for_format(fmt),
            base_dpi=int(request.config.dpi),
        )
        log_event(LOGGER, logging.INFO, 'export_prepare_succeeded', byte_count=len(export_bytes), **log_context)
        return export_bytes
    except ExportValidationError as exc:
        report_path = write_export_debug_report(request, fmt, exc, traceback.format_exc(), export_figure)
        log_event(LOGGER, logging.ERROR, 'export_prepare_validation_failed', error=str(exc), report_path=report_path, **log_context)
        raise ExportValidationError(f"{exc} Debug report: {report_path}", report_path=report_path) from exc
    except Exception as exc:  # pragma: no cover - defensive boundary around renderer stack
        report_path = write_export_debug_report(request, fmt, exc, traceback.format_exc(), export_figure)
        log_event(LOGGER, logging.ERROR, 'export_prepare_render_failed', error=str(exc), report_path=report_path, **log_context)
        raise ExportRenderError(
            f"Export render failed for {fmt.upper()}. Debug report: {report_path}",
            report_path=report_path,
        ) from exc


def _render_on_demand_image_export(
    *,
    cache_key: str,
    prepare_label: str,
    spinner_label: str,
    download_label: str,
    file_name: str,
    mime: str,
    signature: str,
    build_bytes: Callable[[], bytes],
) -> None:
    """Prepare heavy image exports only when explicitly requested."""
    state_key = f"_export_cache_{cache_key}"
    cached = st.session_state.get(state_key)
    if not isinstance(cached, dict) or cached.get("signature") != signature:
        st.session_state.pop(state_key, None)
        cached = None

    if st.button(prepare_label, key=f"{state_key}_prepare", use_container_width=True):
        with st.spinner(spinner_label):
            export_bytes = build_bytes()
        cached = {"signature": signature, "data": export_bytes}
        st.session_state[state_key] = cached

    if isinstance(cached, dict) and isinstance(cached.get("data"), (bytes, bytearray)):
        st.download_button(
            download_label,
            data=bytes(cached["data"]),
            file_name=file_name,
            mime=mime,
            use_container_width=True,
            key=f"{state_key}_download",
        )


def render_export_buttons(
    columns: Sequence[Any],
    request: ExportRequest,
    *,
    translate: TranslateWithKwargsFn,
    build_base_figure: FigureBuilder,
) -> None:
    """Render the shared PNG/SVG/PDF export controls."""
    if len(columns) < 3:
        raise ValueError("render_export_buttons expects three Streamlit column containers.")

    specs = (
        ExportArtifactSpec(
            fmt="png",
            mime="image/png",
            prepare_label=translate("export.prepare", format="PNG"),
            spinner_label=translate("export.preparing", format="PNG"),
            download_label=translate(
                "export.download_png",
                paper=request.config.paper,
                orientation=translate(f"orientation.{request.config.orientation}"),
            ),
            unavailable_message_key="export.png_unavailable",
        ),
        ExportArtifactSpec(
            fmt="svg",
            mime="image/svg+xml",
            prepare_label=translate("export.prepare", format="SVG"),
            spinner_label=translate("export.preparing", format="SVG"),
            download_label=translate("export.download_svg"),
            unavailable_message_key="export.svg_unavailable",
        ),
        ExportArtifactSpec(
            fmt="pdf",
            mime="application/pdf",
            prepare_label=translate("export.prepare", format="PDF"),
            spinner_label=translate("export.preparing", format="PDF"),
            download_label=translate(
                "export.download_pdf",
                paper=request.config.paper,
                orientation=translate(f"orientation.{request.config.orientation}"),
            ),
            unavailable_message_key="export.pdf_unavailable",
        ),
    )

    for column, spec in zip(columns[:3], specs, strict=True):
        with column:
            signature = export_signature(request, spec.fmt)

            def _build_bytes(fmt: ExportFormat = spec.fmt) -> bytes:
                return prepare_export_bytes(request, fmt, build_base_figure)

            try:
                _render_on_demand_image_export(
                    cache_key=f"{request.cache_prefix}_{spec.fmt}",
                    prepare_label=spec.prepare_label,
                    spinner_label=spec.spinner_label,
                    download_label=spec.download_label,
                    file_name=f"{request.config.base_name}.{spec.fmt}",
                    mime=spec.mime,
                    signature=signature,
                    build_bytes=_build_bytes,
                )
            except (ExportValidationError, ExportRenderError) as exc:
                st.caption(translate(spec.unavailable_message_key, error=exc))
