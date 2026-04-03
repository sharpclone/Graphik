"""Shared UI helpers for page controllers."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from src.mode_models import PlotInfoBoxConfig

TranslateFn = callable


@dataclass(frozen=True)
class FontSettings:
    """Collected font sizes from the sidebar."""

    base_font_size: int
    axis_title_font_size: int
    tick_font_size: int
    annotation_font_size: int


def _mode_key(prefix: str, name: str) -> str:
    return f"{prefix}{name}" if prefix else name


def render_plot_info_box_controls(translate, default_font_size: int, *, key_prefix: str = "") -> PlotInfoBoxConfig:
    """Render automatic/manual controls for the plot info box."""
    manual_box = st.checkbox(
        translate("plot_settings.info_box_manual"),
        value=False,
        key=_mode_key(key_prefix, "plot_info_box_manual"),
    )
    if not manual_box:
        return PlotInfoBoxConfig(manual=False, font_size=int(default_font_size))

    st.caption(translate("plot_settings.info_box_caption"))
    box_font_size = int(
        st.number_input(
            translate("plot_settings.info_box_font_size"),
            min_value=6,
            max_value=72,
            value=int(default_font_size),
            step=1,
            key=_mode_key(key_prefix, "plot_info_box_font_size"),
        )
    )
    box_x = float(
        st.number_input(
            translate("plot_settings.info_box_x"),
            min_value=0.0,
            max_value=1.0,
            value=0.02,
            step=0.01,
            key=_mode_key(key_prefix, "plot_info_box_x"),
            format="%.2f",
        )
    )
    box_y = float(
        st.number_input(
            translate("plot_settings.info_box_y"),
            min_value=0.0,
            max_value=1.0,
            value=0.98,
            step=0.01,
            key=_mode_key(key_prefix, "plot_info_box_y"),
            format="%.2f",
        )
    )
    box_width = float(
        st.number_input(
            translate("plot_settings.info_box_width"),
            min_value=0.12,
            max_value=0.95,
            value=0.46,
            step=0.01,
            key=_mode_key(key_prefix, "plot_info_box_width"),
            format="%.2f",
        )
    )
    box_height = float(
        st.number_input(
            translate("plot_settings.info_box_height"),
            min_value=0.08,
            max_value=0.95,
            value=0.28,
            step=0.01,
            key=_mode_key(key_prefix, "plot_info_box_height"),
            format="%.2f",
        )
    )
    return PlotInfoBoxConfig(
        manual=True,
        font_size=box_font_size,
        x=box_x,
        y=box_y,
        max_width=box_width,
        max_height=box_height,
    )


def render_font_controls(translate, *, key_prefix: str = "") -> FontSettings:
    """Render shared font-size controls and return resolved values."""
    use_separate_font_sizes = st.checkbox(
        translate("plot_settings.separate_fonts"),
        value=False,
        key=_mode_key(key_prefix, "use_separate_fonts"),
    )
    if use_separate_font_sizes:
        base_font_size = int(
            st.number_input(
                translate("plot_settings.base_font_size"),
                min_value=8,
                max_value=36,
                value=14,
                step=1,
                key=_mode_key(key_prefix, "base_font_size"),
            )
        )
        axis_title_font_size = int(
            st.number_input(
                translate("plot_settings.axis_title_font_size"),
                min_value=8,
                max_value=42,
                value=16,
                step=1,
                key=_mode_key(key_prefix, "axis_title_font_size"),
            )
        )
        tick_font_size = int(
            st.number_input(
                translate("plot_settings.tick_font_size"),
                min_value=8,
                max_value=30,
                value=12,
                step=1,
                key=_mode_key(key_prefix, "tick_font_size"),
            )
        )
        annotation_font_size = int(
            st.number_input(
                translate("plot_settings.annotation_font_size"),
                min_value=8,
                max_value=30,
                value=12,
                step=1,
                key=_mode_key(key_prefix, "annotation_font_size"),
            )
        )
        return FontSettings(base_font_size, axis_title_font_size, tick_font_size, annotation_font_size)

    global_font_size = int(
        st.number_input(
            translate("plot_settings.font_size_all"),
            min_value=8,
            max_value=42,
            value=14,
            step=1,
            key=_mode_key(key_prefix, "global_font_size"),
        )
    )
    return FontSettings(global_font_size, global_font_size, global_font_size, global_font_size)
