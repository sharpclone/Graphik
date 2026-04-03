"""Streamlit bootstrap for the Graphik plotting application."""

from __future__ import annotations

import base64

import streamlit as st

from pages.normal_mode import render_normal_mode
from pages.statistics_mode import render_statistics_mode
from services.state_service import (
    apply_mode_switch_reset,
    bootstrap_session_state,
    clear_session_snapshot,
    migrate_legacy_widget_keys,
    normalize_legacy_selection,
    reset_corrupted_settings_state,
    reset_view_state,
    save_session_snapshot,
)
from src.config import APP_TITLE, FAVICON_PATH, LOGO_DARKBLUE_PATH, LOGO_WHITE_PATH
from src.data_io import get_sample_dataframe, get_statistics_sample_dataframe, load_table_file
from src.i18n import LANGUAGE_NAMES, translate


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=str(FAVICON_PATH) if FAVICON_PATH.exists() else None,
    layout="wide",
)


SAMPLE_DF = get_sample_dataframe()
STATISTICS_SAMPLE_DF = get_statistics_sample_dataframe()
bootstrap_session_state(st.session_state, SAMPLE_DF)


def _t(key: str, **kwargs: object) -> str:
    return translate(str(st.session_state.get("language", "de")), key, **kwargs)


def _apply_sidebar_brand_css() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding-top: 0 !important;
        }
        [data-testid="stSidebar"] .block-container {
            padding-top: 0 !important;
            padding-bottom: 1rem !important;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            padding-top: 0 !important;
        }
        [data-testid="stSidebar"] .sidebar-brand-shell {
            margin: -1rem -1rem 0.85rem -1rem;
            padding: 0.2rem 0.85rem 0.3rem 0.85rem;
            background: linear-gradient(135deg, rgb(2, 8, 24) 0%, rgb(8, 24, 44) 100%);
            border-bottom: 1px solid rgba(94, 223, 255, 0.12);
            overflow: hidden;
            position: relative;
            top: -0.35rem;
        }
        [data-testid="stSidebar"] .sidebar-brand-shell img {
            width: 100%;
            max-width: 100%;
            height: auto;
            display: block;
        }
        [data-testid="stSidebar"] .sidebar-brand-credit {
            margin: 0.15rem 0 0.65rem 0;
            color: rgba(71, 85, 105, 0.78);
            font-size: 0.78rem;
            letter-spacing: 0.01em;
            padding: 0 0.85rem;
        }
        [data-testid="stDeployButton"],
        .stAppDeployButton,
        button[kind="header"]:has([data-testid="stDeployButton"]) {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_brand_header() -> None:
    logo_path = LOGO_DARKBLUE_PATH if LOGO_DARKBLUE_PATH.exists() else LOGO_WHITE_PATH
    if not logo_path.exists():
        return
    encoded_logo = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    st.markdown(
        f"""
        <div class="sidebar-brand-shell">
            <img src="data:image/png;base64,{encoded_logo}" alt="{APP_TITLE} logo" />
        </div>
        <div class="sidebar-brand-credit">made by Mihai Cazac</div>
        """,
        unsafe_allow_html=True,
    )


def _normalize_legacy_state() -> None:
    migrate_legacy_widget_keys(st.session_state)
    normalize_legacy_selection(st.session_state, "app_mode", {"physics": "normal"})
    for key in (
        "normal.mapping_mode",
        "normal.y_scale_mode",
        "normal.log_decade_mode",
        "normal.fit_model",
        "normal.error_line_mode_widget",
        "normal.png_orientation",
        "stats.png_orientation",
    ):
        normalize_legacy_selection(st.session_state, key)


_normalize_legacy_state()
_apply_sidebar_brand_css()

with st.sidebar:
    _render_brand_header()
    st.header(_t("settings.header"))
    st.selectbox(_t("settings.language"), options=["de", "en"], format_func=lambda code: LANGUAGE_NAMES.get(code, code), key="language")
    st.selectbox(_t("settings.mode"), options=["normal", "statistics"], format_func=lambda mode: _t(f"app_mode.{mode}"), key="app_mode")
    apply_mode_switch_reset(st.session_state)

    st.header(_t("data_source.header"))
    st.checkbox(_t("data_source.remember_settings"), key="remember_settings", help=_t("data_source.remember_settings_help"))
    if st.button(_t("data_source.forget_saved_settings"), use_container_width=True):
        clear_session_snapshot()
        st.session_state["_persisted_user_prefs"] = {}
        st.session_state["_persisted_runtime_state"] = {}
        st.session_state["_skip_snapshot_save_once"] = True
        st.success(_t("data_source.saved_settings_removed"))
        st.rerun()
    if st.button(_t("data_source.reset_corrupted_settings"), use_container_width=True):
        reset_corrupted_settings_state(st.session_state, SAMPLE_DF)
        if "table_editor" in st.session_state:
            del st.session_state["table_editor"]
        st.success(_t("data_source.corrupted_settings_reset"))
        st.rerun()

    current_mode = str(st.session_state.get("app_mode", "normal"))
    sample_button_label = _t("data_source.use_statistics_sample_data") if current_mode == "statistics" else _t("data_source.use_normal_sample_data")
    if st.button(sample_button_label, use_container_width=True):
        st.session_state["table_df"] = STATISTICS_SAMPLE_DF.copy() if current_mode == "statistics" else SAMPLE_DF.copy()
        st.session_state["uploaded_signature"] = ""
        if "table_editor" in st.session_state:
            del st.session_state["table_editor"]
        st.rerun()

    if st.button(_t("data_source.reset_view"), use_container_width=True):
        reset_view_state(st.session_state)
        st.rerun()

    uploaded = st.file_uploader(_t("data_source.upload"), type=["csv", "xlsx", "xls", "ods"])
    if uploaded is not None:
        signature = f"{uploaded.name}-{uploaded.size}"
        if signature != st.session_state.get("uploaded_signature", ""):
            try:
                st.session_state["table_df"] = load_table_file(uploaded, uploaded.name)
                st.session_state["uploaded_signature"] = signature
                if "table_editor" in st.session_state:
                    del st.session_state["table_editor"]
                st.success(_t("data_source.loaded_file", filename=uploaded.name))
                st.rerun()
            except Exception as exc:  # pragma: no cover - user-facing file IO boundary
                st.error(_t("data_source.load_failed", error=exc))

if restore_error := str(st.session_state.get("_snapshot_restore_error", "")).strip():
    st.caption(restore_error)
if save_error := str(st.session_state.get("_snapshot_save_error", "")).strip():
    st.caption(save_error)

st.subheader(_t("table.header"))
st.write(_t("table.description"))

edited_df = st.data_editor(
    st.session_state["table_df"],
    num_rows="dynamic",
    use_container_width=True,
    key="table_editor",
)
st.session_state["table_df"] = edited_df

columns = [str(col) for col in edited_df.columns]
if not columns:
    st.error(_t("table.add_column_error"))
    st.stop()

prefs = render_statistics_mode(edited_df, columns, _t) if str(st.session_state.get("app_mode", "normal")) == "statistics" else render_normal_mode(edited_df, columns, _t)
st.session_state["_prefs"] = prefs
save_session_snapshot(st.session_state)


