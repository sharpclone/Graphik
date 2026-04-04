"""Guided import wizard for uploaded tables."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from pages.common import render_problem_list
from services.import_service import (
    apply_import_selection,
    build_import_wizard_result,
    clear_pending_import,
    has_pending_import,
    wizard_key,
)
from services.validation_service import collect_import_wizard_problems, has_blocking_problems
from src.session_types import SessionStateLike
from src.ui_helpers import safe_default_index

TranslateFn = Callable[..., str]


def render_import_wizard(
    session_state: SessionStateLike,
    translate: TranslateFn,
) -> bool:
    """Render the guided import wizard when a new file is waiting."""
    if not has_pending_import(session_state):
        return False

    raw_df = session_state[wizard_key("raw_df")]
    filename = str(session_state.get(wizard_key("filename"), "table"))
    signature = str(session_state.get(wizard_key("signature"), ""))
    current_mode = str(session_state.get("app_mode", "normal"))

    st.subheader(translate("import_wizard.header"))
    st.caption(translate("import_wizard.caption", filename=filename, rows=str(len(raw_df)), cols=str(len(raw_df.columns))))

    raw_col, preview_col = st.columns(2)
    with raw_col:
        st.markdown(f"**{translate('import_wizard.step_raw_preview')}**")
        st.dataframe(raw_df.head(12), use_container_width=True)
    with preview_col:
        st.markdown(f"**{translate('import_wizard.step_header')}**")
        header_mode = st.selectbox(
            translate("import_wizard.header_mode"),
            options=["auto", "current", "row"],
            format_func=lambda value: translate(f"import_wizard.header_mode.{value}"),
            key=wizard_key("header_mode"),
        )
        header_row_index = None
        if header_mode == "row":
            header_row_display = int(
                st.number_input(
                    translate("import_wizard.header_row"),
                    min_value=1,
                    max_value=max(1, len(raw_df)),
                    value=1,
                    step=1,
                    key=wizard_key("header_row"),
                )
            )
            header_row_index = header_row_display - 1
        try:
            wizard_result = build_import_wizard_result(
                raw_df,
                filename=filename,
                signature=signature,
                header_mode=header_mode,
                header_row_index=header_row_index,
            )
            st.dataframe(wizard_result.preview_df.head(12), use_container_width=True)
        except Exception as exc:  # pragma: no cover - user-facing configuration boundary
            st.error(translate("import_wizard.preview_failed", error=exc))
            wizard_result = None

    if wizard_result is None:
        if st.button(translate("import_wizard.cancel"), use_container_width=True, key=wizard_key("cancel_only")):
            clear_pending_import(session_state)
            st.rerun()
        return True

    columns = [str(col) for col in wizard_result.preview_df.columns]
    st.markdown(f"**{translate('import_wizard.step_columns')}**")
    if current_mode == "statistics":
        stats_column = st.selectbox(
            translate("statistics.column"),
            options=columns,
            index=safe_default_index(columns, wizard_result.suggested_stats_column or ""),
            key=wizard_key("stats_column"),
        )
        x_column = None
        y_column = None
        sigma_y_column = None
        use_zero_error = True
    else:
        mapping_cols = st.columns(3)
        with mapping_cols[0]:
            x_column = st.selectbox(
                translate("mapping.x_column"),
                options=columns,
                index=safe_default_index(columns, wizard_result.suggested_x_column or ""),
                key=wizard_key("x_column"),
            )
        with mapping_cols[1]:
            y_column = st.selectbox(
                translate("mapping.y_column"),
                options=columns,
                index=safe_default_index(columns, wizard_result.suggested_y_column or ""),
                key=wizard_key("y_column"),
            )
        with mapping_cols[2]:
            use_zero_error = bool(
                st.checkbox(
                    translate("mapping.no_error_column"),
                    value=False,
                    key=wizard_key("use_zero_error"),
                )
            )
            sigma_y_column = None
            if not use_zero_error:
                sigma_y_column = st.selectbox(
                    translate("mapping.error_column"),
                    options=columns,
                    index=safe_default_index(columns, wizard_result.suggested_sigma_y_column or ""),
                    key=wizard_key("sigma_y_column"),
                )
        stats_column = None

    problems = collect_import_wizard_problems(
        wizard_result.preview_df,
        mode=current_mode,
        x_column=x_column,
        y_column=y_column,
        sigma_y_column=sigma_y_column,
        use_zero_error=use_zero_error,
        stats_column=stats_column,
        translate=translate,
    )
    render_problem_list(problems, translate, title_key="import_wizard.step_validation")

    actions = st.columns([1.2, 1.2, 3.0])
    with actions[0]:
        confirm = st.button(
            translate("import_wizard.confirm"),
            use_container_width=True,
            disabled=has_blocking_problems(problems),
            key=wizard_key("confirm"),
        )
    with actions[1]:
        cancel = st.button(translate("import_wizard.cancel"), use_container_width=True, key=wizard_key("cancel"))

    if cancel:
        clear_pending_import(session_state)
        st.rerun()

    if confirm:
        apply_import_selection(
            session_state,
            imported_df=wizard_result.preview_df,
            signature=signature,
            mode=current_mode,
            x_column=x_column,
            y_column=y_column,
            sigma_y_column=sigma_y_column,
            stats_column=stats_column,
            use_zero_error=use_zero_error,
        )
        clear_pending_import(session_state)
        st.success(translate("import_wizard.confirmed", filename=filename))
        st.rerun()

    return True

