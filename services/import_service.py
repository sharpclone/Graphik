"""Import-wizard helpers for uploaded tables."""

from __future__ import annotations

import pandas as pd

from services.validation_service import suggest_column_mapping
from src.data_io import apply_header_strategy
from src.session_types import SessionStateLike
from src.ux_models import ImportWizardResult

IMPORT_WIZARD_PREFIX = "_import_wizard."


def wizard_key(name: str) -> str:
    """Return the session-state key for one import-wizard field."""
    return f"{IMPORT_WIZARD_PREFIX}{name}"


def set_pending_import(
    session_state: SessionStateLike,
    *,
    raw_df: pd.DataFrame,
    filename: str,
    signature: str,
) -> None:
    """Store a newly uploaded raw table for the guided import flow."""
    session_state[wizard_key("raw_df")] = raw_df.copy()
    session_state[wizard_key("filename")] = filename
    session_state[wizard_key("signature")] = signature


def clear_pending_import(session_state: SessionStateLike) -> None:
    """Clear the active import wizard state from the current session."""
    for key in list(session_state.keys()):
        if str(key).startswith(IMPORT_WIZARD_PREFIX):
            session_state.pop(key, None)


def has_pending_import(session_state: SessionStateLike) -> bool:
    """Return True when an uploaded file is waiting in the import wizard."""
    return wizard_key("raw_df") in session_state and isinstance(session_state.get(wizard_key("raw_df")), pd.DataFrame)


def build_import_wizard_result(
    raw_df: pd.DataFrame,
    *,
    filename: str,
    signature: str,
    header_mode: str,
    header_row_index: int | None,
) -> ImportWizardResult:
    """Build the wizard preview dataframe and suggested mappings."""
    preview_df = apply_header_strategy(raw_df, header_mode=header_mode, header_row_index=header_row_index)
    columns = [str(col) for col in preview_df.columns]
    x_col, y_col, sigma_col, stats_col = suggest_column_mapping(columns)
    return ImportWizardResult(
        raw_df=raw_df.copy(),
        preview_df=preview_df,
        filename=filename,
        signature=signature,
        header_mode=header_mode,
        header_row_index=header_row_index,
        suggested_x_column=x_col,
        suggested_y_column=y_col,
        suggested_sigma_y_column=sigma_col,
        suggested_stats_column=stats_col,
    )


def apply_import_selection(
    session_state: SessionStateLike,
    *,
    imported_df: pd.DataFrame,
    signature: str,
    mode: str,
    x_column: str | None,
    y_column: str | None,
    sigma_y_column: str | None,
    stats_column: str | None,
    use_zero_error: bool,
) -> None:
    """Commit the wizard result into the live table and seed relevant controls."""
    session_state["table_df"] = imported_df.copy()
    session_state["uploaded_signature"] = signature
    session_state.pop("table_editor", None)

    if mode == "statistics":
        if stats_column:
            session_state["stats.stats_column"] = stats_column
        return

    if x_column:
        session_state["normal.x_col_simple"] = x_column
        session_state["normal.x_col_adv"] = x_column
    if y_column:
        session_state["normal.y_col_simple"] = y_column
        session_state["normal.y_col_adv"] = y_column
    session_state["normal.use_zero_error"] = bool(use_zero_error)
    if sigma_y_column and not use_zero_error:
        session_state["normal.sigma_col_simple"] = sigma_y_column
        session_state["normal.sigma_y_col_adv"] = sigma_y_column

