"""Session-state persistence helpers for the Streamlit app."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any, MutableMapping

import pandas as pd


PERSIST_DIR = Path(".streamlit")
SESSION_STATE_PATH = PERSIST_DIR / "last_session_state.pkl"
SESSION_SCHEMA_VERSION = 1
NON_ASSIGNABLE_WIDGET_KEYS = frozenset({"table_editor"})


def restore_session_snapshot(
    session_state: MutableMapping[str, Any],
    snapshot_path: Path = SESSION_STATE_PATH,
    schema_version: int = SESSION_SCHEMA_VERSION,
    non_assignable_widget_keys: frozenset[str] = NON_ASSIGNABLE_WIDGET_KEYS,
) -> None:
    """Restore picklable UI state from disk once per process run."""
    if session_state.get("_snapshot_restored", False):
        return
    if not snapshot_path.exists():
        session_state["_snapshot_restored"] = True
        return

    try:
        with snapshot_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception:
        session_state["_snapshot_restored"] = True
        return

    if not isinstance(payload, dict):
        session_state["_snapshot_restored"] = True
        return

    version = payload.get("version", 0)
    state = payload.get("state", {})
    if version != schema_version or not isinstance(state, dict):
        session_state["_snapshot_restored"] = True
        return

    prefs = state.get("_prefs", {})
    if isinstance(prefs, dict):
        for key, value in prefs.items():
            if key in non_assignable_widget_keys:
                continue
            if key not in session_state:
                session_state[key] = value

    for key, value in state.items():
        if key == "_prefs":
            continue
        if key in non_assignable_widget_keys:
            continue
        if key not in session_state:
            session_state[key] = value

    session_state["_snapshot_restored"] = True


def save_session_snapshot(
    session_state: MutableMapping[str, Any],
    snapshot_path: Path = SESSION_STATE_PATH,
    persist_dir: Path = PERSIST_DIR,
    non_assignable_widget_keys: frozenset[str] = NON_ASSIGNABLE_WIDGET_KEYS,
) -> None:
    """Persist picklable Streamlit session-state values to disk."""
    if not session_state.get("remember_settings", True):
        if snapshot_path.exists():
            snapshot_path.unlink()
        return

    excluded_keys = {"_snapshot_restored"} | set(non_assignable_widget_keys)
    transient_prefixes = ("_export_cache_",)
    snapshot: dict[str, Any] = {}
    for key, value in session_state.items():
        if key in excluded_keys or key.startswith(transient_prefixes):
            continue
        try:
            pickle.dumps(value)
        except Exception:
            continue
        snapshot[key] = value

    try:
        persist_dir.mkdir(parents=True, exist_ok=True)
        with snapshot_path.open("wb") as handle:
            pickle.dump(
                {"version": SESSION_SCHEMA_VERSION, "state": snapshot},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    except Exception:
        # Persistence is best-effort and should never break plotting flow.
        pass


def clear_session_snapshot(snapshot_path: Path = SESSION_STATE_PATH) -> None:
    """Delete persisted session-state snapshot from disk."""
    if snapshot_path.exists():
        snapshot_path.unlink()


def init_session_state(
    session_state: MutableMapping[str, Any],
    sample_dataframe: pd.DataFrame,
) -> None:
    """Initialize required session-state keys once."""
    if "language" not in session_state:
        session_state["language"] = "de"
    if "app_mode" not in session_state:
        session_state["app_mode"] = "normal"
    if "remember_settings" not in session_state:
        session_state["remember_settings"] = True
    if "table_df" not in session_state:
        session_state["table_df"] = sample_dataframe.copy()
    if "uploaded_signature" not in session_state:
        session_state["uploaded_signature"] = ""
    if "custom_x_range" not in session_state:
        session_state["custom_x_range"] = False
    if "custom_y_range" not in session_state:
        session_state["custom_y_range"] = False
    if "show_error_triangles" not in session_state:
        session_state["show_error_triangles"] = True


def reset_view_state(session_state: MutableMapping[str, Any]) -> None:
    """Reset only the current plot-view toggles."""
    session_state["custom_x_range"] = False
    session_state["custom_y_range"] = False
    session_state["show_error_triangles"] = True
