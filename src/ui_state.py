"""Session-state persistence helpers for the Streamlit app."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
import pickle
from typing import Any, MutableMapping

import pandas as pd

from .errors import StateRestoreError, StateSaveError


PERSIST_DIR = Path(".streamlit")
USER_PREFS_PATH = PERSIST_DIR / "user_prefs.json"
RUNTIME_STATE_PATH = PERSIST_DIR / "last_runtime_state.pkl"
LEGACY_SESSION_STATE_PATH = PERSIST_DIR / "last_session_state.pkl"
SESSION_SCHEMA_VERSION = 2
NON_ASSIGNABLE_WIDGET_KEYS = frozenset({"table_editor"})
GLOBAL_USER_PREF_KEYS = frozenset({"language", "app_mode", "remember_settings"})
GLOBAL_RUNTIME_STATE_KEYS = frozenset({"table_df", "uploaded_signature"})
NORMAL_MODE = "normal"
STATISTICS_MODE = "statistics"
MODE_PREFIXES = {
    NORMAL_MODE: "normal.",
    STATISTICS_MODE: "stats.",
}


NORMAL_USER_PREF_NAMES = frozenset(
    {
        "mapping_mode",
        "use_zero_error",
        "x_col_simple",
        "y_col_simple",
        "sigma_col_simple",
        "x_col_adv",
        "y_col_adv",
        "sigma_y_col_adv",
        "use_latex_plot",
        "auto_axis_labels",
        "x_label_input",
        "y_label_input",
        "y_scale_mode",
        "log_decade_mode",
        "use_separate_fonts",
        "global_font_size",
        "base_font_size",
        "axis_title_font_size",
        "tick_font_size",
        "annotation_font_size",
        "plot_info_box_manual",
        "plot_info_box_font_size",
        "plot_info_box_x",
        "plot_info_box_y",
        "plot_info_box_width",
        "plot_info_box_height",
        "show_grid",
        "grid_mode",
        "x_major_divisions",
        "y_major_divisions",
        "minor_per_major",
        "marker_size",
        "error_bar_thickness",
        "error_bar_cap_width",
        "connect_points",
        "x_tick_decimals",
        "y_tick_decimals",
        "custom_x_range",
        "custom_y_range",
        "x_min_user",
        "x_max_user",
        "y_min_user",
        "y_max_user",
        "fit_model",
        "show_fit_line",
        "show_error_lines",
        "error_line_mode_widget",
        "auto_line_labels",
        "fit_label_input",
        "fit_color",
        "show_fit_slope_label",
        "show_line_equations_on_plot",
        "show_r2_on_plot",
        "visible_error_lines_exp",
        "visible_error_lines_linear",
        "error_color",
        "error_label_max_input",
        "error_label_min_input",
        "show_error_slope_label",
        "show_fit_triangle",
        "auto_fit_points",
        "fit_custom_ax",
        "fit_custom_bx",
        "show_error_triangles",
        "triangle_x_decimals",
        "triangle_y_decimals",
        "export_base",
        "png_paper",
        "png_orientation",
        "png_dpi",
        "png_scale",
        "png_word_like",
        "png_text_pt",
        "png_visual_scale",
        "autoscale_export_axes",
    }
)
STATISTICS_USER_PREF_NAMES = frozenset(
    {
        "stats_column",
        "stats_bins",
        "stats_normalize_density",
        "use_latex_plot",
        "auto_axis_labels",
        "x_label_input",
        "y_label_input",
        "use_separate_fonts",
        "global_font_size",
        "base_font_size",
        "axis_title_font_size",
        "tick_font_size",
        "annotation_font_size",
        "plot_info_box_manual",
        "plot_info_box_font_size",
        "plot_info_box_x",
        "plot_info_box_y",
        "plot_info_box_width",
        "plot_info_box_height",
        "show_grid",
        "x_tick_decimals",
        "y_tick_decimals",
        "show_normal_fit",
        "show_formula_box",
        "show_mean_line",
        "show_std_lines",
        "show_two_sigma",
        "show_three_sigma",
        "histogram_color",
        "fit_color",
        "mean_color",
        "std_color",
        "export_base",
        "png_paper",
        "png_orientation",
        "png_dpi",
        "png_scale",
        "png_word_like",
        "png_text_pt",
        "png_visual_scale",
    }
)


def _namespaced_keys(prefix: str, names: Iterable[str]) -> frozenset[str]:
    return frozenset(f"{prefix}{name}" for name in names)


NORMAL_USER_PREF_KEYS = _namespaced_keys(MODE_PREFIXES[NORMAL_MODE], NORMAL_USER_PREF_NAMES)
STATISTICS_USER_PREF_KEYS = _namespaced_keys(MODE_PREFIXES[STATISTICS_MODE], STATISTICS_USER_PREF_NAMES)
MODE_USER_PREF_KEYS = {
    NORMAL_MODE: NORMAL_USER_PREF_KEYS,
    STATISTICS_MODE: STATISTICS_USER_PREF_KEYS,
}
ALL_USER_PREF_KEYS = frozenset().union(GLOBAL_USER_PREF_KEYS, NORMAL_USER_PREF_KEYS, STATISTICS_USER_PREF_KEYS)
TRANSIENT_STATE_PREFIXES = ("_export_cache_",)
TRANSIENT_STATE_KEYS = frozenset(
    {
        "_snapshot_restored",
        "_snapshot_restore_error",
        "_snapshot_save_error",
        "_prefs",
        "_persisted_user_prefs",
        "_persisted_runtime_state",
        "_last_app_mode",
        "_skip_snapshot_save_once",
    }
)


def _record_state_error(session_state: MutableMapping[str, Any], *, bucket: str, exc: StateRestoreError | StateSaveError) -> None:
    existing = str(session_state.get(bucket, "")).strip()
    message = str(exc)
    if existing:
        session_state[bucket] = f"{existing}\n{message}"
    else:
        session_state[bucket] = message


def _load_runtime_payload(runtime_path: Path) -> dict[str, Any]:
    try:
        with runtime_path.open("rb") as handle:
            payload = pickle.load(handle)
    except (pickle.UnpicklingError, EOFError, OSError, AttributeError, ValueError, TypeError) as exc:
        raise StateRestoreError(f"Failed to restore runtime state from {runtime_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StateRestoreError(f"Saved runtime state in {runtime_path} is not a valid payload dictionary.")
    return payload


def _load_user_prefs_payload(user_prefs_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(user_prefs_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise StateRestoreError(f"Failed to restore user preferences from {user_prefs_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StateRestoreError(f"Saved user preferences in {user_prefs_path} are not a valid payload dictionary.")
    return payload


def _validated_state_section(payload: dict[str, Any], *, section_name: str, schema_version: int, source_path: Path) -> dict[str, Any]:
    version = payload.get("version", 0)
    state = payload.get(section_name, {})
    if version != schema_version or not isinstance(state, dict):
        raise StateRestoreError(
            f"Saved state schema mismatch in {source_path}: version={version!r}, expected={schema_version!r}."
        )
    return state


def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, set):
        return [_normalize_json_value(item) for item in sorted(value, key=str)]
    if isinstance(value, dict):
        return {str(key): _normalize_json_value(inner_value) for key, inner_value in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise StateSaveError(f"User preference value of type {type(value).__name__} is not JSON-serializable.")


def _restore_values(
    session_state: MutableMapping[str, Any],
    values: dict[str, Any],
    *,
    allowed_keys: frozenset[str],
    non_assignable_widget_keys: frozenset[str],
) -> None:
    for key, value in values.items():
        if key not in allowed_keys or key in non_assignable_widget_keys:
            continue
        if key not in session_state:
            session_state[key] = value


def _write_runtime_snapshot(runtime_path: Path, payload: dict[str, Any], persist_dir: Path) -> None:
    try:
        persist_dir.mkdir(parents=True, exist_ok=True)
        with runtime_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except (pickle.PickleError, OSError, AttributeError, TypeError, ValueError) as exc:
        raise StateSaveError(f"Failed to save runtime state to {runtime_path}: {exc}") from exc


def _write_user_prefs_snapshot(user_prefs_path: Path, payload: dict[str, Any], persist_dir: Path) -> None:
    try:
        persist_dir.mkdir(parents=True, exist_ok=True)
        user_prefs_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except (OSError, TypeError, ValueError) as exc:
        raise StateSaveError(f"Failed to save user preferences to {user_prefs_path}: {exc}") from exc


def restore_session_snapshot(
    session_state: MutableMapping[str, Any],
    *,
    user_prefs_path: Path = USER_PREFS_PATH,
    runtime_path: Path = RUNTIME_STATE_PATH,
    schema_version: int = SESSION_SCHEMA_VERSION,
    non_assignable_widget_keys: frozenset[str] = NON_ASSIGNABLE_WIDGET_KEYS,
) -> None:
    """Restore persisted user preferences and runtime state once per process run."""
    if session_state.get("_snapshot_restored", False):
        return

    session_state["_snapshot_restore_error"] = ""
    restored_user_prefs: dict[str, Any] = {}
    restored_runtime_state: dict[str, Any] = {}

    if user_prefs_path.exists():
        try:
            user_payload = _load_user_prefs_payload(user_prefs_path)
            restored_user_prefs = _validated_state_section(
                user_payload,
                section_name="user_prefs",
                schema_version=schema_version,
                source_path=user_prefs_path,
            )
        except StateRestoreError as exc:
            _record_state_error(session_state, bucket="_snapshot_restore_error", exc=exc)

    if runtime_path.exists():
        try:
            runtime_payload = _load_runtime_payload(runtime_path)
            restored_runtime_state = _validated_state_section(
                runtime_payload,
                section_name="runtime_state",
                schema_version=schema_version,
                source_path=runtime_path,
            )
        except StateRestoreError as exc:
            _record_state_error(session_state, bucket="_snapshot_restore_error", exc=exc)

    _restore_values(
        session_state,
        restored_user_prefs,
        allowed_keys=ALL_USER_PREF_KEYS,
        non_assignable_widget_keys=non_assignable_widget_keys,
    )
    _restore_values(
        session_state,
        restored_runtime_state,
        allowed_keys=GLOBAL_RUNTIME_STATE_KEYS,
        non_assignable_widget_keys=non_assignable_widget_keys,
    )
    session_state["_persisted_user_prefs"] = dict(restored_user_prefs)
    session_state["_persisted_runtime_state"] = dict(restored_runtime_state)
    session_state["_snapshot_restored"] = True


def _collect_user_prefs(session_state: MutableMapping[str, Any]) -> dict[str, Any]:
    persisted = session_state.get("_persisted_user_prefs", {})
    merged: dict[str, Any] = {
        key: _normalize_json_value(value)
        for key, value in persisted.items()
        if key in ALL_USER_PREF_KEYS
    } if isinstance(persisted, dict) else {}

    for key in ALL_USER_PREF_KEYS:
        if key not in session_state:
            continue
        merged[key] = _normalize_json_value(session_state[key])
    return merged


def _collect_runtime_state(session_state: MutableMapping[str, Any]) -> dict[str, Any]:
    runtime_state: dict[str, Any] = {}
    for key in GLOBAL_RUNTIME_STATE_KEYS:
        if key in session_state:
            runtime_state[key] = session_state[key]
    return runtime_state


def _remove_persisted_files(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def save_session_snapshot(
    session_state: MutableMapping[str, Any],
    *,
    user_prefs_path: Path = USER_PREFS_PATH,
    runtime_path: Path = RUNTIME_STATE_PATH,
    persist_dir: Path = PERSIST_DIR,
    non_assignable_widget_keys: frozenset[str] = NON_ASSIGNABLE_WIDGET_KEYS,
) -> None:
    """Persist explicit user preferences and lightweight runtime state to disk."""
    del non_assignable_widget_keys  # explicit whitelist persistence no longer needs widget blacklist.
    session_state["_snapshot_save_error"] = ""

    if session_state.pop("_skip_snapshot_save_once", False):
        return

    if not session_state.get("remember_settings", True):
        try:
            _remove_persisted_files(user_prefs_path, runtime_path, LEGACY_SESSION_STATE_PATH)
        except OSError as exc:
            _record_state_error(
                session_state,
                bucket="_snapshot_save_error",
                exc=StateSaveError(f"Failed to remove saved state files: {exc}"),
            )
        return

    try:
        user_prefs = _collect_user_prefs(session_state)
        runtime_state = _collect_runtime_state(session_state)
        _write_user_prefs_snapshot(
            user_prefs_path,
            {"version": SESSION_SCHEMA_VERSION, "user_prefs": user_prefs},
            persist_dir,
        )
        _write_runtime_snapshot(
            runtime_path,
            {"version": SESSION_SCHEMA_VERSION, "runtime_state": runtime_state},
            persist_dir,
        )
        session_state["_persisted_user_prefs"] = dict(user_prefs)
        session_state["_persisted_runtime_state"] = dict(runtime_state)
        if LEGACY_SESSION_STATE_PATH.exists():
            LEGACY_SESSION_STATE_PATH.unlink()
    except StateSaveError as exc:
        _record_state_error(session_state, bucket="_snapshot_save_error", exc=exc)


def clear_session_snapshot(
    *,
    user_prefs_path: Path = USER_PREFS_PATH,
    runtime_path: Path = RUNTIME_STATE_PATH,
    legacy_path: Path = LEGACY_SESSION_STATE_PATH,
) -> None:
    """Delete persisted user preferences and runtime snapshots from disk."""
    _remove_persisted_files(user_prefs_path, runtime_path, legacy_path)


def init_session_state(
    session_state: MutableMapping[str, Any],
    sample_dataframe: pd.DataFrame,
) -> None:
    """Initialize required session-state keys once."""
    if "language" not in session_state:
        session_state["language"] = "de"
    if "app_mode" not in session_state:
        session_state["app_mode"] = NORMAL_MODE
    if "remember_settings" not in session_state:
        session_state["remember_settings"] = True
    if "table_df" not in session_state:
        session_state["table_df"] = sample_dataframe.copy()
    if "uploaded_signature" not in session_state:
        session_state["uploaded_signature"] = ""
    if "_persisted_user_prefs" not in session_state:
        session_state["_persisted_user_prefs"] = {}
    if "_persisted_runtime_state" not in session_state:
        session_state["_persisted_runtime_state"] = {}
    if "_last_app_mode" not in session_state:
        session_state["_last_app_mode"] = str(session_state.get("app_mode", NORMAL_MODE))


def reset_view_state(session_state: MutableMapping[str, Any]) -> None:
    """Reset only the active plot-view toggles for the current mode."""
    active_mode = str(session_state.get("app_mode", NORMAL_MODE))
    if active_mode == NORMAL_MODE:
        session_state[f"{MODE_PREFIXES[NORMAL_MODE]}custom_x_range"] = False
        session_state[f"{MODE_PREFIXES[NORMAL_MODE]}custom_y_range"] = False
        session_state[f"{MODE_PREFIXES[NORMAL_MODE]}show_error_triangles"] = True


def clear_mode_state(session_state: MutableMapping[str, Any], mode: str) -> None:
    """Clear live widget values belonging to one mode namespace."""
    for key in MODE_USER_PREF_KEYS.get(mode, frozenset()):
        session_state.pop(key, None)


def hydrate_mode_preferences(session_state: MutableMapping[str, Any], mode: str) -> None:
    """Restore saved preferences for a mode into the live session if missing."""
    persisted = session_state.get("_persisted_user_prefs", {})
    if not isinstance(persisted, dict):
        return
    _restore_values(
        session_state,
        persisted,
        allowed_keys=MODE_USER_PREF_KEYS.get(mode, frozenset()),
        non_assignable_widget_keys=NON_ASSIGNABLE_WIDGET_KEYS,
    )


def reset_corrupted_settings(session_state: MutableMapping[str, Any], sample_dataframe: pd.DataFrame) -> None:
    """Clear persisted state and return the live session to a known-clean baseline."""
    clear_session_snapshot()
    for key in list(session_state.keys()):
        if key in ALL_USER_PREF_KEYS or key in GLOBAL_RUNTIME_STATE_KEYS or key in TRANSIENT_STATE_KEYS:
            session_state.pop(key, None)
            continue
        if key.startswith(TRANSIENT_STATE_PREFIXES):
            session_state.pop(key, None)
    init_session_state(session_state, sample_dataframe)
    session_state["table_df"] = sample_dataframe.copy()
    session_state["uploaded_signature"] = ""
    session_state["_snapshot_restore_error"] = ""
    session_state["_snapshot_save_error"] = ""
    session_state["_snapshot_restored"] = True




