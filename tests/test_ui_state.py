from __future__ import annotations

from pathlib import Path
import pickle

import pandas as pd

from services.state_service import apply_mode_switch_reset
from src.ui_state import (
    MODE_PREFIXES,
    NORMAL_MODE,
    SESSION_SCHEMA_VERSION,
    STATISTICS_MODE,
    clear_session_snapshot,
    reset_corrupted_settings,
    restore_session_snapshot,
    save_session_snapshot,
)


def test_restore_session_snapshot_records_classified_error_for_invalid_user_prefs_json(tmp_path: Path) -> None:
    user_prefs_path = tmp_path / "user_prefs.json"
    runtime_path = tmp_path / "runtime.pkl"
    user_prefs_path.write_text("not-json", encoding="utf-8")
    session_state: dict[str, object] = {}

    restore_session_snapshot(session_state, user_prefs_path=user_prefs_path, runtime_path=runtime_path)

    assert session_state["_snapshot_restored"] is True
    assert "Failed to restore user preferences" in str(session_state["_snapshot_restore_error"])


def test_save_session_snapshot_persists_only_whitelisted_user_prefs_and_runtime(tmp_path: Path) -> None:
    user_prefs_path = tmp_path / "user_prefs.json"
    runtime_path = tmp_path / "runtime.pkl"
    table_df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    session_state: dict[str, object] = {
        "remember_settings": True,
        "language": "en",
        "app_mode": NORMAL_MODE,
        "normal.use_latex_plot": True,
        "normal.show_grid": False,
        "stats.show_grid": True,
        "table_df": table_df,
        "uploaded_signature": "demo.csv-10",
        "stray_runtime_key": "do-not-save",
        "stray_pref_key": 123,
        "_export_cache_normal_png": {"signature": "abc", "data": b"123"},
    }

    save_session_snapshot(session_state, user_prefs_path=user_prefs_path, runtime_path=runtime_path, persist_dir=tmp_path)

    assert user_prefs_path.exists()
    assert runtime_path.exists()

    user_payload = __import__("json").loads(user_prefs_path.read_text(encoding="utf-8"))
    assert user_payload["version"] == SESSION_SCHEMA_VERSION
    assert user_payload["user_prefs"]["language"] == "en"
    assert user_payload["user_prefs"]["normal.use_latex_plot"] is True
    assert user_payload["user_prefs"]["normal.show_grid"] is False
    assert "stray_pref_key" not in user_payload["user_prefs"]

    runtime_payload = pickle.loads(runtime_path.read_bytes())
    assert runtime_payload["version"] == SESSION_SCHEMA_VERSION
    pd.testing.assert_frame_equal(runtime_payload["runtime_state"]["table_df"], table_df)
    assert runtime_payload["runtime_state"]["uploaded_signature"] == "demo.csv-10"
    assert "stray_runtime_key" not in runtime_payload["runtime_state"]


def test_restore_session_snapshot_restores_split_state_files(tmp_path: Path) -> None:
    user_prefs_path = tmp_path / "user_prefs.json"
    runtime_path = tmp_path / "runtime.pkl"
    source_state: dict[str, object] = {
        "remember_settings": True,
        "language": "de",
        "app_mode": STATISTICS_MODE,
        "stats.stats_column": "height",
        "stats.show_grid": True,
        "table_df": pd.DataFrame({"height": [170, 172]}),
        "uploaded_signature": "stats.csv-4",
    }
    save_session_snapshot(source_state, user_prefs_path=user_prefs_path, runtime_path=runtime_path, persist_dir=tmp_path)

    restored: dict[str, object] = {}
    restore_session_snapshot(restored, user_prefs_path=user_prefs_path, runtime_path=runtime_path)

    assert restored["language"] == "de"
    assert restored["app_mode"] == STATISTICS_MODE
    assert restored["stats.stats_column"] == "height"
    assert restored["stats.show_grid"] is True
    pd.testing.assert_frame_equal(restored["table_df"], pd.DataFrame({"height": [170, 172]}))
    assert restored["uploaded_signature"] == "stats.csv-4"


def test_apply_mode_switch_reset_clears_previous_mode_and_rehydrates_current_mode() -> None:
    session_state: dict[str, object] = {
        "app_mode": STATISTICS_MODE,
        "_last_app_mode": NORMAL_MODE,
        "normal.show_grid": False,
        "normal.use_latex_plot": True,
        "stats.show_grid": True,
        "_prefs": {"foo": "bar"},
        "_export_cache_normal_png": {"signature": "sig", "data": b"1"},
        "_persisted_user_prefs": {
            "stats.show_grid": True,
            "stats.use_latex_plot": False,
        },
    }

    changed = apply_mode_switch_reset(session_state)

    assert changed is True
    assert "normal.show_grid" not in session_state
    assert "normal.use_latex_plot" not in session_state
    assert session_state["stats.show_grid"] is True
    assert session_state["stats.use_latex_plot"] is False
    assert "_prefs" not in session_state
    assert "_export_cache_normal_png" not in session_state
    assert session_state["_last_app_mode"] == STATISTICS_MODE


def test_reset_corrupted_settings_clears_live_and_persisted_state(tmp_path: Path) -> None:
    sample_df = pd.DataFrame({"x": [1.0], "y": [2.0]})
    user_prefs_path = tmp_path / "user_prefs.json"
    runtime_path = tmp_path / "runtime.pkl"
    legacy_path = tmp_path / "legacy.pkl"
    user_prefs_path.write_text("{}", encoding="utf-8")
    runtime_path.write_bytes(pickle.dumps({"version": SESSION_SCHEMA_VERSION, "runtime_state": {}}))
    legacy_path.write_bytes(b"legacy")

    clear_session_snapshot(user_prefs_path=user_prefs_path, runtime_path=runtime_path, legacy_path=legacy_path)
    assert not user_prefs_path.exists()
    assert not runtime_path.exists()
    assert not legacy_path.exists()

    session_state: dict[str, object] = {
        "language": "en",
        "app_mode": STATISTICS_MODE,
        "normal.show_grid": False,
        "stats.show_grid": True,
        "table_df": pd.DataFrame({"a": [1]}),
        "uploaded_signature": "abc",
        "_snapshot_restore_error": "bad restore",
        "_snapshot_save_error": "bad save",
    }
    reset_corrupted_settings(session_state, sample_df)

    assert session_state["language"] == "de"
    assert session_state["app_mode"] == NORMAL_MODE
    pd.testing.assert_frame_equal(session_state["table_df"], sample_df)
    assert session_state["uploaded_signature"] == ""
    assert session_state["_snapshot_restore_error"] == ""
    assert session_state["_snapshot_save_error"] == ""
    assert f"{MODE_PREFIXES[NORMAL_MODE]}show_grid" not in session_state
