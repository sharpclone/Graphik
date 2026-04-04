# Internal Refactor Changelog

## 2026-04-04

### Export architecture
- Unified export flow behind `src/export_pipeline.py` with shared `ExportConfig` and `ExportRequest`.
- Switched image export backend to Matplotlib while keeping Plotly for interactive preview.
- Added export validation, structured debug reports, parity tests, and unsupported-feature fail-fast behavior.

### Application structure
- Split the old monolithic `app.py` into mode controllers and service layers:
  - `pages/normal_mode.py`
  - `pages/statistics_mode.py`
  - `services/analysis_service.py`
  - `services/export_service.py`
  - `services/state_service.py`
- Introduced typed mode/result models in `src/mode_models.py`.

### State management
- Replaced broad session snapshot persistence with explicit whitelists.
- Split persisted data into `user_prefs.json` and `last_runtime_state.pkl`.
- Namespaced mode-specific keys under `normal.*` and `stats.*`.
- Added explicit reset flow for corrupted settings.

### UX
- Added an import wizard with header detection, guided mapping, validation, and explicit confirmation.
- Added consolidated problem lists for normal/statistics modes.
- Added export presets and downloadable reproducible export preset JSON.
- Added clean export preview and explicit info-box status messages.

### Quality gates
- Added structured logging helpers in `src/logging_utils.py`.
- Added CI configuration with Ruff, Black, isort, mypy, function-length checks, and pytest.
- Added internal function-length guard script in `scripts/check_function_lengths.py`.

### Type safety and quality tooling
- Added strict source typing coverage with `mypy` and `pyright` config plus passing source checks.
- Added structured runtime logging with export/state context in `src/logging_utils.py`.
- Added internal function metrics report with line/branch baselines in `scripts/check_function_lengths.py`.
- Added extra persistence regression tests for schema mismatch, corrupt runtime state, and transient-key exclusions.
