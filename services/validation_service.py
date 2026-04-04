"""Validation and UX problem-list helpers for plotting modes."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from src.mode_models import NormalModeConfig, PlotInfoBoxConfig, StatisticsModeConfig
from src.ux_models import ProblemItem

TranslateFn = Callable[..., str]


NORMAL_X_HINTS = ("x", "m", "mass", "dicke", "thickness", "time", "t")
NORMAL_Y_HINTS = ("y", "auslenkung", "intensity", "imp", "count", "T", "value")
NORMAL_SIGMA_HINTS = ("sigma", "err", "error", "unc", "delta")


def suggest_column_mapping(columns: Sequence[str]) -> tuple[str | None, str | None, str | None, str | None]:
    """Suggest x/y/error/statistics columns from available names."""

    def _match(hints: Sequence[str], *, exclude: set[str] | None = None) -> str | None:
        exclude = exclude or set()
        lowered = {col: str(col).strip().lower() for col in columns}
        for hint in hints:
            for col, normalized in lowered.items():
                if col in exclude:
                    continue
                if normalized == hint or normalized.startswith(f"{hint} ") or hint in normalized:
                    return col
        for col in columns:
            if col not in exclude:
                return col
        return None

    x_col = _match(NORMAL_X_HINTS)
    y_col = _match(NORMAL_Y_HINTS, exclude={x_col} if x_col else set())
    sigma_col = _match(NORMAL_SIGMA_HINTS, exclude={value for value in (x_col, y_col) if value})
    stats_col = _match(("measurement", "value", "values", "sample", "x", "y", "m"))
    return x_col, y_col, sigma_col, stats_col


def has_blocking_problems(problems: Sequence[ProblemItem]) -> bool:
    """Return True when the problem list contains a blocking issue."""
    return any(problem.blocking for problem in problems)


def _problem(
    severity: str,
    title: str,
    detail: str,
    *,
    code: str,
    blocking: bool = False,
    related_fields: Sequence[str] = (),
) -> ProblemItem:
    return ProblemItem(
        severity=severity,
        title=title,
        detail=detail,
        code=code,
        blocking=blocking,
        related_fields=tuple(related_fields),
    )


def _column_numeric_diagnostics(df: pd.DataFrame, column_name: str) -> tuple[pd.Series | None, list[int], bool]:
    if column_name not in df.columns:
        return None, [], True
    series = df[column_name]
    numeric = pd.to_numeric(series, errors="coerce")
    invalid_mask = numeric.isna() & series.notna() & series.astype(str).str.strip().ne("")
    invalid_rows = [int(index) + 1 for index in series.index[invalid_mask].tolist()]
    empty_column = bool(series.dropna().astype(str).str.strip().eq("").all()) if not series.dropna().empty else True
    return numeric, invalid_rows, empty_column


def collect_normal_mode_problems(
    source_df: pd.DataFrame,
    config: NormalModeConfig,
    *,
    translate: TranslateFn,
) -> tuple[ProblemItem, ...]:
    """Collect user-facing validation problems for normal plotting mode."""
    problems: list[ProblemItem] = []
    required = [(config.x_column, "x"), (config.y_column, "y")]
    if not config.use_zero_error:
        required.append((config.sigma_y_column, "sigma_y"))

    diagnostics: dict[str, pd.Series] = {}
    for column_name, field_name in required:
        numeric, invalid_rows, empty_column = _column_numeric_diagnostics(source_df, column_name)
        if numeric is None:
            problems.append(
                _problem(
                    "error",
                    translate("validation.missing_column_title", column=column_name),
                    translate("validation.missing_column_detail", column=column_name),
                    code=f"missing_{field_name}",
                    blocking=True,
                    related_fields=(field_name,),
                )
            )
            continue
        diagnostics[field_name] = numeric
        if empty_column:
            problems.append(
                _problem(
                    "error",
                    translate("validation.empty_column_title", column=column_name),
                    translate("validation.empty_column_detail", column=column_name),
                    code=f"empty_{field_name}",
                    blocking=True,
                    related_fields=(field_name,),
                )
            )
        if invalid_rows:
            preview_rows = ", ".join(str(value) for value in invalid_rows[:6])
            more_suffix = "..." if len(invalid_rows) > 6 else ""
            problems.append(
                _problem(
                    "warning",
                    translate("validation.non_numeric_rows_title", column=column_name),
                    translate(
                        "validation.non_numeric_rows_detail",
                        count=str(len(invalid_rows)),
                        rows=f"{preview_rows}{more_suffix}",
                    ),
                    code=f"invalid_{field_name}",
                    related_fields=(field_name,),
                )
            )

    if config.use_zero_error:
        diagnostics["sigma_y"] = pd.Series(np.zeros(len(source_df), dtype=float), index=source_df.index)
    elif config.sigma_y_column not in source_df.columns:
        problems.append(
            _problem(
                "error",
                translate("validation.missing_sigma_title"),
                translate("validation.missing_sigma_detail"),
                code="missing_sigma_y",
                blocking=True,
                related_fields=("sigma_y",),
            )
        )

    if not {"x", "y", "sigma_y"}.issubset(diagnostics):
        return tuple(problems)

    complete_mask = diagnostics["x"].notna() & diagnostics["y"].notna() & diagnostics["sigma_y"].notna()
    complete_count = int(complete_mask.sum())
    if complete_count < 2:
        problems.append(
            _problem(
                "error",
                translate("validation.too_few_points_title"),
                translate("validation.too_few_points_detail", count=str(complete_count)),
                code="too_few_points",
                blocking=True,
            )
        )
        return tuple(problems)

    y_values = diagnostics["y"][complete_mask].to_numpy(dtype=float)
    sigma_values = diagnostics["sigma_y"][complete_mask].to_numpy(dtype=float)

    if np.any(sigma_values < 0):
        problems.append(
            _problem(
                "error",
                translate("validation.negative_sigma_title"),
                translate("validation.negative_sigma_detail"),
                code="negative_sigma",
                blocking=True,
                related_fields=("sigma_y",),
            )
        )

    if config.y_axis_type == "log" and np.any(y_values <= 0):
        problems.append(
            _problem(
                "error",
                translate("validation.log_axis_title"),
                translate("validation.log_axis_detail"),
                code="log_axis_non_positive",
                blocking=True,
                related_fields=("y",),
            )
        )

    if config.fit_model == "exp" and np.any(y_values <= 0):
        problems.append(
            _problem(
                "warning",
                translate("validation.exp_unavailable_title"),
                translate("validation.exp_unavailable_detail"),
                code="exp_unavailable",
                blocking=True,
                related_fields=("y",),
            )
        )

    if config.y_axis_type == "log" and np.any((y_values - sigma_values) <= 0):
        problems.append(
            _problem(
                "warning",
                translate("validation.log_error_clip_title"),
                translate("validation.log_error_clip_detail"),
                code="log_error_clip",
                related_fields=("y", "sigma_y"),
            )
        )

    return tuple(problems)




def collect_import_wizard_problems(
    preview_df: pd.DataFrame,
    *,
    mode: str,
    x_column: str | None,
    y_column: str | None,
    sigma_y_column: str | None,
    use_zero_error: bool,
    stats_column: str | None,
    translate: TranslateFn,
) -> tuple[ProblemItem, ...]:
    """Collect simplified validation messages for the guided import flow."""
    problems: list[ProblemItem] = []
    if mode == "statistics":
        if not stats_column:
            problems.append(
                _problem(
                    "error",
                    translate("validation.missing_column_title", column="statistics"),
                    translate("validation.import_choose_stats_detail"),
                    code="wizard_missing_stats",
                    blocking=True,
                    related_fields=("stats_column",),
                )
            )
            return tuple(problems)
        temp_config = StatisticsModeConfig(
            stats_column=stats_column,
            bins=16,
            normalize_density=False,
            use_math_text=True,
            auto_axis_labels=True,
            x_label=stats_column,
            y_label="Count",
            base_font_size=14,
            axis_title_font_size=14,
            tick_font_size=12,
            annotation_font_size=12,
            plot_info_box=PlotInfoBoxConfig(),
            show_grid=True,
            x_tick_decimals=2,
            y_tick_decimals=2,
            show_normal_fit=True,
            show_formula_box=True,
            show_mean_line=True,
            show_std_lines=True,
            show_two_sigma=False,
            show_three_sigma=False,
            histogram_color="#7aa6ff",
            fit_color="#d62728",
            mean_color="#222222",
            std_color="#2ca02c",
        )
        return collect_statistics_mode_problems(preview_df, temp_config, translate=translate)

    missing_columns = [label for label, value in (("x", x_column), ("y", y_column)) if not value]
    if missing_columns:
        problems.append(
            _problem(
                "error",
                translate("validation.import_choose_columns_title"),
                translate("validation.import_choose_columns_detail", columns=", ".join(missing_columns)),
                code="wizard_missing_required_columns",
                blocking=True,
                related_fields=tuple(missing_columns),
            )
        )
        return tuple(problems)
    if not use_zero_error and not sigma_y_column:
        problems.append(
            _problem(
                "error",
                translate("validation.missing_sigma_title"),
                translate("validation.import_choose_sigma_detail"),
                code="wizard_missing_sigma",
                blocking=True,
                related_fields=("sigma_y",),
            )
        )
    required_columns = [value for value in (x_column, y_column) if value]
    if sigma_y_column and not use_zero_error:
        required_columns.append(sigma_y_column)
    for column_name in required_columns:
        numeric, invalid_rows, empty_column = _column_numeric_diagnostics(preview_df, column_name)
        if numeric is None:
            problems.append(
                _problem(
                    "error",
                    translate("validation.missing_column_title", column=column_name),
                    translate("validation.missing_column_detail", column=column_name),
                    code=f"wizard_missing_{column_name}",
                    blocking=True,
                )
            )
            continue
        if empty_column:
            problems.append(
                _problem(
                    "error",
                    translate("validation.empty_column_title", column=column_name),
                    translate("validation.empty_column_detail", column=column_name),
                    code=f"wizard_empty_{column_name}",
                    blocking=True,
                )
            )
        if invalid_rows:
            preview_rows = ", ".join(str(value) for value in invalid_rows[:6])
            more_suffix = "..." if len(invalid_rows) > 6 else ""
            problems.append(
                _problem(
                    "warning",
                    translate("validation.non_numeric_rows_title", column=column_name),
                    translate(
                        "validation.non_numeric_rows_detail",
                        count=str(len(invalid_rows)),
                        rows=f"{preview_rows}{more_suffix}",
                    ),
                    code=f"wizard_invalid_{column_name}",
                )
            )
    return tuple(problems)

def collect_statistics_mode_problems(
    source_df: pd.DataFrame,
    config: StatisticsModeConfig,
    *,
    translate: TranslateFn,
) -> tuple[ProblemItem, ...]:
    """Collect user-facing validation and availability problems for statistics mode."""
    problems: list[ProblemItem] = []
    numeric, invalid_rows, empty_column = _column_numeric_diagnostics(source_df, config.stats_column)
    if numeric is None:
        return (
            _problem(
                "error",
                translate("validation.missing_column_title", column=config.stats_column),
                translate("validation.missing_column_detail", column=config.stats_column),
                code="missing_stats_column",
                blocking=True,
                related_fields=("stats_column",),
            ),
        )

    if empty_column:
        problems.append(
            _problem(
                "error",
                translate("validation.empty_column_title", column=config.stats_column),
                translate("validation.empty_column_detail", column=config.stats_column),
                code="empty_stats_column",
                blocking=True,
                related_fields=("stats_column",),
            )
        )

    if invalid_rows:
        preview_rows = ", ".join(str(value) for value in invalid_rows[:6])
        more_suffix = "..." if len(invalid_rows) > 6 else ""
        problems.append(
            _problem(
                "warning",
                translate("validation.non_numeric_rows_title", column=config.stats_column),
                translate(
                    "validation.non_numeric_rows_detail",
                    count=str(len(invalid_rows)),
                    rows=f"{preview_rows}{more_suffix}",
                ),
                code="stats_non_numeric_rows",
                related_fields=("stats_column",),
            )
        )

    numeric_values = numeric.dropna().to_numpy(dtype=float)
    if numeric_values.size < 2:
        problems.append(
            _problem(
                "error",
                translate("validation.statistics_too_few_title"),
                translate("validation.statistics_too_few_detail", count=str(int(numeric_values.size))),
                code="stats_too_few",
                blocking=True,
                related_fields=("stats_column",),
            )
        )
        return tuple(problems)

    if np.allclose(numeric_values, numeric_values[0]):
        problems.append(
            _problem(
                "info",
                translate("validation.constant_data_title"),
                translate("validation.constant_data_detail"),
                code="constant_data",
                related_fields=("stats_column",),
            )
        )

    return tuple(problems)

