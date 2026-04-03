"""Optional data preprocessing helpers for derived laboratory quantities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .calculations import compute_sigma_y_from_t, compute_y_from_t


@dataclass(frozen=True)
class DerivedColumnsResult:
    """Result of adding derived y/sigma_y columns to a measurement table."""

    dataframe: pd.DataFrame
    y_column: str = "y"
    sigma_y_column: str = "sigma_y"


def add_period_squared_columns(
    source_df: pd.DataFrame,
    *,
    t_mean_col: str,
    sigma_t_col: str,
    y_column: str = "y",
    sigma_y_column: str = "sigma_y",
) -> DerivedColumnsResult:
    """Add y=T^2 and sigma_y=2*T*sigma_T as explicit preprocessing output."""
    if t_mean_col not in source_df.columns:
        raise ValueError(f"Selected T_mean column '{t_mean_col}' does not exist.")
    if sigma_t_col not in source_df.columns:
        raise ValueError(f"Selected sigma_T column '{sigma_t_col}' does not exist.")

    t_numeric = pd.to_numeric(source_df[t_mean_col], errors="coerce")
    sigma_t_numeric = pd.to_numeric(source_df[sigma_t_col], errors="coerce")
    invalid_t = t_numeric.isna() & source_df[t_mean_col].notna()
    invalid_sigma = sigma_t_numeric.isna() & source_df[sigma_t_col].notna()
    if invalid_t.any():
        row_index = int(invalid_t[invalid_t].index[0])
        raise ValueError(f"Column '{t_mean_col}' contains a non-numeric value in row {row_index + 1}.")
    if invalid_sigma.any():
        row_index = int(invalid_sigma[invalid_sigma].index[0])
        raise ValueError(f"Column '{sigma_t_col}' contains a non-numeric value in row {row_index + 1}.")

    out = source_df.copy()
    out[y_column] = compute_y_from_t(t_numeric.to_numpy())
    out[sigma_y_column] = compute_sigma_y_from_t(t_numeric.to_numpy(), sigma_t_numeric.to_numpy())
    return DerivedColumnsResult(dataframe=out, y_column=y_column, sigma_y_column=sigma_y_column)
