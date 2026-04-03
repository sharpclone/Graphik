"""Data loading, cleaning, and preparation utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from io import BytesIO, StringIO
import re
from typing import BinaryIO

import numpy as np
import pandas as pd



@dataclass(frozen=True)
class PreparedData:
    """Prepared numeric dataset for plotting/analysis."""

    dataframe: pd.DataFrame


def get_sample_dataframe() -> pd.DataFrame:
    """Return default sample dataset for the normal plotting mode."""
    return pd.DataFrame(
        {
            "m": [50, 100, 150, 200, 250],
            "y": [0.8464, 1.4641, 2.1609, 2.7556, 3.4969],
            "sigma_y": [0.01288, 0.0242, 0.01764, 0.0332, 0.0374],
        }
    )


def get_statistics_sample_dataframe() -> pd.DataFrame:
    """Return sample data for testing the statistics mode."""
    return pd.DataFrame(
        {
            "measurement": [
                9.81,
                10.02,
                9.74,
                10.15,
                9.95,
                10.08,
                9.88,
                9.67,
                10.21,
                9.92,
                10.05,
                9.99,
                9.83,
                10.11,
                9.77,
                9.90,
                10.18,
                9.86,
                10.07,
                9.94,
            ]
        }
    )


def _decode_text_content(content: bytes | str) -> str:
    """Decode uploaded text using common spreadsheet export encodings."""
    if isinstance(content, str):
        return content

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _detect_delimiter(text: str) -> str:
    """Detect common spreadsheet delimiters, falling back to comma."""
    sample_lines = [line for line in text.splitlines() if line.strip()]
    sample = "\n".join(sample_lines[:10])
    if not sample:
        return ","

    delimiters = ",;\t|"
    try:
        return str(csv.Sniffer().sniff(sample, delimiters=delimiters).delimiter)
    except csv.Error:
        counts = {delimiter: sample.count(delimiter) for delimiter in delimiters}
        detected = max(counts, key=counts.get)
        return detected if counts[detected] > 0 else ","


def _detect_decimal_separator(text: str, delimiter: str) -> str:
    """Infer decimal separator for delimited text."""
    if delimiter == ",":
        return "."

    sample_lines = [line for line in text.splitlines() if line.strip()]
    sample = "\n".join(sample_lines[:20])
    comma_decimals = len(re.findall(r"(?<!\d)\d+,\d+(?!\d)", sample))
    dot_decimals = len(re.findall(r"(?<!\d)\d+\.\d+(?!\d)", sample))
    return "," if comma_decimals > dot_decimals else "."


def _read_delimited_text(text: str) -> pd.DataFrame:
    """Read CSV-like text with delimiter and decimal auto-detection."""
    delimiter = _detect_delimiter(text)
    decimal = _detect_decimal_separator(text, delimiter)
    return pd.read_csv(StringIO(text), sep=delimiter, decimal=decimal)


def load_csv_file(file_obj: BinaryIO) -> pd.DataFrame:
    """Load CSV file-like object into dataframe."""
    content = file_obj.read()
    text = _decode_text_content(content)
    return _read_delimited_text(text)


def load_table_file(file_obj: BinaryIO, filename: str) -> pd.DataFrame:
    """Load CSV, Excel, or OpenDocument table file-like object into dataframe."""
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    content = file_obj.read()

    if suffix == "csv":
        text = _decode_text_content(content)
        return sanitize_dataframe(_read_delimited_text(text))

    if suffix in {"xlsx", "xls"}:
        if not isinstance(content, bytes):
            content = str(content).encode("utf-8")
        return sanitize_dataframe(pd.read_excel(BytesIO(content)))

    if suffix == "ods":
        if not isinstance(content, bytes):
            content = str(content).encode("utf-8")
        return sanitize_dataframe(pd.read_excel(BytesIO(content), engine="odf"))

    raise ValueError("Unsupported file format. Use .csv, .xlsx, .xls, or .ods.")


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dataframe columns and drop fully empty rows."""
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    cleaned = cleaned.dropna(axis=0, how="all")
    cleaned = _promote_first_row_as_header_if_likely(cleaned)
    cleaned = _rename_all_unnamed_columns(cleaned)
    return cleaned.reset_index(drop=True)


def _as_numeric_or_nan(value: object) -> float:
    """Convert scalar to float if possible, otherwise NaN."""
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else float("nan")


def _is_text_label(value: object) -> bool:
    """Return True if value looks like a non-empty textual label."""
    if value is None or pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    return np.isnan(_as_numeric_or_nan(text))


def _unique_headers(values: list[str], fallback: list[str]) -> list[str]:
    """Build unique header names, filling blanks with fallback names."""
    out: list[str] = []
    for idx, raw in enumerate(values):
        base = raw.strip() if raw.strip() else fallback[idx]
        candidate = base
        suffix = 2
        while candidate in out:
            candidate = f"{base}_{suffix}"
            suffix += 1
        out.append(candidate)
    return out


def _promote_first_row_as_header_if_likely(df: pd.DataFrame) -> pd.DataFrame:
    """
    Promote first row to headers when it appears to be a label row.

    Heuristic:
    - first row contains mostly textual labels,
    - second row contains mostly numeric data.
    """
    if df.empty or df.shape[0] < 2:
        return df

    ncols = int(df.shape[1])
    threshold = max(2, ncols // 2)
    max_scan = min(8, int(df.shape[0] - 1))

    header_row_index: int | None = None
    for idx in range(max_scan):
        first = df.iloc[idx]
        second = df.iloc[idx + 1]

        first_text_count = sum(_is_text_label(v) for v in first.tolist())
        second_numeric_count = sum(not np.isnan(_as_numeric_or_nan(v)) for v in second.tolist())

        if first_text_count >= threshold and second_numeric_count >= threshold:
            header_row_index = idx
            break

    if header_row_index is None:
        return df

    header_values = [
        str(v).strip() if v is not None and not pd.isna(v) else ""
        for v in df.iloc[header_row_index].tolist()
    ]
    fallback_headers = [str(c).strip() for c in df.columns]
    new_headers = _unique_headers(header_values, fallback_headers)

    promoted = df.iloc[header_row_index + 1 :].reset_index(drop=True).copy()
    promoted.columns = new_headers
    return promoted


def _rename_all_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename spreadsheet columns when all incoming names are 'Unnamed:*'.

    This commonly happens when OpenDocument header cells contain formulas
    without cached plain-text labels.
    """
    cols = [str(col).strip() for col in df.columns]
    if not cols:
        return df

    if all(col.lower().startswith("unnamed") for col in cols):
        defaults = ["m", "T_mean", "sigma_T", "y", "sigma_y"]
        renamed: list[str] = []
        for idx, _ in enumerate(cols):
            if idx < len(defaults):
                base = defaults[idx]
            else:
                base = f"col_{idx + 1}"

            name = base
            counter = 2
            while name in renamed:
                name = f"{base}_{counter}"
                counter += 1
            renamed.append(name)
        copy_df = df.copy()
        copy_df.columns = renamed
        return copy_df

    return df


def _trim_leading_non_data_rows(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    """
    Drop leading rows until the first row where all required columns are numeric.

    This allows spreadsheet title/header rows above the numeric table while
    preserving strict validation once numeric data begins.
    """
    if df.empty or not required_columns:
        return df

    numeric_df = pd.DataFrame(
        {col: pd.to_numeric(df[col], errors="coerce") for col in required_columns}
    )
    complete_numeric = numeric_df.notna().all(axis=1).to_numpy()
    valid_indices = np.flatnonzero(complete_numeric)
    if valid_indices.size == 0:
        return df

    first_valid = int(valid_indices[0])
    if first_valid == 0:
        return df

    return df.iloc[first_valid:].reset_index(drop=True)


def _to_numeric(series: pd.Series, column_name: str) -> pd.Series:
    """Convert series to numeric and raise on invalid entries."""
    numeric = pd.to_numeric(series, errors="coerce")
    invalid_mask = numeric.isna() & series.notna()
    if invalid_mask.any():
        bad_index = int(invalid_mask[invalid_mask].index[0])
        bad_value = series.loc[bad_index]
        raise ValueError(
            f"Column '{column_name}' contains non-numeric value '{bad_value}' in row {bad_index + 1}."
        )
    return numeric


def prepare_measurement_data(
    source_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    sigma_y_col: str,
) -> PreparedData:
    """Build the validated analysis dataframe with explicit x, y, and sigma_y columns."""
    df = sanitize_dataframe(source_df)

    if x_col not in df.columns:
        raise ValueError(f"Selected x column '{x_col}' does not exist.")
    if y_col not in df.columns:
        raise ValueError("Choose a valid y column.")
    if sigma_y_col not in df.columns:
        raise ValueError("Choose a valid sigma_y column.")

    required_columns = [x_col, y_col, sigma_y_col]
    df = _trim_leading_non_data_rows(df, required_columns)

    out = pd.DataFrame()
    out["x"] = _to_numeric(df[x_col], x_col)
    out["y"] = _to_numeric(df[y_col], y_col)
    out["sigma_y"] = _to_numeric(df[sigma_y_col], sigma_y_col)

    out = out.dropna(subset=["x", "y", "sigma_y"]).reset_index(drop=True)

    if out.shape[0] < 2:
        raise ValueError("At least two complete numeric rows are required.")

    if (out["sigma_y"] < 0).any():
        raise ValueError("sigma_y values must be non-negative.")

    return PreparedData(dataframe=out)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Export dataframe as UTF-8 CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")
