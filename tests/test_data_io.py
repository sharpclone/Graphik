"""Unit tests for file loading utilities."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

from src.data_io import get_statistics_sample_dataframe, load_table_file, prepare_measurement_data, sanitize_dataframe


def test_load_table_file_csv() -> None:
    content = b"m,T_mean,sigma_T\n50,0.92,0.007\n100,1.21,0.01\n"
    df = load_table_file(BytesIO(content), "test.csv")

    assert list(df.columns) == ["m", "T_mean", "sigma_T"]
    assert df.shape == (2, 3)


def test_load_table_file_csv_semicolon_decimal_comma() -> None:
    content = b"m;T_mean;sigma_T\n50;0,92;0,007\n100;1,21;0,01\n"
    df = load_table_file(BytesIO(content), "test.csv")

    assert list(df.columns) == ["m", "T_mean", "sigma_T"]
    assert df["T_mean"].tolist() == [0.92, 1.21]
    assert df["sigma_T"].tolist() == [0.007, 0.01]


def test_load_table_file_csv_cp1252_headers() -> None:
    content = "Dicke;Intensität\n0;5253\n20;4460\n".encode("cp1252")
    df = load_table_file(BytesIO(content), "test.csv")

    assert list(df.columns) == ["Dicke", "Intensität"]
    assert df["Intensität"].tolist() == [5253, 4460]


def test_load_table_file_xlsx() -> None:
    source = pd.DataFrame({"m": [50, 100], "T_mean": [0.92, 1.21], "sigma_T": [0.007, 0.01]})
    buf = BytesIO()
    source.to_excel(buf, index=False)
    buf.seek(0)

    df = load_table_file(buf, "test.xlsx")

    assert list(df.columns) == ["m", "T_mean", "sigma_T"]
    assert df.shape == (2, 3)


def test_load_table_file_ods() -> None:
    source = pd.DataFrame({"m": [50, 100], "T_mean": [0.92, 1.21], "sigma_T": [0.007, 0.01]})
    buf = BytesIO()
    source.to_excel(buf, index=False, engine="odf")
    buf.seek(0)

    df = load_table_file(buf, "test.ods")

    assert list(df.columns) == ["m", "T_mean", "sigma_T"]
    assert df.shape == (2, 3)


def test_load_table_file_unsupported_extension() -> None:
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_table_file(BytesIO(b"dummy"), "test.txt")


def test_sanitize_dataframe_renames_fully_unnamed_columns() -> None:
    source = pd.DataFrame(
        [[50, 0.92, 0.007, 0.8464, 0.0129]],
        columns=["Unnamed: 0", "Unnamed: 1", "Unnamed: 2", "Unnamed: 3", "Unnamed: 4"],
    )
    out = sanitize_dataframe(source)
    assert list(out.columns) == ["x", "y", "sigma_y", "col_4", "col_5"]


def test_sanitize_dataframe_promotes_first_text_row_to_headers() -> None:
    source = pd.DataFrame(
        [
            ["Masse, [g]", "Auslenkung x, [mm]", "err"],
            [50, 160, 1],
            [100, 335, 1],
        ],
        columns=["Unnamed: 0", "Unnamed: 1", "Unnamed: 2"],
    )
    out = sanitize_dataframe(source)
    assert list(out.columns) == ["Masse, [g]", "Auslenkung x, [mm]", "err"]
    assert out.iloc[0].tolist() == [50, 160, 1]


def test_sanitize_dataframe_promotes_header_after_leading_empty_rows() -> None:
    source = pd.DataFrame(
        [
            [None, None, None],
            [None, None, None],
            ["Masse, [g]", "Auslenkung x, [mm]", "err"],
            [50, 160, 1],
            [100, 335, 1],
        ],
        columns=["Unnamed: 0", "Unnamed: 1", "Unnamed: 2"],
    )
    out = sanitize_dataframe(source)

    assert list(out.columns) == ["Masse, [g]", "Auslenkung x, [mm]", "err"]
    assert out.iloc[0].tolist() == [50, 160, 1]


def test_prepare_measurement_data_trims_leading_non_data_rows() -> None:
    source = pd.DataFrame(
        {
            "m": [None, None, "Masse, [g]", 50, 100],
            "T_mean": [None, None, None, 0.92, 1.21],
            "sigma_T": [None, None, None, 0.007, 0.01],
            "y": [None, None, None, 0.8464, 1.4641],
            "sigma_y": [None, None, None, 0.0129, 0.0242],
        }
    )
    prepared = prepare_measurement_data(
        source_df=source,
        x_col="m",
        y_col="y",
        sigma_y_col="sigma_y",
    )

    assert prepared.dataframe.shape == (2, 3)
    assert prepared.dataframe["x"].tolist() == [50.0, 100.0]


def test_get_statistics_sample_dataframe_has_numeric_measurements() -> None:
    df = get_statistics_sample_dataframe()

    assert list(df.columns) == ["measurement"]
    assert df.shape[0] >= 10
    assert pd.api.types.is_numeric_dtype(df["measurement"])
