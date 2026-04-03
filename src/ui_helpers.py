"""Small UI-facing helpers for labels and displayed equations."""

from __future__ import annotations

import re

import numpy as np

from .config import DEFAULT_X_LABEL, DEFAULT_Y_LABEL
from .i18n import translate


_TEX_UNICODE_MAP = {
    r"\Delta": "Δ",
    r"\delta": "δ",
    r"\sigma": "σ",
    r"\Sigma": "Σ",
    r"\mu": "μ",
    r"\lambda": "λ",
    r"\Lambda": "Λ",
    r"\pi": "π",
    r"\Pi": "Π",
    r"\phi": "φ",
    r"\Phi": "Φ",
    r"\omega": "ω",
    r"\Omega": "Ω",
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\tau": "τ",
    r"\theta": "θ",
    r"\rho": "ρ",
    r"\pm": "±",
    r"\cdot": "·",
    r"\times": "×",
}

_SUPERSCRIPT_CHARS = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
    "⁼": "=",
    "⁽": "(",
    "⁾": ")",
    "ⁱ": "i",
    "ⁿ": "n",
}

_SUBSCRIPT_CHARS = {
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
    "₊": "+",
    "₋": "-",
    "₌": "=",
    "₍": "(",
    "₎": ")",
    "ₐ": "a",
    "ₑ": "e",
    "ₕ": "h",
    "ᵢ": "i",
    "ⱼ": "j",
    "ₖ": "k",
    "ₗ": "l",
    "ₘ": "m",
    "ₙ": "n",
    "ₒ": "o",
    "ₚ": "p",
    "ᵣ": "r",
    "ₛ": "s",
    "ₜ": "t",
    "ᵤ": "u",
    "ᵥ": "v",
    "ₓ": "x",
}

_SUPERSCRIPT_PATTERN = re.compile(
    "(" + "|".join(re.escape(char) for char in _SUPERSCRIPT_CHARS) + ")+"
)
_SUBSCRIPT_PATTERN = re.compile(
    "(" + "|".join(re.escape(char) for char in _SUBSCRIPT_CHARS) + ")+"
)
_MATH_TRIGGER_PATTERN = re.compile(r"(\\[A-Za-z]+|[_^]|[ΔδσΣμλΛπΠφΦωΩαβγτθρ±·×]|[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ])")


def safe_default_index(options: list[str], preferred: str, fallback: int = 0) -> int:
    """Return preferred option index if present, otherwise fallback."""
    if preferred in options:
        return options.index(preferred)
    return fallback


def _replace_tex_macros(text: str) -> str:
    """Replace common TeX macros with readable Unicode symbols."""
    out = str(text)
    for macro, unicode_char in _TEX_UNICODE_MAP.items():
        out = out.replace(macro, unicode_char)
    return out


def _replace_unicode_scripts(text: str) -> str:
    """Normalize Unicode superscripts/subscripts into ^{...}/_{...} form."""
    out = str(text)
    out = _SUPERSCRIPT_PATTERN.sub(
        lambda match: "^{" + "".join(_SUPERSCRIPT_CHARS[ch] for ch in match.group(0)) + "}",
        out,
    )
    out = _SUBSCRIPT_PATTERN.sub(
        lambda match: "_{" + "".join(_SUBSCRIPT_CHARS[ch] for ch in match.group(0)) + "}",
        out,
    )
    return out


def _convert_scripts_to_html(text: str) -> str:
    """Convert ^ / _ notation into HTML sup/sub tags for Plotly text rendering."""
    out = str(text)
    previous = None
    while out != previous:
        previous = out
        out = re.sub(r"\^\{([^{}]+)\}", r"<sup>\1</sup>", out)
        out = re.sub(r"_\{([^{}]+)\}", r"<sub>\1</sub>", out)
        out = re.sub(r"\^([A-Za-z0-9+\-().]+)", r"<sup>\1</sup>", out)
        out = re.sub(r"_([A-Za-z0-9+\-.]+)", r"<sub>\1</sub>", out)
    return out


def prettify_plot_text(text: str) -> str:
    """Convert TeX-like math notation to web-friendly Unicode/HTML."""
    out = _replace_tex_macros(_replace_unicode_scripts(str(text).strip()))
    out = _convert_scripts_to_html(out)
    out = out.replace("{", "").replace("}", "")
    return out


def to_plot_math_text(text: str, use_latex: bool) -> str:
    """Convert math-style label input into Plotly-friendly rich text when enabled."""
    stripped = str(text).strip()
    if not stripped:
        return str(text)
    if not use_latex:
        return stripped
    if not _MATH_TRIGGER_PATTERN.search(stripped):
        return stripped
    return prettify_plot_text(stripped)


def squared_label_from_column(column_name: str) -> str:
    """Build readable squared label from a source column header."""
    label = str(column_name).strip()
    if not label:
        return "T²"
    match = re.match(r"^(.*?)\s*\[(.*?)\]\s*$", label)
    if match:
        quantity = match.group(1).strip()
        unit = match.group(2).strip()
        quantity = quantity if quantity else "T"
        return f"({quantity})² [{unit}²]"
    return f"({label})²"


def auto_axis_labels(
    x_column: str,
    y_column: str | None,
) -> tuple[str, str]:
    """Suggest axis labels based on selected explicit x/y columns."""
    x_label = str(x_column).strip() if str(x_column).strip() else DEFAULT_X_LABEL
    y_col = str(y_column).strip() if y_column else ""
    y_label = y_col if y_col else DEFAULT_Y_LABEL
    return x_label, y_label


def auto_line_labels(error_method: str, lang: str = "de") -> tuple[str, str, str]:
    """Suggest line labels based on selected error-line method."""
    fit_label = translate(lang, "fit.default_label")
    if error_method == "protocol":
        return (
            fit_label,
            translate(lang, "error.default_label.max.standard"),
            translate(lang, "error.default_label.min.standard"),
        )
    return (
        fit_label,
        translate(lang, "error.default_label.max.centroid"),
        translate(lang, "error.default_label.min.centroid"),
    )


def format_linear_equation(label: str, slope: float, intercept: float, decimals: int = 5) -> str:
    """Format y = a*x + b equation text."""
    sign = "+" if intercept >= 0 else "-"
    return (
        f"{label}: y = {slope:.{int(decimals)}g}·x {sign} {abs(intercept):.{int(decimals)}g}"
    )


def format_exponential_equation(
    label: str,
    slope_log: float,
    intercept_log: float,
    decimals: int = 5,
) -> str:
    """Format y = k*exp(a*x) equation text from log-space line params."""
    prefactor = float(np.exp(intercept_log))
    return (
        f"{label}: y = {prefactor:.{int(decimals)}g}·e^({slope_log:.{int(decimals)}g}·x)"
    )


def fit_line_help_text(fit_model: str, lang: str = "de") -> str:
    """Return tooltip text for the Ausgleichsgerade toggle."""
    if fit_model == "exp":
        return translate(lang, "tooltip.fit_exp")
    return translate(lang, "tooltip.fit_linear")


def error_line_help_text(fit_model: str, error_method: str, lang: str = "de") -> str:
    """Return tooltip text for the Fehlergeraden toggle."""
    model_note = translate(
        lang,
        "tooltip.error_model_note.exp" if fit_model == "exp" else "tooltip.error_model_note.linear",
    )

    if error_method == "protocol":
        return translate(lang, "tooltip.error_protocol", model_note=model_note)

    return translate(lang, "tooltip.error_centroid", model_note=model_note)


def _clean_label_for_symbol(label: str) -> str:
    """Extract the variable-like part from an axis label."""
    text = str(label).strip()
    if not text:
        return ""
    if text.startswith("$") and text.endswith("$") and len(text) >= 2:
        text = text[1:-1].strip()
    text = re.sub(r"\[.*?\]", "", text).strip()
    text = text.rstrip(",:;")
    if text.startswith("(") and text.endswith(")") and len(text) > 2:
        text = text[1:-1].strip()
    return text


def auto_delta_symbol_from_label(label: str, fallback_axis: str) -> str:
    """Return auto-detected delta symbol from label, with axis fallback."""
    cleaned = _clean_label_for_symbol(label)
    if cleaned:
        return f"Δ{cleaned}"
    return f"Δ{fallback_axis.upper()}"


def auto_triangle_delta_symbols(x_label: str, y_label: str) -> tuple[str, str]:
    """Return auto-detected horizontal and vertical delta symbols."""
    return (
        auto_delta_symbol_from_label(x_label, "X"),
        auto_delta_symbol_from_label(y_label, "Y"),
    )
