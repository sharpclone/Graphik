# Graphik: Technical Description for Expert Review

## 1. Purpose and Scope

Graphik is a local-first plotting application built with Streamlit. It is designed to let non-programmer users import tabular data, configure a scientific plot visually, compute selected analytical overlays, and export publication-ready figures and summaries.

The current application has two runtime modes:

1. `Normal`
   Used for generic x-y plotting with optional vertical uncertainties, linear/exponential fitting, subjective error lines, slope triangles, configurable grids, and export.
2. `Statistics`
   Used for one-dimensional descriptive statistics, histogram rendering, normal-distribution overlays, mean/standard-deviation markers, and export.

The system is intentionally local:

- during development it runs via `streamlit run app.py`
- in the packaged Windows build it runs as a local web app behind a desktop launcher and opens `http://127.0.0.1:8501`

The application is no longer limited to a single physics experiment. It still contains several features that are especially useful for German lab-report style plots, but the implementation is now general enough to support a broader set of x-y workflows.

---

## 2. Runtime Architecture

The application is organized into a thin Streamlit front-end and a set of focused support modules.

### 2.1 Entry Point

Main entry point:
- [app.py](D:\grafiken\app.py)

This file is responsible for:

- starting the Streamlit page
- restoring persisted UI state
- rendering the sidebar and the active mode UI
- orchestrating data preparation
- invoking analysis functions
- building the Plotly preview figure
- preparing export figures
- triggering on-demand export generation

### 2.2 Core Modules

- [src/data_io.py](D:\grafiken\src\data_io.py)
  File import, CSV/Excel/ODS loading, dataframe sanitization, numeric validation, and sample datasets.

- [src/calculations.py](D:\grafiken\src\calculations.py)
  Linear regression, exponential regression via log transform, error-line methods, compatibility checks, and slope-formatting helpers.

- [src/plotting.py](D:\grafiken\src\plotting.py)
  Plotly preview-figure construction, axis/grid configuration, line rendering, triangles, and export entry point.

- [src/export_utils.py](D:\grafiken\src\export_utils.py)
  Export-oriented figure scaling, auto/manual plot info box logic, autoscaling for export, summary generation, and related helpers.

- [src/mpl_export.py](D:\grafiken\src\mpl_export.py)
  Matplotlib export backend. It converts the Plotly preview figure into a Matplotlib-rendered image/PDF/SVG.

- [src/statistics.py](D:\grafiken\src\statistics.py)
  Descriptive statistics and normal-distribution helper functions for the Statistics mode.

- [src/geometry.py](D:\grafiken\src\geometry.py)
  Deterministic triangle-point selection and small geometry helpers.

- [src/ui_helpers.py](D:\grafiken\src\ui_helpers.py)
  Label suggestions, plot math-text conversion, equation formatting, and tooltip text.

- [src/ui_state.py](D:\grafiken\src\ui_state.py)
  Persistent session-state snapshotting across app restarts.

### 2.3 Packaged Desktop Runtime

- [launcher.py](D:\grafiken\launcher.py)
  Desktop launcher for the packaged Windows build.

The launcher:

- chooses a localhost port, default `8501`
- checks whether Graphik is already running on that port
- opens the browser immediately if an instance is already healthy
- otherwise starts Streamlit programmatically
- waits until the local server responds before opening the browser
- forces a light theme in the packaged build

The Windows bundle is produced by:

- [Graphik.spec](D:\grafiken\Graphik.spec)
- [build_windows.bat](D:\grafiken\build_windows.bat)

---

## 3. Startup Sequence

The startup logic in [app.py](D:\grafiken\app.py) is:

1. `st.set_page_config(...)`
   Configures title, favicon, and wide layout.

2. `_restore_session_snapshot(...)`
   Loads the last saved session snapshot from `.streamlit/last_session_state.pkl` if persistence is enabled.

3. `_init_session_state(...)`
   Ensures the app has required defaults such as:
   - language
   - app mode
   - remember settings flag
   - initial table dataframe
   - x/y range toggles
   - error triangle visibility

4. The UI is then rendered from restored state.

This means the app is not stateless across restarts. Most user configuration survives app reloads unless explicitly cleared.

---

## 4. Data Ingestion Pipeline

## 4.1 Input Sources

In both modes, the user can work from:

- built-in sample data
- manual table editing in `st.data_editor`
- uploaded files: `.csv`, `.xlsx`, `.xls`, `.ods`

The relevant loader entry point is:
- `load_table_file(...)` in [src/data_io.py](D:\grafiken\src\data_io.py)

## 4.2 CSV Parsing Strategy

CSV import is intentionally robust for spreadsheet exports.

The loader automatically handles:

- delimiters: `,`, `;`, tab, `|`
- decimal separators: `.` and `,`
- common encodings: `utf-8-sig`, `utf-8`, `cp1252`, `latin-1`

This is implemented by:

- `_decode_text_content(...)`
- `_detect_delimiter(...)`
- `_detect_decimal_separator(...)`
- `_read_delimited_text(...)`

The purpose is to support CSV files produced by Excel or LibreOffice in both English and European locales.

## 4.3 Excel and ODS Parsing

For `.xlsx` / `.xls`:
- `pandas.read_excel(BytesIO(content))`

For `.ods`:
- `pandas.read_excel(BytesIO(content), engine="odf")`

ODS support is important because many users prepare lab tables in LibreOffice.

## 4.4 Dataframe Sanitization

After file loading, `sanitize_dataframe(...)` is applied.

This performs:

1. column-name normalization
   - converts all column names to stripped strings

2. drop fully empty rows

3. heuristic header promotion
   - `_promote_first_row_as_header_if_likely(...)`
   - scans the first rows and detects the common case:
     - one row contains mostly textual labels
     - the following row contains mostly numeric data
   - if detected, the text row becomes the header row

4. all-unnamed fallback renaming
   - `_rename_all_unnamed_columns(...)`
   - if all columns are `Unnamed:*`, Graphik assigns deterministic fallback names such as:
     - `m`
     - `T_mean`
     - `sigma_T`
     - `y`
     - `sigma_y`

This is specifically useful for `.ods` files where header cells may contain formulas rather than cached plain-text labels.

## 4.5 Numeric Validation

Once the user maps columns, `prepare_measurement_data(...)` validates and produces the normalized analysis dataframe:

- required output columns are always:
  - `x`
  - `y`
  - `sigma_y`

- selected columns are converted via `_to_numeric(...)`
- any non-numeric cell triggers a precise `ValueError`
  including column name and row index

This prevents silent coercion of malformed spreadsheet inputs.

### Important current product decision

Although [src/data_io.py](D:\grafiken\src\data_io.py) still contains support for deriving:

- `y = T^2`
- `sigma_y = 2 T sigma_T`

that path is currently disabled in the active UI.

In [app.py](D:\grafiken\app.py), the app always calls:

- `derive_y=False`
- `derive_sigma_y=False`

So the current user-facing product expects the user to provide already-prepared `y` and `sigma_y` columns.

This is an intentional simplification: the app is currently positioned as a plotting and analysis tool, not an uncertainty-propagation calculator.

---

## 5. Session State and Persistence

State persistence is implemented in [src/ui_state.py](D:\grafiken\src\ui_state.py).

### 5.1 What is persisted

The app stores picklable values from `st.session_state` in:
- `.streamlit/last_session_state.pkl`

Persisted examples include:

- mode selection
- language
- column mappings
- label strings
- grid settings
- font settings
- export settings
- plot info box settings

### 5.2 What is deliberately not persisted

The persistence layer excludes:

- non-assignable Streamlit widget keys such as `table_editor`
- transient export caches whose keys start with `_export_cache_`
- non-picklable data

This prevents two common Streamlit failure modes:

- writing to a widget key after instantiation
- persisting large export byte arrays into long-lived session snapshots

---

## 6. Mode Structure

The application branches into two major paths.

## 6.1 Normal Mode

Normal mode is the main x-y plotting workflow.

The user maps:

- x column
- y column
- vertical uncertainty column, or zero error if explicitly enabled

The prepared dataframe then becomes:

- `analysis_df`
- columns normalized to `x`, `y`, `sigma_y`

All subsequent plotting and analysis operates on that normalized structure.

## 6.2 Statistics Mode

Statistics mode uses one numeric column only.

The user selects:

- the numeric measurement column
- histogram bin count
- density normalization toggle
- whether to overlay a normal fit
- whether to show mean / sigma markers / formula box

The core numeric sample is converted to a clean numeric vector and passed to:

- `describe_distribution(...)`
- `normal_curve_points(...)`

This mode shares the same export framework as Normal mode, but the chart-building logic differs.

---

## 7. Normal Mode Analysis Pipeline

## 7.1 Fit Model Selection

Graphik supports two fit models in the current app logic:

1. linear model
   - `y = a x + b`
   - computed by `linear_regression(...)`

2. exponential model
   - `y = k e^(a x)`
   - implemented as linear regression in log space:
     - `ln(y) = a x + ln(k)`
   - computed by `exponential_regression(...)`

For exponential fits:

- all `y` values must be strictly positive
- uncertainty handling for error-line analysis uses:
  - `z = ln(y)`
  - `sigma_z = sigma_y / y`
  from `logarithmic_transform_with_uncertainty(...)`

## 7.2 Linear Regression Details

`linear_regression(...)` in [src/calculations.py](D:\grafiken\src\calculations.py) wraps `scipy.stats.linregress`.

Returned values:

- slope `k_fit` (displayed in the UI as `a_fit`)
- intercept `l_fit` (displayed as `b_fit`)
- correlation coefficient `r_value`
- p-value `p_value`
- standard error of slope `std_err`

The app later displays `R^2 = r_value^2` when enabled.

## 7.3 Error-Line Methods

The app currently exposes two subjective error-line strategies.

### 7.3.1 Standard endpoint method

Internal selector value: `protocol`
Displayed label: `Standardmethode (Endpunktgeraden)`

This path calls:
- `protocol_endpoint_error_lines(...)`

The standard method is intended to match common lab-report practice more closely than the strict centroid method.

The key idea is:

- construct extreme plausible lines using endpoint logic and error intervals
- use those lines as the subjective slope envelope

This method is more permissive and more likely to produce visible error lines for real student data.

### 7.3.2 Strict centroid method

Internal selector value: `centroid`
Displayed label: `Schwerpunktmethode (streng)`

This path calls:
- `centroid_error_lines(...)`

Algorithm:

1. compute centroid
   - `x_bar = mean(x_i)`
   - `y_bar = mean(y_i)`

2. force both extreme lines to pass through the centroid

3. for each point `(x_i, y_i ± sigma_i)` derive an admissible slope interval

4. intersect all admissible intervals

5. if the global interval is non-empty:
   - `a_min = lower bound`
   - `a_max = upper bound`
   - corresponding intercepts are computed from the centroid

6. if the intersection is empty:
   - the app reports that no centroid-compatible error lines exist

This method is mathematically stricter but often infeasible when error bars are small.

### 7.3.3 Additional implemented but currently non-primary helpers

The calculations module also contains:

- `free_intercept_error_lines(...)`
- `endpoint_extreme_error_lines(...)`

These are useful alternative formulations, but the current UI path centers on the standard endpoint method and the strict centroid method.

## 7.4 Slope Triangles

Slope triangles are deterministic, not random.

Geometry lives in [src/geometry.py](D:\grafiken\src\geometry.py).

Relevant functions:

- `auto_triangle_points(...)`
- `custom_points_from_x(...)`
- `right_triangle_corner(...)`
- `triangle_deltas(...)`

For the fit line:

- if automatic mode is enabled, the app selects points well inside the visible x interval using a margin fraction
- if manual mode is enabled, the user supplies x-values for A and B and the app computes the corresponding y-values on the line

For error lines in the linear case:

- the app chooses deterministic x positions near the left and right ends of the data interval
- points are reordered when necessary so that triangle annotations remain readable and deltas remain conceptually positive

Displayed labels for the triangle are not hard-coded. They are derived from the current axis labels via:

- `auto_triangle_delta_symbols(...)`

Examples:

- `m [g]` -> `Δm`
- `T^2 [s^2]` -> `ΔT^2`
- fallback -> `ΔX`, `ΔY`

## 7.5 On-plot Equation / Information Box

The information box is built from lines collected during analysis.

Possible contents include:

- fit equation
- error-line equations
- `R^2`
- statistics formula in Statistics mode

Placement is handled by `add_plot_text_block(...)` in [src/export_utils.py](D:\grafiken\src\export_utils.py).

This is one of the more complex UI algorithms in the codebase.

### Auto-placement algorithm

1. estimate the visible data bounds from traces and axis ranges
2. collect normalized point samples from:
   - markers
   - line segments
   - bar bodies
   - error bar extents
3. estimate text-box width and height from:
   - number of lines
   - font size
   - approximate character widths
4. wrap lines if necessary
5. downscale font only if needed
6. generate candidate positions in normalized paper coordinates
7. score candidates by:
   - overlap count with existing plot content
   - minimum distance to data
   - penalty for being too close to the bottom axis
   - preference for higher placement
8. if overlap remains high, the font is reduced further and candidate search is repeated

### Manual mode

If manual mode is enabled, the user can set:

- x
- y
- max width
- max height
- text size

The same layout is reused for export.

---

## 8. Plot Preview Construction

Preview rendering is built with Plotly in [src/plotting.py](D:\grafiken\src\plotting.py).

## 8.1 PlotStyle Abstraction

The `PlotStyle` dataclass centralizes visual parameters such as:

- axis labels
- whether the grid is enabled
- y-axis type (`linear` or `log`)
- number of visible decades for log-y
- whether points are connected
- marker/error-bar styling
- font sizes
- major/minor grid spacing
- custom axis ranges

This makes preview and export easier to keep in sync because the preview figure is built from a compact structured style object rather than a large number of unrelated parameters.

## 8.2 Base Figure Creation

`create_base_figure(...)` builds the initial Plotly figure.

It always starts with the measured dataset trace:

- `go.Scatter`
- black markers
- optional connecting line
- vertical error bars

For log-y plots:

- lower error bars are clipped to remain strictly positive
- this avoids rendering artifacts or illegal values on a log axis

## 8.3 Grid and Axis Logic

Graphik supports both coarse and fine grid modes.

For linear axes:

- major tick spacing is set via `tick0` and `dtick`
- optional minor grid uses Plotly `minor` axis configuration

For log-y:

- y-axis type is `log`
- if no explicit major `dtick` is given, Plotly uses `D2` by default
- the app can also generate explicit semilog-paper style horizontal line shapes via `_add_semilog_paper_gridlines(...)`
  when minor log grid readability is otherwise insufficient

This is important because Plotly’s native log-minor grid can be visually sparse or inconsistent, especially in semilog lab-style plots.

## 8.4 Math-style Labels

User-entered labels are converted by [src/ui_helpers.py](D:\grafiken\src\ui_helpers.py).

The app does not rely on full LaTeX rendering in the Plotly preview.
Instead it converts a TeX-like input syntax into a Plotly-friendly mix of:

- Unicode symbols
- HTML `<sup>` / `<sub>` tags

Examples supported:

- `\Delta m`
- `T_i`
- `T^2`
- `\sigma`
- `e^{ax}`

The main entry point is:
- `to_plot_math_text(...)`

This design was chosen because it is more robust inside the Streamlit + Plotly preview than full MathJax-based rendering.

---

## 9. Statistics Mode Pipeline

Statistics mode is separate from the x-y scientific fit pipeline.

## 9.1 Numeric Extraction

The user selects one column.
That column is coerced to numeric values.
Non-numeric rows are discarded at the statistics computation stage.

## 9.2 Summary Statistics

`describe_distribution(...)` computes:

- sample size
- mean
- sample variance (`ddof=1`)
- sample standard deviation
- median
- min / max
- Q1 / Q3

## 9.3 Normal Fit Overlay

If enabled and if the data are not constant:

- `normal_curve_points(...)` constructs a smooth x-grid
- `normal_pdf(...)` evaluates the fitted Gaussian

The plot can then display:

- histogram
- fitted normal curve
- mean marker
- ±1σ, ±2σ, ±3σ markers
- formula box

This mode reuses the same export system as Normal mode, which is why export consistency matters across both modes.

---

## 10. Export Pipeline

The export system is one of the most important parts of the application.

The application intentionally separates:

- Plotly preview rendering
- Matplotlib export rendering

This was done because Plotly/Kaleido-based export proved less stable for the packaged Windows build.

## 10.1 Why the app uses two renderers

Preview uses Plotly because it provides:

- interactivity
- browser-native zooming
- responsive layout inside Streamlit
- easier user feedback while adjusting settings

Export uses Matplotlib because it provides:

- stable local PDF/SVG/PNG generation
- no browser dependency during export
- predictable packaging inside PyInstaller

## 10.2 Export Triggering

Export is on-demand.

The user does not regenerate PNG/SVG/PDF every time the plot changes.
Instead:

1. a `Prepare ...` button is clicked
2. Graphik computes the export bytes only then
3. the resulting artifact is cached in session state
4. the corresponding `Download ...` button is shown

This behavior is implemented by `_render_on_demand_image_export(...)` in [app.py](D:\grafiken\app.py).

The cache key includes a signature generated by `_export_signature(...)`.
The signature hashes:

- `fig.to_json()`
- export-specific settings such as:
  - format
  - width / height
  - DPI
  - paper size / orientation
  - target text size
  - visual scale
  - autoscale toggle
  - plot info box settings

Whenever any relevant input changes, the export cache is invalidated automatically.

## 10.3 Export Figure Preparation

Before export, the preview figure is copied and transformed.

### Normal mode export builder

In [app.py](D:\grafiken\app.py), `_build_normal_export_figure()` performs:

1. choose figure source:
   - either the current preview figure
   - or an autoscaled copy using `_autoscale_figure_to_data(...)`

2. set explicit export width and height in pixels based on paper size and DPI

3. scale the figure for export using `_scale_figure_for_export(...)`

4. compute the export font size for the plot info box via `_export_plot_text_font_size(...)`

5. reinsert the info box into the export figure with `_place_plot_text_block(...)`

### Statistics mode export builder

`_build_statistics_export_figure()` does the analogous work for histogram-based figures.

## 10.4 Export Scaling Strategy

Scaling is handled by `scale_figure_for_export(...)` in [src/export_utils.py](D:\grafiken\src\export_utils.py).

This function deliberately distinguishes:

- text scaling
- non-text visual scaling

### Text scaling

Controlled by:
- `target_text_pt`
- `text_unit="pt"`

This means export text size is interpreted as real typographic size in points, not browser pixels.

### Non-text scaling

Controlled by:
- `visual_scale`

This affects:

- line widths
- marker sizes
- error bar thickness and cap width

The crucial design rule is:
- `visual_scale` must not inflate text again

That separation was added specifically to avoid oversized export text while still allowing thicker lines/markers on large paper sizes.

## 10.5 Matplotlib Translation Layer

The actual conversion from Plotly figure to image bytes is done by:

- `figure_to_image_bytes(...)` in [src/plotting.py](D:\grafiken\src\plotting.py)
- `plotly_figure_to_image_bytes(...)` in [src/mpl_export.py](D:\grafiken\src\mpl_export.py)

The export backend performs the following stages.

### Stage 1: create Matplotlib figure

- create `Figure(figsize=(width_px/dpi, height_px/dpi), dpi=dpi)`
- create one axes object

### Stage 2: apply layout

`_apply_layout(...)` transfers:

- paper background
- plot background
- margins
- global font
- x and y axis formatting
- figure title

### Stage 3: axis formatting

`_apply_axis_format(...)` handles:

- x/y titles
- tick font sizes
- linear and log scales
- explicit axis ranges
- major grid
- fixed tick values / tick labels when provided
- decimal formatting when `tickformat` is simple fixed-point
- minor locators and minor gridlines

### Stage 4: draw traces

`_draw_traces(...)` dispatches by trace type.

Currently implemented:

- `scatter`
- `bar`

For scatter traces:
- lines
- markers
- `error_y`
- `error_x`

For bar traces:
- bar width
- fill color
- edge color
- edge width
- opacity

### Stage 5: draw shapes

`_draw_shapes(...)` handles Plotly layout shapes.

Currently implemented:

- `line`
- `rect`
- `circle`

This is important because Graphik uses shapes for some grid fallbacks and plot decorations.

### Stage 6: draw annotations

`_draw_annotations(...)` converts Plotly annotations into Matplotlib text objects.

Supported aspects include:

- `xref` / `yref`
- `paper` vs data coordinates
- x/y shifts
- font size / color
- background box
- border color / width
- anchors and alignment

### Stage 7: draw legend

`_draw_legend(...)` translates Plotly legend configuration into a Matplotlib legend.

Special handling exists for horizontal legends:

- a dedicated top band is reserved
- the subplot top margin is increased if necessary
- legend placement is constrained to stay inside the figure

This was added because otherwise top legends were clipped in PDF export.

### Stage 8: fit decorative elements into the export canvas

`_ensure_export_decorations_fit(...)` performs an important post-layout correction.

It renders once with `FigureCanvasAgg`, measures bounding boxes for:

- axis labels
- tick labels
- title
- offset texts
- legend

Then it expands subplot margins if any decoration would fall outside the figure.

This step is critical for solving clipping issues such as:

- missing y-axis title in PDF
- cut-off legend at the top or left

### Stage 9: save to bytes

The export backend finally calls `savefig(...)` into a `BytesIO` buffer.

Format-specific notes:

- `png`
  - uses explicit `dpi * scale`
- `svg`
  - exports text as text (`svg.fonttype = none`)
- `pdf`
  - uses embedded TrueType-compatible font settings (`pdf.fonttype = 42`)

After saving:

- bytes are returned to Streamlit
- the Matplotlib figure is cleared to reduce memory retention

## 10.6 Gridline Styling in Export

The preview and export engines do not render transparency identically.

As a result, a gray grid that looks light in the browser can appear too dark in vector export.

To compensate, [src/mpl_export.py](D:\grafiken\src\mpl_export.py) now applies `_soften_grid_style(...)` during export.

This function:

- blends grid color slightly toward white
- reduces alpha
- reduces linewidth slightly

It applies to:

- major gridlines
- minor gridlines
- explicit semilog gridline shapes

This is an export-only correction. The Plotly preview remains unchanged.

---

## 11. Summary Export

Graphik also exports textual summaries.

### Normal mode summary

Built by:
- `_build_summary_text(...)`

Contains:

- raw data
- normalized analysis data
- fit results
- error-line results
- final slope presentation

### Statistics mode summary

Built by:
- `_build_statistics_summary_text(...)`

Contains:

- raw table snapshot
- extracted numeric values
- histogram settings
- descriptive statistics
- optional normal-fit information

These exports are intended to reduce manual copying into a protocol/report.

---

## 12. Failure Modes and Current Safeguards

This section is intentionally explicit because it is often the most useful part for expert feedback.

### 12.1 File import failures

Possible causes:

- unsupported format
- malformed spreadsheet content
- non-tabular data layout

Current mitigations:

- CSV delimiter/decimal auto-detection
- ODS engine support
- header-promotion heuristic
- unnamed-column fallback naming
- explicit exceptions on unsupported suffixes

### 12.2 Numeric validation failures

Possible causes:

- text left inside numeric body rows
- unit strings inside data rows
- formulas exported as text

Current mitigations:

- strict numeric conversion with precise error message
- trimming to first fully numeric data row in the required columns

### 12.3 Log-axis invalidity

Possible causes:

- non-positive y values
- lower error bar reaching zero or below
- custom log range with non-positive bounds

Current mitigations:

- hard stop if y contains non-positive values in log mode
- info message when lower error bars must be clipped visually
- lower log error bars are clipped to positive floor for preview

### 12.4 Strict centroid error lines unavailable

Possible causes:

- small uncertainties
- centroid not lying inside all admissible intervals
- no common slope interval intersection

Current mitigation:

- show a warning instead of crashing
- keep the rest of the analysis and plotting functional

### 12.5 Export clipping

Historically this was a real issue.

Current mitigations:

- dedicated legend-top band in export
- post-layout bbox measurement for labels/ticks/legend
- margin correction before `savefig`

### 12.6 Export performance

Possible issue:

- large export generation can be expensive

Current mitigation:

- image export is on-demand, not recomputed on every UI interaction
- export bytes are cached by signature until relevant inputs change

### 12.7 Build/runtime issues in packaged Windows version

Possible causes:

- missing hidden imports
- missing metadata
- stale/broken output folder during rebuild
- app already running and locking the bundle folder

Current mitigations:

- PyInstaller spec includes explicit hidden imports and metadata
- launcher detects existing running instance
- rebuild process requires closing the running packaged app first

---

## 13. Test Coverage Relevant to Mechanics

The test suite includes checks for:

- regression calculations
- error-line compatibility and bounds
- export-format generation (`png`, `svg`, `pdf`)
- histogram export
- text-size scaling behavior
- separation of text scaling vs non-text scaling
- major/minor grid preservation
- x-log export handling
- legend staying inside the figure
- y-axis label staying inside the figure
- softened export grid style

This does not prove the app is bug-free, but it does provide regression protection around the most failure-prone areas of the pipeline.

---

## 14. Current Design Tradeoffs

These are not bugs, but deliberate choices that may be worth expert review.

1. Plotly preview, Matplotlib export
   - improves export stability
   - but requires a translation layer and therefore constant synchronization effort

2. UI-level simplification of uncertainty derivation
   - easier for users
   - but the codebase still contains latent derivation paths not currently used in the UI

3. Strict centroid error-line method retained
   - mathematically defensible
   - but often unusable for real data

4. Auto-placement of the plot info box
   - better than fixed placement in many cases
   - but still heuristic and not guaranteed optimal for every dataset

5. Session snapshot persistence
   - improves usability
   - but adds state complexity and requires careful exclusion of transient widget/export values

---

## 15. Suggested Questions for Expert Feedback

If this document is being sent to an expert reviewer, the most useful questions are likely:

1. Is the current separation between Plotly preview and Matplotlib export the right long-term architecture?
2. Is the standard endpoint method the right default for subjective error lines in the target domain?
3. Should the currently dormant y/sigma derivation functionality be removed completely or reintroduced as an optional preprocessing module?
4. Are the current heuristics for plot-info-box placement acceptable, or should that subsystem become fully manual?
5. Is the current approach to semilog paper rendering sufficiently faithful for teaching/lab usage?
6. Are there additional export invariants that should be enforced by tests, especially regarding typography and legend placement?
7. Is there a cleaner abstraction that could further reduce divergence risk between preview and export?

---

## 16. Concise End-to-End Flow

For completeness, the operational flow is:

1. Start app
2. Restore saved settings
3. Load sample data or uploaded table
4. Sanitize dataframe and infer headers if necessary
5. Let user map x/y/error columns
6. Validate and normalize data into `x`, `y`, `sigma_y`
7. Build analysis state
   - fit model
   - error-line method
   - labels
   - axis/grid settings
   - triangle settings
8. Build Plotly preview figure
9. Add analytical overlays and info box
10. Show scientific output panel
11. On export request:
    - clone/adapt figure
    - optionally autoscale axes for export
    - scale text and non-text elements appropriately
    - render via Matplotlib
    - fit decorations within page bounds
    - write PNG/SVG/PDF bytes
12. Offer download to user

That is the current implemented mechanism of Graphik.

## Export backend support contract

The Matplotlib export backend intentionally supports only a defined subset of preview features: scatter traces with markers/lines and optional error bars, vertical bar traces, shape types line/rect/circle, text annotations without arrows, and linear/log x/y axes. If a preview figure contains unsupported elements, export now fails fast with a clear "unsupported feature in export backend" validation message instead of silently generating an incomplete artifact.
