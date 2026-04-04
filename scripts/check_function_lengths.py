"""Check internal function-size and branching-complexity baselines."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Final

MAX_FUNCTION_LINES: Final[int] = 120
MAX_BRANCH_COMPLEXITY: Final[int] = 8
EXCLUDE_DIRS: Final[set[str]] = {'.venv', 'build', 'dist', 'dist_portable', '__pycache__'}
TARGET_ROOTS: Final[tuple[str, ...]] = ('app.py', 'pages', 'services', 'src')
BASELINE_ALLOWLIST: Final[set[str]] = {
    'pages.import_wizard.render_import_wizard',
    'pages.normal_mode.render_normal_controls',
    'pages.normal_mode.render_normal_mode',
    'pages.statistics_mode.render_statistics_mode',
    'services.analysis_service.analyze_statistics_mode',
    'services.analysis_service.build_normal_analysis_result',
    'services.validation_service.suggest_column_mapping',
    'services.validation_service.suggest_column_mapping._match',
    'services.validation_service.collect_import_wizard_problems',
    'services.validation_service.collect_normal_mode_problems',
    'src.calculations.centroid_error_lines',
    'src.data_io._promote_first_row_as_header_if_likely',
    'src.export_pipeline.render_export_settings',
    'src.export_pipeline._axis_range_is_valid',
    'src.export_pipeline._validate_axis_positive_for_log',
    'src.export_pipeline._validate_figure',
    'src.export_pipeline.validate_export_request',
    'src.export_utils.add_plot_text_block',
    'src.export_utils.add_plot_text_block._collect_normalized_points',
    'src.export_utils.add_plot_text_block._extract_axis_bounds',
    'src.export_utils.add_plot_text_block._candidate_key',
    'src.export_utils.remove_plot_text_block',
    'src.export_utils.scale_figure_for_export',
    'src.export_utils.autoscale_figure_to_data',
    'src.mpl_export.validate_supported_export_features',
    'src.mpl_export._is_gridline_shape',
    'src.mpl_export._error_arrays',
    'src.mpl_export._apply_axis_format',
    'src.mpl_export._apply_layout',
    'src.mpl_export._plot_scatter',
    'src.mpl_export._draw_shape',
    'src.mpl_export._draw_annotation',
    'src.mpl_export._draw_legend',
    'src.mpl_export._ensure_export_decorations_fit',
    'src.mpl_export._ensure_export_decorations_fit._append_artist_bbox',
    'src.plotting._add_semilog_paper_gridlines',
    'src.plotting.create_base_figure',
    'src.ui_state._normalize_json_value',
    'src.ui_state.init_session_state',
}
REPORT_PATH: Final[Path] = Path('.streamlit') / 'code_quality_report.txt'


@dataclass(frozen=True)
class FunctionMetrics:
    qualified_name: str
    path: Path
    line_count: int
    branch_complexity: int


def iter_python_files() -> list[Path]:
    files: list[Path] = []
    for root in TARGET_ROOTS:
        path = Path(root)
        if path.is_file():
            files.append(path)
            continue
        for file_path in path.rglob('*.py'):
            if any(part in EXCLUDE_DIRS for part in file_path.parts):
                continue
            files.append(file_path)
    return sorted(files)


def _end_lineno(node: ast.AST) -> int:
    end_lineno = getattr(node, 'end_lineno', None)
    if end_lineno is None:
        return int(getattr(node, 'lineno', 0))
    return int(end_lineno)


def _branch_complexity(node: ast.AST) -> int:
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.Match, ast.ExceptHandler, ast.With, ast.AsyncWith, ast.IfExp)):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += max(0, len(child.values) - 1)
        elif isinstance(child, ast.comprehension):
            complexity += 1 + len(child.ifs)
    return complexity


def collect_metrics(path: Path) -> list[FunctionMetrics]:
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    metrics: list[FunctionMetrics] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def _qualified_name(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
            module_name = str(path.with_suffix('')).replace('\\', '.').replace('/', '.')
            joined = '.'.join(self.stack + [node.name])
            return f'{module_name}.{joined}'

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def _record(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            qualified_name = self._qualified_name(node)
            line_count = _end_lineno(node) - int(node.lineno) + 1
            metrics.append(
                FunctionMetrics(
                    qualified_name=qualified_name,
                    path=path,
                    line_count=line_count,
                    branch_complexity=_branch_complexity(node),
                )
            )

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._record(node)
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._record(node)
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

    Visitor().visit(tree)
    return metrics


def write_report(metrics: list[FunctionMetrics]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'Internal limits: {MAX_FUNCTION_LINES} lines, branch complexity {MAX_BRANCH_COMPLEXITY}',
        '',
        'Functions exceeding at least one limit:',
    ]
    violations = [m for m in metrics if m.line_count > MAX_FUNCTION_LINES or m.branch_complexity > MAX_BRANCH_COMPLEXITY]
    if not violations:
        lines.append('none')
    else:
        for item in sorted(violations, key=lambda metric: (metric.path.as_posix(), metric.qualified_name)):
            status = 'baseline' if item.qualified_name in BASELINE_ALLOWLIST else 'new'
            lines.append(
                f'- [{status}] {item.qualified_name} | lines={item.line_count} | branches={item.branch_complexity} | {item.path}'
            )
    REPORT_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> int:
    metrics = [metric for file_path in iter_python_files() for metric in collect_metrics(file_path)]
    write_report(metrics)
    violations = [
        metric
        for metric in metrics
        if (metric.line_count > MAX_FUNCTION_LINES or metric.branch_complexity > MAX_BRANCH_COMPLEXITY)
        and metric.qualified_name not in BASELINE_ALLOWLIST
    ]
    if not violations:
        print(f'Function metrics check passed. Report written to {REPORT_PATH}.')
        return 0
    print('Functions exceeding internal metrics baseline:')
    for violation in violations:
        print(
            f'- {violation.qualified_name} [lines={violation.line_count}, branches={violation.branch_complexity}] ({violation.path})'
        )
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
