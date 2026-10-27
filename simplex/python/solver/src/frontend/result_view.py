from dataclasses import dataclass

import numpy as np

from PySide6.QtWidgets import (
    QWidget,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QScrollArea,
)

from core import (
    SolveResult,
    OptimalResult,
    UnboundedResult,
    InfeasibleResult,
    GeneralCertificate,
)


@dataclass
class SolverRun:
    name: str
    result: SolveResult | None
    elapsed_seconds: float
    error: Exception | None = None


class SolverResultPanel(QGroupBox):
    def __init__(self, title: str):
        super().__init__(title)

        self.content_layout = QVBoxLayout(self)
        self.content_layout.setContentsMargins(14, 18, 14, 14)
        self.content_layout.setSpacing(10)

        self._show_placeholder()

    def show_run(self, run: SolverRun):
        self.setTitle(run.name)
        self._clear()

        self._add_secondary("Solve time")

        time_label = QLabel(self._format_time(run.elapsed_seconds))

        time_font = time_label.font()
        time_font.setPointSize(time_font.pointSize() + 2)
        time_font.setBold(True)
        time_label.setFont(time_font)

        self.content_layout.addWidget(time_label)

        if run.error is not None:
            self._add_status("ERROR")

            error_label = QLabel(str(run.error))
            error_label.setWordWrap(True)

            self.content_layout.addWidget(error_label)
            self.content_layout.addStretch()
            return

        if run.result is None:
            self._add_status("NO RESULT")
            self.content_layout.addStretch()
            return

        result = run.result

        if isinstance(result, OptimalResult):
            self._show_optimal(result)

        elif isinstance(result, UnboundedResult):
            self._show_unbounded(result)

        elif isinstance(result, InfeasibleResult):
            self._show_infeasible(result)

        else:
            self._add_status("UNKNOWN RESULT")

            self.content_layout.addWidget(
                QLabel(type(result).__name__)
            )

        self.content_layout.addStretch()

    def _show_optimal(self, result: OptimalResult):
        self._add_status("OPTIMAL")

        self._add_secondary("Objective value")

        objective_label = QLabel(self._format_number(result.objective_value))

        font = objective_label.font()
        font.setPointSize(font.pointSize() + 4)
        font.setBold(True)
        objective_label.setFont(font)

        self.content_layout.addWidget(objective_label)

        self._add_vector(
            "Solution",
            result.solution,
            "x",
        )

        self._add_secondary("SEF basis")

        basis_names = ", ".join(
            f"x{i + 1}"
            for i in result.basis.indices
        )

        self.content_layout.addWidget(
            QLabel(basis_names)
        )

        if isinstance(result.certificate, GeneralCertificate):
            self._show_general_certificate(result.certificate)
        else:
            self._add_vector(
                "Optimality certificate",
                result.certificate,
                "y",
            )

    def _show_unbounded(self, result: UnboundedResult):
        self._add_status("UNBOUNDED")

        self._add_vector(
            "Feasible point",
            result.solution,
            "x",
        )

        self._add_vector(
            "Unbounded direction",
            result.direction,
            "d",
        )

    def _show_infeasible(self, result: InfeasibleResult):
        self._add_status("INFEASIBLE")

        if isinstance(result.certificate, GeneralCertificate):
            self._show_general_certificate(result.certificate)
        else:
            self._add_vector(
                "Infeasibility certificate",
                result.certificate,
                "y",
            )

    def _show_general_certificate(self, certificate: GeneralCertificate):
        self._add_vector(
            "Constraint multipliers",
            certificate.constraint_multipliers,
            "y",
        )

        self._add_vector(
            "Lower-bound multipliers",
            certificate.lower_bound_multipliers,
            "l",
        )

        self._add_vector(
            "Upper-bound multipliers",
            certificate.upper_bound_multipliers,
            "u",
        )

    def _add_status(self, text: str):
        label = QLabel(text)
        label.setObjectName("statusLabel")

        self.content_layout.addWidget(label)

    def _add_secondary(self, text: str):
        label = QLabel(text)
        label.setObjectName("secondaryLabel")

        self.content_layout.addWidget(label)

    def _add_vector(self, title: str, values, prefix: str):
        values = np.asarray(values).reshape(-1)

        self._add_secondary(title)

        if len(values) == 0:
            self.content_layout.addWidget(QLabel("None"))
            return

        table = QTableWidget(1, len(values))

        table.verticalHeader().setVisible(False)
        table.setHorizontalHeaderLabels([f"{prefix}{i + 1}" for i in range(len(values))])

        for i, value in enumerate(values):
            table.setItem(
                0,
                i,
                QTableWidgetItem(self._format_number(value)),
            )

        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)

        table.setFixedHeight(72)

        self.content_layout.addWidget(table)

    def _show_placeholder(self):
        label = QLabel("No result yet")
        label.setObjectName("secondaryLabel")

        self.content_layout.addWidget(label)
        self.content_layout.addStretch()

    def _clear(self):
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)

            if item is None:
                continue

            widget = item.widget()

            if widget is not None:
                widget.deleteLater()

    @staticmethod
    def _format_number(value) -> str:
        value = float(value)

        if abs(value) < 1e-12:
            value = 0.0

        return f"{value:.10g}"

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 1e-6:
            return f"{seconds * 1e9:.3f} ns"

        if seconds < 1e-3:
            return f"{seconds * 1e6:.3f} µs"

        if seconds < 1:
            return f"{seconds * 1e3:.3f} ms"

        return f"{seconds:.3f} s"


class SolverComparisonView(QGroupBox):
    def __init__(self):
        super().__init__("Solver Comparison")

        self.canonical_panel = SolverResultPanel("Canonical")
        self.tableau_panel = SolverResultPanel("Tableau")

        self.summary_label = QLabel("Enter a problem and press Solve.")
        self.summary_label.setObjectName("comparisonLabel")
        self.summary_label.setWordWrap(True)

        result_widget = QWidget()
        result_layout = QHBoxLayout(result_widget)

        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(12)

        result_layout.addWidget(self.canonical_panel)
        result_layout.addWidget(self.tableau_panel)

        result_layout.setStretch(0, 1)
        result_layout.setStretch(1, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(result_widget)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setSpacing(12)

        layout.addWidget(self.summary_label)
        layout.addWidget(scroll)

    def show_runs(self, canonical_run: SolverRun, tableau_run: SolverRun):
        self.canonical_panel.show_run(canonical_run)
        self.tableau_panel.show_run(tableau_run)

        self.summary_label.setText(
            self._build_summary(canonical_run, tableau_run)
        )

    def _build_summary(self, canonical_run: SolverRun, tableau_run: SolverRun) -> str:
        if canonical_run.error is not None or tableau_run.error is not None:
            errors = []

            if canonical_run.error is not None:
                errors.append(f"Canonical failed: {canonical_run.error}")

            if tableau_run.error is not None:
                errors.append(f"Tableau failed: {tableau_run.error}")

            return " | ".join(errors)

        canonical_result = canonical_run.result
        tableau_result = tableau_run.result

        if canonical_result is None or tableau_result is None:
            return "Could not compare solver results."

        messages = []

        if type(canonical_result) is type(tableau_result):
            if isinstance(canonical_result, OptimalResult) and isinstance(tableau_result, OptimalResult):
                same_objective = np.isclose(
                    canonical_result.objective_value,
                    tableau_result.objective_value,
                    atol=1e-8,
                    rtol=1e-8,
                )

                same_solution = np.allclose(
                    canonical_result.solution,
                    tableau_result.solution,
                    atol=1e-8,
                    rtol=1e-8,
                )

                if same_objective and same_solution:
                    messages.append("Both solvers returned the same optimum and solution.")
                elif same_objective:
                    messages.append("Both solvers returned the same optimum but different optimal solutions.")
                else:
                    messages.append("WARNING: the solvers returned different objective values.")

            elif isinstance(canonical_result, UnboundedResult):
                messages.append("Both solvers report the LP as unbounded.")

            elif isinstance(canonical_result, InfeasibleResult):
                messages.append("Both solvers report the LP as infeasible.")

        else:
            messages.append(
                f"WARNING: result types differ: {type(canonical_result).__name__} vs {type(tableau_result).__name__}."
            )

        canonical_time = canonical_run.elapsed_seconds
        tableau_time = tableau_run.elapsed_seconds

        if canonical_time > 0 and tableau_time > 0:
            if canonical_time < tableau_time:
                ratio = tableau_time / canonical_time
                messages.append(f"Canonical was {ratio:.2f}× faster on this run.")

            elif tableau_time < canonical_time:
                ratio = canonical_time / tableau_time
                messages.append(f"Tableau was {ratio:.2f}× faster on this run.")

            else:
                messages.append("Both solvers took approximately the same time.")

        return " ".join(messages)