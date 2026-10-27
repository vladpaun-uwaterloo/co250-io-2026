from dataclasses import dataclass

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QSplitter,
    QFrame,
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


class SolverResultPanel(QWidget):
    def __init__(self, name: str):
        super().__init__()

        self.setObjectName("resultCard")

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 18, 20, 20)
        self.main_layout.setSpacing(12)

        self.header_layout = QHBoxLayout()

        self.name_label = QLabel(name)
        self.name_label.setObjectName("solverName")

        self.time_label = QLabel("—")
        self.time_label.setObjectName("timeBadge")

        self.header_layout.addWidget(self.name_label)
        self.header_layout.addStretch()
        self.header_layout.addWidget(self.time_label)

        self.main_layout.addLayout(self.header_layout)

        self.status_label = QLabel("Not run")
        self.status_label.setObjectName("resultStatus")

        self.main_layout.addWidget(self.status_label)

        self.content = QWidget()

        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 4, 0, 0)
        self.content_layout.setSpacing(12)

        self.main_layout.addWidget(self.content)
        self.main_layout.addStretch()

    def show_run(self, run: SolverRun):
        self.name_label.setText(run.name)
        self.time_label.setText(
            self._format_time(run.elapsed_seconds)
        )

        self._clear_content()

        if run.error is not None:
            self.status_label.setText("Error")
            self.status_label.setProperty("state", "error")
            self._refresh_style(self.status_label)

            message = QLabel(str(run.error))
            message.setObjectName("errorText")
            message.setWordWrap(True)

            self.content_layout.addWidget(message)
            return

        result = run.result

        if isinstance(result, OptimalResult):
            self.status_label.setText("Optimal")
            self.status_label.setProperty("state", "success")
            self._refresh_style(self.status_label)

            self._show_optimal(result)
            return

        if isinstance(result, UnboundedResult):
            self.status_label.setText("Unbounded")
            self.status_label.setProperty("state", "warning")
            self._refresh_style(self.status_label)

            self._show_unbounded(result)
            return

        if isinstance(result, InfeasibleResult):
            self.status_label.setText("Infeasible")
            self.status_label.setProperty("state", "warning")
            self._refresh_style(self.status_label)

            self._show_infeasible(result)
            return

        self.status_label.setText("Unknown result")
        self.status_label.setProperty("state", "error")
        self._refresh_style(self.status_label)

    def _show_optimal(self, result: OptimalResult):
        self._add_label("Objective value")

        objective = QLabel(
            self._format_number(result.objective_value)
        )
        objective.setObjectName("objectiveValue")

        self.content_layout.addWidget(objective)

        self._add_vector(
            "Solution",
            result.solution,
            "x",
        )

        self._add_label("SEF basis")

        basis = QLabel(
            ", ".join(
                f"x{i + 1}"
                for i in result.basis.indices
            )
        )

        basis.setObjectName("valueText")
        self.content_layout.addWidget(basis)

        if isinstance(
            result.certificate,
            GeneralCertificate,
        ):
            self._show_general_certificate(
                result.certificate
            )
        else:
            self._add_vector(
                "Optimality certificate",
                result.certificate,
                "y",
            )

    def _show_unbounded(
        self,
        result: UnboundedResult,
    ):
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

    def _show_infeasible(
        self,
        result: InfeasibleResult,
    ):
        if isinstance(
            result.certificate,
            GeneralCertificate,
        ):
            self._show_general_certificate(
                result.certificate
            )
        else:
            self._add_vector(
                "Infeasibility certificate",
                result.certificate,
                "y",
            )

    def _show_general_certificate(
        self,
        certificate: GeneralCertificate,
    ):
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

    def _add_label(self, text: str):
        label = QLabel(text)
        label.setObjectName("fieldLabel")

        self.content_layout.addWidget(label)

    def _add_vector(
        self,
        title: str,
        values,
        prefix: str,
    ):
        values = np.asarray(values).reshape(-1)

        self._add_label(title)

        table = QTableWidget(
            1,
            len(values),
        )

        table.setObjectName("resultTable")
        table.verticalHeader().setVisible(False)

        table.setHorizontalHeaderLabels(
            [
                f"{prefix}{i + 1}"
                for i in range(len(values))
            ]
        )

        for i, value in enumerate(values):
            table.setItem(
                0,
                i,
                QTableWidgetItem(
                    self._format_number(value)
                ),
            )

        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )

        table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )

        table.setFixedHeight(72)

        self.content_layout.addWidget(table)

    def _clear_content(self):
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)

            if item is None:
                continue

            widget = item.widget()

            if widget is not None:
                widget.deleteLater()

    @staticmethod
    def _refresh_style(widget):
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    @staticmethod
    def _format_number(value) -> str:
        value = float(value)

        if abs(value) < 1e-12:
            value = 0.0

        return f"{value:.10g}"

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 1e-6:
            return f"{seconds * 1e9:.2f} ns"

        if seconds < 1e-3:
            return f"{seconds * 1e6:.2f} µs"

        if seconds < 1:
            return f"{seconds * 1e3:.2f} ms"

        return f"{seconds:.3f} s"


class SolverComparisonView(QWidget):
    def __init__(self):
        super().__init__()

        self.canonical_panel = SolverResultPanel(
            "Canonical"
        )

        self.tableau_panel = SolverResultPanel(
            "Tableau"
        )

        title = QLabel("Solver comparison")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Both implementations solve the same problem independently."
        )
        subtitle.setObjectName("pageSubtitle")

        self.summary = QFrame()
        self.summary.setObjectName("summaryCard")

        summary_layout = QVBoxLayout(self.summary)
        summary_layout.setContentsMargins(
            16,
            12,
            16,
            12,
        )

        self.summary_label = QLabel(
            "Enter a problem and run both solvers."
        )

        self.summary_label.setObjectName(
            "summaryText"
        )
        self.summary_label.setWordWrap(True)

        summary_layout.addWidget(
            self.summary_label
        )

        splitter = QSplitter(
            Qt.Orientation.Horizontal
        )

        splitter.setObjectName(
            "resultSplitter"
        )

        splitter.addWidget(
            self.canonical_panel
        )

        splitter.addWidget(
            self.tableau_panel
        )

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(
            18,
            18,
            18,
            18,
        )
        layout.setSpacing(12)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self.summary)
        layout.addWidget(splitter, 1)

    def show_runs(
        self,
        canonical_run: SolverRun,
        tableau_run: SolverRun,
    ):
        self.canonical_panel.show_run(
            canonical_run
        )

        self.tableau_panel.show_run(
            tableau_run
        )

        self.summary_label.setText(
            self._build_summary(
                canonical_run,
                tableau_run,
            )
        )

    def _build_summary(
        self,
        canonical_run: SolverRun,
        tableau_run: SolverRun,
    ) -> str:
        if (
            canonical_run.error is not None
            or tableau_run.error is not None
        ):
            errors = []

            if canonical_run.error is not None:
                errors.append(
                    f"Canonical failed: {canonical_run.error}"
                )

            if tableau_run.error is not None:
                errors.append(
                    f"Tableau failed: {tableau_run.error}"
                )

            return "   •   ".join(errors)

        canonical = canonical_run.result
        tableau = tableau_run.result

        if canonical is None or tableau is None:
            return "Unable to compare results."

        messages = []

        if type(canonical) is not type(tableau):
            messages.append(
                f"Results disagree: {type(canonical).__name__} vs {type(tableau).__name__}."
            )

        elif isinstance(
            canonical,
            OptimalResult,
        ) and isinstance(
            tableau,
            OptimalResult,
        ):
            same_objective = np.isclose(
                canonical.objective_value,
                tableau.objective_value,
                atol=1e-8,
                rtol=1e-8,
            )

            same_solution = np.allclose(
                canonical.solution,
                tableau.solution,
                atol=1e-8,
                rtol=1e-8,
            )

            if same_objective and same_solution:
                messages.append(
                    "Both solvers agree on the optimum and solution."
                )

            elif same_objective:
                messages.append(
                    "Both solvers agree on the optimum but found different optimal solutions."
                )

            else:
                messages.append(
                    "The solvers returned different objective values."
                )

        elif isinstance(
            canonical,
            UnboundedResult,
        ):
            messages.append(
                "Both solvers report the problem as unbounded."
            )

        elif isinstance(
            canonical,
            InfeasibleResult,
        ):
            messages.append(
                "Both solvers report the problem as infeasible."
            )

        canonical_time = (
            canonical_run.elapsed_seconds
        )

        tableau_time = (
            tableau_run.elapsed_seconds
        )

        if canonical_time > 0 and tableau_time > 0:
            if canonical_time < tableau_time:
                ratio = (
                    tableau_time
                    / canonical_time
                )

                messages.append(
                    f"Canonical was {ratio:.2f}× faster."
                )

            elif tableau_time < canonical_time:
                ratio = (
                    canonical_time
                    / tableau_time
                )

                messages.append(
                    f"Tableau was {ratio:.2f}× faster."
                )

        return "   •   ".join(messages)