from time import perf_counter

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QTabWidget,
    QSplitter,
)

from core import CanonicalSimplexSolver, TableauSimplexSolver

from .sef_input import SEFInputWidget
from .general_input import GeneralLPInputWidget
from .result_view import SolverComparisonView, SolverRun


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Simplex Lab")
        self.resize(1550, 900)
        self.setMinimumSize(1250, 700)

        self.canonical_solver = CanonicalSimplexSolver()
        self.tableau_solver = TableauSimplexSolver()

        self.sef_input = SEFInputWidget()
        self.general_input = GeneralLPInputWidget()
        self.result_view = SolverComparisonView()

        self.sef_input.solve_requested.connect(self._solve)
        self.general_input.solve_requested.connect(self._solve)

        tabs = QTabWidget()
        tabs.setObjectName("problemTabs")
        tabs.setMinimumWidth(620)

        tabs.addTab(self.sef_input, "Standard Equality Form")
        tabs.addTab(self.general_input, "General LP")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("mainSplitter")

        splitter.addWidget(tabs)
        splitter.addWidget(self.result_view)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        splitter.setSizes([650, 900])
        splitter.setChildrenCollapsible(False)

        title = QLabel("Simplex Lab")
        title.setObjectName("appTitle")

        subtitle = QLabel(
            "Build a linear program and compare the canonical and tableau simplex implementations."
        )
        subtitle.setObjectName("appSubtitle")

        central = QWidget()

        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 16, 20, 20)
        layout.setSpacing(14)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(splitter, 1)

        self.setCentralWidget(central)

    def _solve(self, lp):
        canonical_run = self._run_solver(
            "Canonical",
            self.canonical_solver,
            lp,
        )

        tableau_run = self._run_solver(
            "Tableau",
            self.tableau_solver,
            lp,
        )

        self.result_view.show_runs(
            canonical_run,
            tableau_run,
        )

    @staticmethod
    def _run_solver(name, solver, lp) -> SolverRun:
        start = perf_counter()

        try:
            result = solver.solve(lp)
            elapsed = perf_counter() - start

            return SolverRun(
                name=name,
                result=result,
                elapsed_seconds=elapsed,
            )

        except Exception as exc:
            elapsed = perf_counter() - start

            return SolverRun(
                name=name,
                result=None,
                elapsed_seconds=elapsed,
                error=exc,
            )