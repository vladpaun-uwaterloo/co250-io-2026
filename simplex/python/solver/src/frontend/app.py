from time import perf_counter

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
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

        self.setWindowTitle("Simplex Solver Comparison")
        self.resize(1500, 820)
        self.setMinimumSize(1100, 700)

        self.canonical_solver = CanonicalSimplexSolver()
        self.tableau_solver = TableauSimplexSolver()

        self.sef_input = SEFInputWidget()
        self.general_input = GeneralLPInputWidget()
        self.result_view = SolverComparisonView()

        self.sef_input.solve_requested.connect(self._solve)
        self.general_input.solve_requested.connect(self._solve)

        tabs = QTabWidget()
        tabs.addTab(self.sef_input, "Standard Equality Form")
        tabs.addTab(self.general_input, "General LP")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(tabs)
        splitter.addWidget(self.result_view)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([600, 900])
        splitter.setChildrenCollapsible(False)

        central = QWidget()

        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(splitter)

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