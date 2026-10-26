from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QTabWidget,
    QSplitter,
    QMessageBox,
)

from core.solver import solve

from .sef_input import SEFInputWidget
from .general_input import GeneralLPInputWidget
from .result_view import ResultView


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Simplex Solver")
        self.resize(1000, 800)

        self.sef_input = SEFInputWidget()
        self.general_input = GeneralLPInputWidget()
        self.result_view = ResultView()

        self.sef_input.solve_requested.connect(self._solve_sef)

        tabs = QTabWidget()
        tabs.addTab(self.sef_input, "Standard Equality Form")
        tabs.addTab(self.general_input, "General LP")

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(tabs)
        splitter.addWidget(self.result_view)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(splitter)

        self.setCentralWidget(central)

    def _solve_sef(self, lp):
        try:
            result = solve(lp)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Solver Error",
                str(exc),
            )
            return

        self.result_view.show_result(result)