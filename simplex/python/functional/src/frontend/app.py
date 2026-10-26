from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QSplitter,
    QMessageBox,
)

from core.solver import solve

from .sef_input import SEFInputWidget
from .result_view import ResultView


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Simplex Solver")
        self.resize(1250, 760)
        self.setMinimumSize(950, 650)

        self.sef_input = SEFInputWidget()
        self.result_view = ResultView()

        self.sef_input.solve_requested.connect(self._solve)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        splitter.addWidget(self.sef_input)
        splitter.addWidget(self.result_view)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        splitter.setSizes([780, 420])
        splitter.setChildrenCollapsible(False)

        self.result_view.setMinimumWidth(320)

        central = QWidget()

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.addWidget(splitter)

        self.setCentralWidget(central)

    def _solve(self, lp):
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