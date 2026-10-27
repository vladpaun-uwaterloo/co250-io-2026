import numpy as np

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QMessageBox,
    QScrollArea,
)

from core import SEFLinearProgram

from .components import SectionCard, fit_table_height


class SEFInputWidget(QWidget):
    solve_requested = Signal(object)

    def __init__(self):
        super().__init__()

        self.variable_count = QSpinBox()
        self.variable_count.setRange(1, 100)
        self.variable_count.setValue(2)
        self.variable_count.setFixedWidth(90)

        self.constraint_count = QSpinBox()
        self.constraint_count.setRange(1, 100)
        self.constraint_count.setValue(2)
        self.constraint_count.setFixedWidth(90)

        self.objective_table = QTableWidget()
        self.constraint_table = QTableWidget()

        self.offset_input = QLineEdit("0")
        self.offset_input.setFixedWidth(130)

        self.solve_button = QPushButton("Solve & Compare")
        self.solve_button.setObjectName("primaryButton")
        self.solve_button.setFixedHeight(42)

        self._configure_tables()
        self._build_ui()

        self.variable_count.valueChanged.connect(self._resize_tables)
        self.constraint_count.valueChanged.connect(self._resize_tables)
        self.solve_button.clicked.connect(self._solve)

        self._resize_tables()

    def _configure_tables(self):
        for table in (self.objective_table, self.constraint_table):
            table.horizontalHeader().setFixedHeight(34)
            table.verticalHeader().setDefaultSectionSize(36)
            table.setAlternatingRowColors(True)
            table.setShowGrid(True)

        self.objective_table.verticalHeader().setVisible(False)

    def _build_ui(self):
        content = QWidget()

        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(16, 16, 16, 20)
        content_layout.setSpacing(14)

        size_card = SectionCard(
            "Problem size",
            "Choose the dimensions of the standard equality form problem.",
        )

        size_row = QHBoxLayout()
        size_row.setSpacing(12)

        size_row.addWidget(QLabel("Variables"))
        size_row.addWidget(self.variable_count)

        size_row.addSpacing(16)

        size_row.addWidget(QLabel("Constraints"))
        size_row.addWidget(self.constraint_count)

        size_row.addStretch()

        size_card.content_layout.addLayout(size_row)
        content_layout.addWidget(size_card)

        objective_card = SectionCard(
            "Objective",
            "Maximize  cᵀx + offset",
        )

        objective_card.content_layout.addWidget(self.objective_table)

        offset_row = QHBoxLayout()
        offset_row.addWidget(QLabel("Objective offset"))
        offset_row.addWidget(self.offset_input)
        offset_row.addStretch()

        objective_card.content_layout.addLayout(offset_row)

        content_layout.addWidget(objective_card)

        constraints_card = SectionCard(
            "Constraints",
            "Ax = b,   x ≥ 0",
        )

        constraints_card.content_layout.addWidget(self.constraint_table)
        content_layout.addWidget(constraints_card)

        content_layout.addWidget(self.solve_button)
        content_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

    def _resize_tables(self):
        old_n = self.objective_table.columnCount()
        old_m = self.constraint_table.rowCount()

        old_objective = [
            self._cell_text(self.objective_table, 0, j)
            for j in range(old_n)
        ]

        old_A = [
            [
                self._cell_text(self.constraint_table, i, j)
                for j in range(old_n)
            ]
            for i in range(old_m)
        ]

        old_b = []

        if self.constraint_table.columnCount() == old_n + 1:
            old_b = [
                self._cell_text(self.constraint_table, i, old_n)
                for i in range(old_m)
            ]

        n = self.variable_count.value()
        m = self.constraint_count.value()

        self.objective_table.setRowCount(1)
        self.objective_table.setColumnCount(n)
        self.objective_table.setHorizontalHeaderLabels([f"x{j + 1}" for j in range(n)])

        for j in range(n):
            value = old_objective[j] if j < len(old_objective) else "0"
            self.objective_table.setItem(0, j, QTableWidgetItem(value))

        self.objective_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        fit_table_height(self.objective_table)

        self.constraint_table.setRowCount(m)
        self.constraint_table.setColumnCount(n + 1)
        self.constraint_table.setHorizontalHeaderLabels([f"x{j + 1}" for j in range(n)] + ["b"])
        self.constraint_table.setVerticalHeaderLabels([str(i + 1) for i in range(m)])

        for i in range(m):
            for j in range(n):
                value = old_A[i][j] if i < old_m and j < old_n else "0"
                self.constraint_table.setItem(i, j, QTableWidgetItem(value))

            rhs = old_b[i] if i < len(old_b) else "0"
            self.constraint_table.setItem(i, n, QTableWidgetItem(rhs))

        self.constraint_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        fit_table_height(self.constraint_table, visible_rows=7)

    def get_lp(self) -> SEFLinearProgram:
        n = self.variable_count.value()
        m = self.constraint_count.value()

        c = np.array(
            [
                self._read_finite(self.objective_table, 0, j, f"c[{j + 1}]")
                for j in range(n)
            ],
            dtype=float,
        )

        A = np.empty((m, n), dtype=float)
        b = np.empty(m, dtype=float)

        for i in range(m):
            for j in range(n):
                A[i, j] = self._read_finite(
                    self.constraint_table,
                    i,
                    j,
                    f"A[{i + 1}, {j + 1}]",
                )

            b[i] = self._read_finite(
                self.constraint_table,
                i,
                n,
                f"b[{i + 1}]",
            )

        try:
            objective_offset = float(self.offset_input.text().strip())
        except ValueError:
            raise ValueError("Objective offset must be a number.")

        if not np.isfinite(objective_offset):
            raise ValueError("Objective offset must be finite.")

        return SEFLinearProgram(
            c=c,
            A=A,
            b=b,
            objective_offset=objective_offset,
        )

    def _solve(self):
        try:
            lp = self.get_lp()
        except ValueError as exc:
            QMessageBox.warning(
                self,
                "Invalid Input",
                str(exc),
            )
            return

        self.solve_requested.emit(lp)

    @staticmethod
    def _cell_text(table: QTableWidget, row: int, column: int) -> str:
        item = table.item(row, column)

        if item is None:
            return "0"

        text = item.text().strip()
        return text if text else "0"

    @staticmethod
    def _read_finite(table: QTableWidget, row: int, column: int, name: str) -> float:
        text = SEFInputWidget._cell_text(table, row, column)

        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"{name} must be a number.")

        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")

        return value