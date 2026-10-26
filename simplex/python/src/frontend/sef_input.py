import numpy as np

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QSpinBox,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QGroupBox,
    QMessageBox,
)

from core.models import SEFLinearProgram


class SEFInputWidget(QWidget):
    solve_requested = Signal(object)

    def __init__(self):
        super().__init__()

        self.variable_count = QSpinBox()
        self.variable_count.setRange(1, 100)
        self.variable_count.setValue(2)

        self.constraint_count = QSpinBox()
        self.constraint_count.setRange(1, 100)
        self.constraint_count.setValue(2)

        self.objective_table = QTableWidget()
        self.constraint_table = QTableWidget()

        self.offset_input = QLineEdit("0")

        self.solve_button = QPushButton("Solve")

        self._build_ui()

        self.variable_count.valueChanged.connect(
            lambda _: self._resize_tables()
        )
        self.constraint_count.valueChanged.connect(
            lambda _: self._resize_tables()
        )

        self.solve_button.clicked.connect(self._solve)

        self._resize_tables()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        dimensions = QGroupBox("Problem Size")
        dimensions_layout = QFormLayout(dimensions)

        dimensions_layout.addRow(
            "Variables:",
            self.variable_count,
        )
        dimensions_layout.addRow(
            "Constraints:",
            self.constraint_count,
        )

        layout.addWidget(dimensions)

        objective_group = QGroupBox("Objective")
        objective_layout = QVBoxLayout(objective_group)

        objective_layout.addWidget(
            QLabel("Maximize  cᵀx + offset")
        )
        objective_layout.addWidget(self.objective_table)

        offset_layout = QHBoxLayout()
        offset_layout.addWidget(QLabel("Objective offset:"))
        offset_layout.addWidget(self.offset_input)
        offset_layout.addStretch()

        objective_layout.addLayout(offset_layout)

        layout.addWidget(objective_group)

        constraints_group = QGroupBox("Constraints")
        constraints_layout = QVBoxLayout(constraints_group)

        constraints_layout.addWidget(
            QLabel("Ax = b")
        )
        constraints_layout.addWidget(self.constraint_table)

        layout.addWidget(constraints_group)

        buttons = QHBoxLayout()
        buttons.addStretch()
        buttons.addWidget(self.solve_button)

        layout.addLayout(buttons)

    def _resize_tables(self):
        old_n = self.objective_table.columnCount()
        old_m = self.constraint_table.rowCount()

        objective_values = [
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
                self._cell_text(
                    self.constraint_table,
                    i,
                    old_n,
                )
                for i in range(old_m)
            ]

        n = self.variable_count.value()
        m = self.constraint_count.value()

        # Objective
        self.objective_table.setRowCount(1)
        self.objective_table.setColumnCount(n)

        self.objective_table.setHorizontalHeaderLabels(
            [f"x{j + 1}" for j in range(n)]
        )

        self.objective_table.setVerticalHeaderLabels(["c"])

        for j in range(n):
            value = (
                objective_values[j]
                if j < len(objective_values)
                else "0"
            )
            self._set_cell(
                self.objective_table,
                0,
                j,
                value,
            )

        self.objective_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        # Constraints
        self.constraint_table.setRowCount(m)
        self.constraint_table.setColumnCount(n + 1)

        headers = [
            f"x{j + 1}"
            for j in range(n)
        ] + ["b"]

        self.constraint_table.setHorizontalHeaderLabels(headers)

        self.constraint_table.setVerticalHeaderLabels(
            [str(i + 1) for i in range(m)]
        )

        for i in range(m):
            for j in range(n):
                value = "0"

                if i < old_m and j < old_n:
                    value = old_A[i][j]

                self._set_cell(
                    self.constraint_table,
                    i,
                    j,
                    value,
                )

            rhs = (
                old_b[i]
                if i < len(old_b)
                else "0"
            )

            self._set_cell(
                self.constraint_table,
                i,
                n,
                rhs,
            )

        self.constraint_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

    def get_lp(self) -> SEFLinearProgram:
        n = self.variable_count.value()
        m = self.constraint_count.value()

        c = np.array(
            [
                self._read_finite(
                    self.objective_table,
                    0,
                    j,
                    f"c[{j + 1}]",
                )
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
            objective_offset = float(
                self.offset_input.text().strip()
            )
        except ValueError:
            raise ValueError(
                "Objective offset must be a number."
            )

        if not np.isfinite(objective_offset):
            raise ValueError(
                "Objective offset must be finite."
            )

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
    def _cell_text(
        table: QTableWidget,
        row: int,
        column: int,
    ) -> str:
        item = table.item(row, column)

        if item is None:
            return "0"

        text = item.text().strip()

        return text if text else "0"

    @staticmethod
    def _set_cell(
        table: QTableWidget,
        row: int,
        column: int,
        value: str,
    ):
        table.setItem(
            row,
            column,
            QTableWidgetItem(value),
        )

    @staticmethod
    def _read_finite(
        table: QTableWidget,
        row: int,
        column: int,
        name: str,
    ) -> float:
        item = table.item(row, column)

        text = (
            item.text().strip()
            if item is not None
            else ""
        )

        if not text:
            text = "0"

        try:
            value = float(text)
        except ValueError:
            raise ValueError(
                f"{name} must be a number."
            )

        if not np.isfinite(value):
            raise ValueError(
                f"{name} must be finite."
            )

        return value