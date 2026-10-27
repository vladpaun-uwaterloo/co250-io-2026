import numpy as np

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QLineEdit,
    QComboBox,
    QPushButton,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QScrollArea,
)

from core import LinearProgram, ObjectiveSense, ConstraintSense

from .components import SectionCard, fit_table_height


SENSE_FROM_SYMBOL = {
    "≤": ConstraintSense.LE,
    "=": ConstraintSense.EQ,
    "≥": ConstraintSense.GE,
}


class GeneralLPInputWidget(QWidget):
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

        self.objective_sense = QComboBox()
        self.objective_sense.addItems(["MAX", "MIN"])
        self.objective_sense.setFixedWidth(100)

        self.objective_table = QTableWidget()
        self.constraint_table = QTableWidget()
        self.bounds_table = QTableWidget()

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
        for table in (
            self.objective_table,
            self.constraint_table,
            self.bounds_table,
        ):
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
            "Configure the number of variables and constraints.",
        )

        size_row = QHBoxLayout()

        size_row.addWidget(QLabel("Variables"))
        size_row.addWidget(self.variable_count)

        size_row.addSpacing(16)

        size_row.addWidget(QLabel("Constraints"))
        size_row.addWidget(self.constraint_count)

        size_row.addStretch()

        size_card.content_layout.addLayout(size_row)
        content_layout.addWidget(size_card)

        objective_card = SectionCard("Objective")

        objective_options = QHBoxLayout()

        objective_options.addWidget(QLabel("Sense"))
        objective_options.addWidget(self.objective_sense)

        objective_options.addSpacing(16)

        objective_options.addWidget(QLabel("Offset"))
        objective_options.addWidget(self.offset_input)

        objective_options.addStretch()

        objective_card.content_layout.addLayout(objective_options)
        objective_card.content_layout.addWidget(self.objective_table)

        content_layout.addWidget(objective_card)

        constraints_card = SectionCard(
            "Constraints",
            "Choose ≤, =, or ≥ independently for each row.",
        )

        constraints_card.content_layout.addWidget(self.constraint_table)
        content_layout.addWidget(constraints_card)

        bounds_card = SectionCard(
            "Variable bounds",
            'Use finite numbers, "inf", or "-inf".',
        )

        bounds_card.content_layout.addWidget(self.bounds_table)
        content_layout.addWidget(bounds_card)

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
            self._cell_text(self.objective_table, 0, j, "0")
            for j in range(old_n)
        ]

        old_A = [
            [
                self._cell_text(self.constraint_table, i, j, "0")
                for j in range(old_n)
            ]
            for i in range(old_m)
        ]

        old_senses = []

        for i in range(old_m):
            widget = self.constraint_table.cellWidget(i, old_n)

            if isinstance(widget, QComboBox):
                old_senses.append(widget.currentText())
            else:
                old_senses.append("≤")

        old_b = []

        if self.constraint_table.columnCount() >= old_n + 2:
            old_b = [
                self._cell_text(
                    self.constraint_table,
                    i,
                    old_n + 1,
                    "0",
                )
                for i in range(old_m)
            ]

        old_bounds = [
            (
                self._cell_text(self.bounds_table, j, 0, "0"),
                self._cell_text(self.bounds_table, j, 1, "inf"),
            )
            for j in range(self.bounds_table.rowCount())
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
        self.constraint_table.setColumnCount(n + 2)
        self.constraint_table.setHorizontalHeaderLabels(
            [f"x{j + 1}" for j in range(n)] + ["Sense", "RHS"]
        )
        self.constraint_table.setVerticalHeaderLabels([str(i + 1) for i in range(m)])

        for i in range(m):
            for j in range(n):
                value = old_A[i][j] if i < old_m and j < old_n else "0"
                self.constraint_table.setItem(
                    i,
                    j,
                    QTableWidgetItem(value),
                )

            sense_box = QComboBox()
            sense_box.addItems(["≤", "=", "≥"])
            sense_box.setFixedHeight(35)

            if i < len(old_senses):
                sense_box.setCurrentText(old_senses[i])

            self.constraint_table.setCellWidget(i, n, sense_box)

            rhs = old_b[i] if i < len(old_b) else "0"

            self.constraint_table.setItem(
                i,
                n + 1,
                QTableWidgetItem(rhs),
            )

        self.constraint_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        fit_table_height(self.constraint_table, visible_rows=7)

        self.bounds_table.setRowCount(n)
        self.bounds_table.setColumnCount(2)
        self.bounds_table.setHorizontalHeaderLabels(["Lower", "Upper"])
        self.bounds_table.setVerticalHeaderLabels([f"x{j + 1}" for j in range(n)])

        for j in range(n):
            if j < len(old_bounds):
                lower, upper = old_bounds[j]
            else:
                lower, upper = "0", "inf"

            self.bounds_table.setItem(
                j,
                0,
                QTableWidgetItem(lower),
            )

            self.bounds_table.setItem(
                j,
                1,
                QTableWidgetItem(upper),
            )

        self.bounds_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        fit_table_height(self.bounds_table, visible_rows=5)

    def get_lp(self) -> LinearProgram:
        n = self.variable_count.value()
        m = self.constraint_count.value()

        c = np.empty(n)
        A = np.empty((m, n))
        b = np.empty(m)

        for j in range(n):
            c[j] = self._read_finite(
                self.objective_table,
                0,
                j,
                f"c[{j + 1}]",
            )

        constraint_senses = []

        for i in range(m):
            for j in range(n):
                A[i, j] = self._read_finite(
                    self.constraint_table,
                    i,
                    j,
                    f"A[{i + 1}, {j + 1}]",
                )

            widget = self.constraint_table.cellWidget(i, n)

            if not isinstance(widget, QComboBox):
                raise RuntimeError(f"Missing constraint sense in row {i + 1}.")

            constraint_senses.append(
                SENSE_FROM_SYMBOL[widget.currentText()]
            )

            b[i] = self._read_finite(
                self.constraint_table,
                i,
                n + 1,
                f"b[{i + 1}]",
            )

        lower_bounds = np.empty(n)
        upper_bounds = np.empty(n)

        for j in range(n):
            lower_bounds[j] = self._read_bound(
                self.bounds_table,
                j,
                0,
                "0",
            )

            upper_bounds[j] = self._read_bound(
                self.bounds_table,
                j,
                1,
                "inf",
            )

            if lower_bounds[j] > upper_bounds[j]:
                raise ValueError(
                    f"Lower bound of x{j + 1} exceeds upper bound."
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

        return LinearProgram(
            c=c,
            A=A,
            b=b,
            objective_sense=ObjectiveSense[self.objective_sense.currentText()],
            constraint_senses=constraint_senses,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
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
        default: str,
    ) -> str:
        item = table.item(row, column)

        if item is None:
            return default

        text = item.text().strip()
        return text if text else default

    @classmethod
    def _read_finite(
        cls,
        table: QTableWidget,
        row: int,
        column: int,
        name: str,
    ) -> float:
        text = cls._cell_text(
            table,
            row,
            column,
            "0",
        )

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

    @classmethod
    def _read_bound(
        cls,
        table: QTableWidget,
        row: int,
        column: int,
        default: str,
    ) -> float:
        text = cls._cell_text(
            table,
            row,
            column,
            default,
        )

        text = text.lower().replace("∞", "inf")

        if text in {"inf", "+inf"}:
            return np.inf

        if text == "-inf":
            return -np.inf

        try:
            return float(text)
        except ValueError:
            raise ValueError(
                f'Invalid bound "{text}". Use a number, "inf", or "-inf".'
            )