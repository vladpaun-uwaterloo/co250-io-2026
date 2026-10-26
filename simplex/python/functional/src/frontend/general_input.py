import numpy as np

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QLineEdit,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QGroupBox,
    QSizePolicy,
)

from core.models import (
    LinearProgram,
    ObjectiveSense,
    ConstraintSense,
)


class GeneralLPInputWidget(QWidget):
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
        self.offset_input.setFixedWidth(140)

        self._configure_tables()
        self._build_ui()

        self.variable_count.valueChanged.connect(
            self._resize_tables
        )

        self.constraint_count.valueChanged.connect(
            self._resize_tables
        )

        self._resize_tables()

    def _configure_tables(self):
        self.objective_table.verticalHeader().setVisible(
            False
        )

        self.objective_table.setFixedHeight(76)

        self.objective_table.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        self.objective_table.horizontalHeader().setFixedHeight(
            32
        )

        self.constraint_table.horizontalHeader().setFixedHeight(
            32
        )

        self.constraint_table.verticalHeader().setDefaultSectionSize(
            34
        )

        self.constraint_table.setAlternatingRowColors(
            True
        )

        self.bounds_table.horizontalHeader().setFixedHeight(
            32
        )

        self.bounds_table.verticalHeader().setDefaultSectionSize(
            34
        )

        self.bounds_table.setAlternatingRowColors(
            True
        )

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        main_layout.setContentsMargins(
            18,
            18,
            18,
            18,
        )

        main_layout.setSpacing(14)

        # Problem size
        size_group = QGroupBox(
            "Problem Size"
        )

        size_layout = QHBoxLayout(
            size_group
        )

        size_layout.addWidget(
            QLabel("Variables")
        )

        size_layout.addWidget(
            self.variable_count
        )

        size_layout.addSpacing(20)

        size_layout.addWidget(
            QLabel("Constraints")
        )

        size_layout.addWidget(
            self.constraint_count
        )

        size_layout.addStretch()

        main_layout.addWidget(
            size_group
        )

        # Objective
        objective_group = QGroupBox(
            "Objective"
        )

        objective_layout = QVBoxLayout(
            objective_group
        )

        objective_layout.setSpacing(8)

        top_row = QHBoxLayout()

        top_row.addWidget(
            QLabel("Sense")
        )

        top_row.addWidget(
            self.objective_sense
        )

        top_row.addSpacing(20)

        top_row.addWidget(
            QLabel("Offset")
        )

        top_row.addWidget(
            self.offset_input
        )

        top_row.addStretch()

        objective_layout.addLayout(
            top_row
        )

        objective_layout.addWidget(
            self.objective_table
        )

        main_layout.addWidget(
            objective_group
        )

        # Constraints
        constraints_group = QGroupBox(
            "Constraints"
        )

        constraints_layout = QVBoxLayout(
            constraints_group
        )

        constraints_layout.setSpacing(8)

        description = QLabel(
            "Each row represents one linear constraint."
        )

        description.setObjectName(
            "secondaryLabel"
        )

        constraints_layout.addWidget(
            description
        )

        constraints_layout.addWidget(
            self.constraint_table
        )

        main_layout.addWidget(
            constraints_group
        )

        # Bounds
        bounds_group = QGroupBox(
            "Variable Bounds"
        )

        bounds_layout = QVBoxLayout(
            bounds_group
        )

        bounds_layout.setSpacing(8)

        description = QLabel(
            'Use numbers, "inf", or "-inf".'
        )

        description.setObjectName(
            "secondaryLabel"
        )

        bounds_layout.addWidget(
            description
        )

        bounds_layout.addWidget(
            self.bounds_table
        )

        main_layout.addWidget(
            bounds_group
        )

        info = QLabel(
            "General LP solving is not connected yet."
        )

        info.setObjectName(
            "secondaryLabel"
        )

        main_layout.addWidget(info)

        main_layout.addStretch()

    def _resize_tables(self):
        old_n = self.objective_table.columnCount()
        old_m = self.constraint_table.rowCount()

        old_objective = [
            self._cell_text(
                self.objective_table,
                0,
                j,
                "0",
            )
            for j in range(old_n)
        ]

        old_A = [
            [
                self._cell_text(
                    self.constraint_table,
                    i,
                    j,
                    "0",
                )
                for j in range(old_n)
            ]
            for i in range(old_m)
        ]

        old_senses = []

        for i in range(old_m):
            widget = (
                self.constraint_table.cellWidget(
                    i,
                    old_n,
                )
            )

            if isinstance(
                widget,
                QComboBox,
            ):
                old_senses.append(
                    widget.currentText()
                )
            else:
                old_senses.append(
                    "LE"
                )

        old_b = []

        if (
            self.constraint_table.columnCount()
            >= old_n + 2
        ):
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
                self._cell_text(
                    self.bounds_table,
                    j,
                    0,
                    "0",
                ),
                self._cell_text(
                    self.bounds_table,
                    j,
                    1,
                    "inf",
                ),
            )
            for j in range(
                self.bounds_table.rowCount()
            )
        ]

        n = self.variable_count.value()
        m = self.constraint_count.value()

        # Objective
        self.objective_table.setRowCount(
            1
        )

        self.objective_table.setColumnCount(
            n
        )

        self.objective_table.setHorizontalHeaderLabels(
            [
                f"x{j + 1}"
                for j in range(n)
            ]
        )

        for j in range(n):
            value = (
                old_objective[j]
                if j < len(old_objective)
                else "0"
            )

            self.objective_table.setItem(
                0,
                j,
                QTableWidgetItem(
                    value
                ),
            )

        self.objective_table.setRowHeight(
            0,
            34,
        )

        self.objective_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        # Constraints
        self.constraint_table.setRowCount(
            m
        )

        self.constraint_table.setColumnCount(
            n + 2
        )

        headers = (
            [
                f"x{j + 1}"
                for j in range(n)
            ]
            + ["Sense", "RHS"]
        )

        self.constraint_table.setHorizontalHeaderLabels(
            headers
        )

        self.constraint_table.setVerticalHeaderLabels(
            [
                str(i + 1)
                for i in range(m)
            ]
        )

        for i in range(m):
            for j in range(n):
                value = "0"

                if (
                    i < old_m
                    and j < old_n
                ):
                    value = old_A[i][j]

                self.constraint_table.setItem(
                    i,
                    j,
                    QTableWidgetItem(
                        value
                    ),
                )

            sense_box = QComboBox()

            sense_box.addItems(
                ["LE", "EQ", "GE"]
            )

            if i < len(old_senses):
                sense_box.setCurrentText(
                    old_senses[i]
                )

            self.constraint_table.setCellWidget(
                i,
                n,
                sense_box,
            )

            rhs = (
                old_b[i]
                if i < len(old_b)
                else "0"
            )

            self.constraint_table.setItem(
                i,
                n + 1,
                QTableWidgetItem(
                    rhs
                ),
            )

        self.constraint_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        visible_rows = min(
            m,
            7,
        )

        self.constraint_table.setFixedHeight(
            32
            + visible_rows * 34
            + 8
        )

        # Bounds
        self.bounds_table.setRowCount(
            n
        )

        self.bounds_table.setColumnCount(
            2
        )

        self.bounds_table.setHorizontalHeaderLabels(
            ["Lower", "Upper"]
        )

        self.bounds_table.setVerticalHeaderLabels(
            [
                f"x{j + 1}"
                for j in range(n)
            ]
        )

        for j in range(n):
            if j < len(old_bounds):
                lower, upper = (
                    old_bounds[j]
                )
            else:
                lower = "0"
                upper = "inf"

            self.bounds_table.setItem(
                j,
                0,
                QTableWidgetItem(
                    lower
                ),
            )

            self.bounds_table.setItem(
                j,
                1,
                QTableWidgetItem(
                    upper
                ),
            )

        self.bounds_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        visible_bounds = min(
            n,
            5,
        )

        self.bounds_table.setFixedHeight(
            32
            + visible_bounds * 34
            + 8
        )

    def get_lp(self) -> LinearProgram:
        n = self.variable_count.value()
        m = self.constraint_count.value()

        c = np.empty(
            n,
            dtype=float,
        )

        for j in range(n):
            c[j] = self._read_finite(
                self.objective_table,
                0,
                j,
                f"c[{j + 1}]",
            )

        A = np.empty(
            (m, n),
            dtype=float,
        )

        b = np.empty(
            m,
            dtype=float,
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

            sense_widget = (
                self.constraint_table.cellWidget(
                    i,
                    n,
                )
            )

            if not isinstance(
                sense_widget,
                QComboBox,
            ):
                raise RuntimeError(
                    "Missing constraint sense "
                    f"selector in row {i + 1}."
                )

            constraint_senses.append(
                ConstraintSense[
                    sense_widget.currentText()
                ]
            )

            b[i] = self._read_finite(
                self.constraint_table,
                i,
                n + 1,
                f"b[{i + 1}]",
            )

        lower_bounds = np.empty(
            n,
            dtype=float,
        )

        upper_bounds = np.empty(
            n,
            dtype=float,
        )

        for j in range(n):
            lower_bounds[j] = (
                self._read_bound(
                    self.bounds_table,
                    j,
                    0,
                    f"lower bound of x{j + 1}",
                )
            )

            upper_bounds[j] = (
                self._read_bound(
                    self.bounds_table,
                    j,
                    1,
                    f"upper bound of x{j + 1}",
                )
            )

            if (
                lower_bounds[j]
                > upper_bounds[j]
            ):
                raise ValueError(
                    f"Lower bound of x{j + 1} "
                    "is greater than its upper bound."
                )

        try:
            objective_offset = float(
                self.offset_input.text().strip()
            )

        except ValueError:
            raise ValueError(
                "Objective offset must be a number."
            )

        if not np.isfinite(
            objective_offset
        ):
            raise ValueError(
                "Objective offset must be finite."
            )

        objective_sense = ObjectiveSense[
            self.objective_sense.currentText()
        ]

        return LinearProgram(
            c=c,
            A=A,
            b=b,
            objective_sense=objective_sense,
            constraint_senses=constraint_senses,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            objective_offset=objective_offset,
        )

    @staticmethod
    def _cell_text(
        table: QTableWidget,
        row: int,
        column: int,
        default: str,
    ) -> str:

        item = table.item(
            row,
            column,
        )

        if item is None:
            return default

        value = item.text().strip()

        return (
            value
            if value
            else default
        )

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
        name: str,
    ) -> float:

        text = cls._cell_text(
            table,
            row,
            column,
            "0",
        )

        normalized = (
            text
            .lower()
            .replace(
                "∞",
                "inf",
            )
        )

        if normalized in {
            "inf",
            "+inf",
        }:
            return np.inf

        if normalized == "-inf":
            return -np.inf

        try:
            return float(
                normalized
            )

        except ValueError:
            raise ValueError(
                f"{name} must be a number, "
                "inf, or -inf."
            )