import numpy as np

from PySide6.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
)

from core.models import (
    OptimalResult,
    UnboundedResult,
    InfeasibleResult,
)


class ResultView(QGroupBox):
    def __init__(self):
        super().__init__("Result")

        self.result_layout = QVBoxLayout(self)

        self.result_layout.setContentsMargins(
            16,
            20,
            16,
            16,
        )

        self.result_layout.setSpacing(12)

        self._show_placeholder()

    def clear(self):
        while self.result_layout.count() > 0:
            item = self.result_layout.takeAt(0)

            if item is None:
                continue

            widget = item.widget()

            if widget is not None:
                widget.deleteLater()

    def show_result(self, result):
        self.clear()

        if isinstance(
            result,
            OptimalResult,
        ):
            self._show_optimal(result)

        elif isinstance(
            result,
            UnboundedResult,
        ):
            self._show_unbounded(result)

        elif isinstance(
            result,
            InfeasibleResult,
        ):
            self._show_infeasible(result)

        else:
            self.result_layout.addWidget(
                QLabel(
                    f"Unknown result type: "
                    f"{type(result).__name__}"
                )
            )

        self.result_layout.addStretch()

    def _show_optimal(
        self,
        result: OptimalResult,
    ):
        self._add_status("OPTIMAL")

        value_title = QLabel(
            "Objective value"
        )

        value_title.setObjectName(
            "secondaryLabel"
        )

        self.result_layout.addWidget(
            value_title
        )

        value = QLabel(
            self._format_number(
                result.objective_value
            )
        )

        font = value.font()
        font.setPointSize(
            font.pointSize() + 5
        )
        font.setBold(True)
        value.setFont(font)

        self.result_layout.addWidget(
            value
        )

        self._add_vector(
            "Solution",
            result.solution,
            "x",
        )

        title = QLabel("Basis")
        title.setObjectName(
            "secondaryLabel"
        )

        self.result_layout.addWidget(
            title
        )

        basis_names = ", ".join(
            f"x{i + 1}"
            for i in result.basis.indices
        )

        self.result_layout.addWidget(
            QLabel(basis_names)
        )

        self._add_vector(
            "Optimality certificate",
            result.certificate,
            "y",
        )

    def _show_unbounded(
        self,
        result: UnboundedResult,
    ):
        self._add_status(
            "UNBOUNDED"
        )

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
        self._add_status(
            "INFEASIBLE"
        )

        self._add_vector(
            "Infeasibility certificate",
            result.certificate,
            "y",
        )

    def _add_status(
        self,
        text: str,
    ):
        label = QLabel(text)

        label.setObjectName(
            "statusLabel"
        )

        self.result_layout.addWidget(
            label
        )

    def _add_vector(
        self,
        title: str,
        values,
        prefix: str,
    ):
        values = np.asarray(
            values
        ).reshape(-1)

        label = QLabel(title)

        label.setObjectName(
            "secondaryLabel"
        )

        self.result_layout.addWidget(
            label
        )

        table = QTableWidget(
            1,
            len(values),
        )

        table.verticalHeader().setVisible(
            False
        )

        table.setHorizontalHeaderLabels(
            [
                f"{prefix}{i + 1}"
                for i in range(len(values))
            ]
        )

        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        table.horizontalHeader().setFixedHeight(
            30
        )

        table.setRowHeight(
            0,
            34,
        )

        for i, value in enumerate(
            values
        ):
            item = QTableWidgetItem(
                self._format_number(
                    value
                )
            )

            table.setItem(
                0,
                i,
                item,
            )

        table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )

        table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )

        table.setFixedHeight(72)

        self.result_layout.addWidget(
            table
        )

    def _show_placeholder(self):
        title = QLabel(
            "No result yet"
        )

        title.setObjectName(
            "statusLabel"
        )

        self.result_layout.addWidget(
            title
        )

        description = QLabel(
            "Enter a standard equality form "
            "problem and press Solve."
        )

        description.setObjectName(
            "secondaryLabel"
        )

        description.setWordWrap(True)

        self.result_layout.addWidget(
            description
        )

        self.result_layout.addStretch()

    @staticmethod
    def _format_number(
        value,
    ) -> str:
        value = float(value)

        if abs(value) < 1e-12:
            value = 0.0

        return f"{value:.10g}"