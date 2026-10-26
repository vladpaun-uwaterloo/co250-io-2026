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

        self.result_layout = QVBoxLayout(
            self
        )

        self._show_placeholder()

    def clear(self):
        while (
            self.result_layout.count()
            > 0
        ):
            item = self.result_layout.takeAt(
                0
            )

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
                    "Unknown result type: "
                    f"{type(result).__name__}"
                )
            )

        self.result_layout.addStretch()

    def _show_optimal(
        self,
        result: OptimalResult,
    ):
        self._add_status(
            "OPTIMAL"
        )

        self.result_layout.addWidget(
            QLabel(
                "Objective value: "
                f"{self._format_number(result.objective_value)}"
            )
        )

        self._add_vector(
            "Solution",
            result.solution,
            "x",
        )

        basis_names = ", ".join(
            f"x{i + 1}"
            for i in result.basis.indices
        )

        self.result_layout.addWidget(
            QLabel(
                f"Basis: {basis_names}"
            )
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

        font = label.font()
        font.setPointSize(
            font.pointSize() + 3
        )
        font.setBold(True)

        label.setFont(font)

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

        self.result_layout.addWidget(
            QLabel(title)
        )

        table = QTableWidget(
            1,
            len(values),
        )

        table.setHorizontalHeaderLabels(
            [
                f"{prefix}{i + 1}"
                for i in range(
                    len(values)
                )
            ]
        )

        table.verticalHeader().setVisible(
            False
        )

        for i, value in enumerate(
            values
        ):
            table.setItem(
                0,
                i,
                QTableWidgetItem(
                    self._format_number(
                        value
                    )
                ),
            )

        table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )

        table.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
        )

        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        table.setMaximumHeight(90)

        self.result_layout.addWidget(
            table
        )

    def _show_placeholder(self):
        self.result_layout.addWidget(
            QLabel(
                "Enter an SEF problem "
                "and press Solve."
            )
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