from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QSizePolicy,
    QTableWidget,
)


class SectionCard(QWidget):
    def __init__(self, title: str, subtitle: str | None = None):
        super().__init__()

        self.setObjectName("sectionCard")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        self.content_layout = QVBoxLayout(self)
        self.content_layout.setContentsMargins(18, 16, 18, 18)
        self.content_layout.setSpacing(12)

        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        self.content_layout.addWidget(title_label)

        if subtitle is not None:
            subtitle_label = QLabel(subtitle)
            subtitle_label.setObjectName("sectionSubtitle")
            subtitle_label.setWordWrap(True)
            self.content_layout.addWidget(subtitle_label)


def fit_table_height(table: QTableWidget, visible_rows: int | None = None, row_height: int = 36):
    rows = table.rowCount()

    if visible_rows is not None:
        rows = min(rows, visible_rows)

    height = table.horizontalHeader().height() + rows * row_height + table.frameWidth() * 2 + 2
    table.setFixedHeight(height)

    table.setVerticalScrollBarPolicy(
        Qt.ScrollBarPolicy.ScrollBarAsNeeded
        if visible_rows is not None and table.rowCount() > visible_rows
        else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    )