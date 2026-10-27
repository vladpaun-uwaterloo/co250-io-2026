APP_STYLE = """
QWidget {
    background-color: #181a1f;
    color: #e6e8eb;
    font-size: 14px;
}

QMainWindow {
    background-color: #181a1f;
}

QLabel {
    background-color: transparent;
    border: none;
}


/* Tabs */

QTabWidget::pane {
    border: 1px solid #343842;
    border-radius: 8px;
    background-color: #1d2026;
    top: -1px;
}

QTabBar::tab {
    background-color: #20232a;
    color: #aeb4bd;
    border: 1px solid #343842;
    padding: 9px 18px;
    min-width: 120px;
}

QTabBar::tab:selected {
    background-color: #292d35;
    color: #ffffff;
    border-bottom: 2px solid #4f9cf9;
}

QTabBar::tab:hover {
    background-color: #262a31;
}


/* Group boxes */

QGroupBox {
    background-color: #1d2026;
    border: 1px solid #343842;
    border-radius: 8px;
    margin-top: 12px;
    padding: 14px 10px 10px 10px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #dfe3e8;
    background-color: transparent;
}


/* Inputs */

QLineEdit,
QSpinBox,
QComboBox {
    background-color: #1b1e24;
    color: #f1f3f5;
    border: 1px solid #3a3f49;
    border-radius: 5px;
    padding: 6px 8px;
    min-height: 24px;
}

QLineEdit:focus,
QSpinBox:focus,
QComboBox:focus {
    border: 1px solid #4f9cf9;
}

QSpinBox::up-button,
QSpinBox::down-button {
    width: 18px;
}


/* Tables */

QTableWidget {
    background-color: #191c21;
    alternate-background-color: #1c2026;
    color: #edf0f3;
    border: 1px solid #343842;
    border-radius: 6px;
    gridline-color: #30343c;
    selection-background-color: #294d73;
    selection-color: #ffffff;
}

QTableWidget::item {
    padding: 6px;
}

QTableWidget::item:selected {
    background-color: #294d73;
    color: #ffffff;
}

QHeaderView::section {
    background-color: #292d35;
    color: #cbd0d7;
    border: none;
    border-right: 1px solid #3a3f49;
    border-bottom: 1px solid #3a3f49;
    padding: 7px;
    font-weight: 600;
}


/* Inline table editors */

QTableWidget QLineEdit {
    background-color: #1b1e24;
    color: #f1f3f5;
    border: 1px solid #4f9cf9;
    border-radius: 0px;
    padding: 0px 6px;
    min-height: 0px;
    selection-background-color: #3685d6;
    selection-color: #ffffff;
}

QTableWidget QComboBox {
    border-radius: 0px;
    padding: 0px 6px;
    min-height: 0px;
}


/* Buttons */

QPushButton {
    background-color: #3685d6;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 9px 20px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #4796e6;
}

QPushButton:pressed {
    background-color: #2e74bd;
}

QPushButton:disabled {
    background-color: #353941;
    color: #777d86;
}


/* Labels */

QLabel#secondaryLabel {
    color: #969da7;
}

QLabel#statusLabel {
    color: #ffffff;
    font-size: 20px;
    font-weight: 700;
}

QLabel#comparisonLabel {
    background-color: #20242b;
    border: 1px solid #343842;
    border-radius: 6px;
    padding: 10px 12px;
    color: #d5d9df;
    font-weight: 500;
}


/* Splitter */

QSplitter::handle {
    background-color: #343842;
}

QSplitter::handle:horizontal {
    width: 1px;
}

QSplitter::handle:vertical {
    height: 1px;
}


/* Scroll areas */

QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollArea > QWidget > QWidget {
    background-color: transparent;
}


/* Scrollbars */

QScrollBar:vertical {
    background: #181a1f;
    width: 10px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #444a55;
    border-radius: 5px;
    min-height: 25px;
}

QScrollBar::handle:vertical:hover {
    background: #555c68;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background: #181a1f;
    height: 10px;
    margin: 0;
}

QScrollBar::handle:horizontal {
    background: #444a55;
    border-radius: 5px;
    min-width: 25px;
}

QScrollBar::handle:horizontal:hover {
    background: #555c68;
}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0px;
}
"""