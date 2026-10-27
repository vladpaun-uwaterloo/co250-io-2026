APP_STYLE = """
QWidget {
    background-color: #15171c;
    color: #e5e7eb;
    font-size: 14px;
}

QMainWindow {
    background-color: #15171c;
}

QLabel {
    background-color: transparent;
}


/* Main header */

QLabel#appTitle {
    font-size: 24px;
    font-weight: 700;
    color: #f8fafc;
}

QLabel#appSubtitle {
    color: #8f96a3;
    font-size: 13px;
    margin-bottom: 4px;
}


/* Page headings */

QLabel#pageTitle {
    font-size: 20px;
    font-weight: 700;
    color: #f8fafc;
}

QLabel#pageSubtitle {
    color: #8f96a3;
    margin-bottom: 4px;
}


/* Tabs */

QTabWidget#problemTabs::pane {
    border: 1px solid #2c313a;
    border-radius: 10px;
    background-color: #181b21;
}

QTabBar::tab {
    background-color: transparent;
    color: #8f96a3;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 11px 18px;
    min-width: 130px;
}

QTabBar::tab:hover {
    color: #d4d7dd;
}

QTabBar::tab:selected {
    color: #f4f6f8;
    border-bottom: 2px solid #4f9cf9;
}


/* Cards */

QWidget#sectionCard {
    background-color: #1c1f26;
    border: 1px solid #2c313a;
    border-radius: 10px;
}

QLabel#sectionTitle {
    color: #f1f3f5;
    font-size: 15px;
    font-weight: 650;
}

QLabel#sectionSubtitle {
    color: #8f96a3;
    font-size: 12px;
}


/* Inputs */

QLineEdit,
QSpinBox,
QComboBox {
    background-color: #171a20;
    color: #f1f3f5;
    border: 1px solid #343a45;
    border-radius: 6px;
    padding: 6px 9px;
    min-height: 25px;
}

QLineEdit:hover,
QSpinBox:hover,
QComboBox:hover {
    border-color: #454c59;
}

QLineEdit:focus,
QSpinBox:focus,
QComboBox:focus {
    border-color: #4f9cf9;
}


/* Tables */

QTableWidget {
    background-color: #171a20;
    alternate-background-color: #191d23;
    color: #e7e9ed;
    border: 1px solid #303640;
    border-radius: 7px;
    gridline-color: #292e36;
    selection-background-color: #284d75;
    selection-color: white;
}

QTableWidget::item {
    padding: 6px;
}

QHeaderView::section {
    background-color: #242831;
    color: #b8bec8;
    border: none;
    border-right: 1px solid #343a45;
    border-bottom: 1px solid #343a45;
    padding: 7px;
    font-weight: 600;
}

QTableCornerButton::section {
    background-color: #242831;
    border: none;
    border-right: 1px solid #343a45;
    border-bottom: 1px solid #343a45;
}


/* Editors embedded inside tables */

QTableWidget QLineEdit {
    background-color: #16191f;
    border: 1px solid #4f9cf9;
    border-radius: 0px;
    padding: 0px 6px;
    margin: 0px;
    min-height: 0px;
}

QTableWidget QComboBox {
    background-color: #171a20;
    border: none;
    border-radius: 0px;
    padding: 0px 6px;
    margin: 0px;
    min-height: 0px;
}

QTableWidget QComboBox:hover {
    background-color: #20242b;
}

QTableWidget QComboBox:focus {
    border: 1px solid #4f9cf9;
}


/* Buttons */

QPushButton {
    background-color: #2f7fd2;
    color: white;
    border: none;
    border-radius: 7px;
    padding: 9px 18px;
    font-weight: 650;
}

QPushButton:hover {
    background-color: #378ce5;
}

QPushButton:pressed {
    background-color: #286eb7;
}

QPushButton#primaryButton {
    font-size: 14px;
}


/* Result area */

QWidget#resultCard {
    background-color: #1c1f26;
    border: 1px solid #2c313a;
    border-radius: 10px;
}

QLabel#solverName {
    color: #f8fafc;
    font-size: 17px;
    font-weight: 700;
}

QLabel#timeBadge {
    background-color: #242831;
    color: #aeb5c0;
    border-radius: 6px;
    padding: 5px 9px;
    font-size: 12px;
    font-weight: 600;
}

QLabel#resultStatus {
    font-size: 21px;
    font-weight: 750;
    margin-top: 2px;
}

QLabel#resultStatus[state="success"] {
    color: #79d69a;
}

QLabel#resultStatus[state="warning"] {
    color: #e4bd70;
}

QLabel#resultStatus[state="error"] {
    color: #e17d82;
}

QLabel#objectiveValue {
    color: #f8fafc;
    font-size: 28px;
    font-weight: 750;
}

QLabel#fieldLabel {
    color: #8f96a3;
    font-size: 12px;
    font-weight: 600;
    margin-top: 4px;
}

QLabel#valueText {
    color: #e3e6ea;
}

QLabel#errorText {
    color: #c6cad1;
}


/* Summary */

QFrame#summaryCard {
    background-color: #1d222a;
    border: 1px solid #303640;
    border-radius: 8px;
}

QLabel#summaryText {
    background-color: transparent;
    color: #c6cbd3;
}


/* Splitters */

QSplitter#mainSplitter::handle,
QSplitter#resultSplitter::handle {
    background-color: transparent;
}

QSplitter#mainSplitter::handle:horizontal {
    width: 12px;
}

QSplitter#resultSplitter::handle:horizontal {
    width: 12px;
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
    background: transparent;
    width: 9px;
    margin: 2px;
}

QScrollBar::handle:vertical {
    background: #3a404b;
    border-radius: 4px;
    min-height: 28px;
}

QScrollBar::handle:vertical:hover {
    background: #4a515d;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background: transparent;
    height: 9px;
}

QScrollBar::handle:horizontal {
    background: #3a404b;
    border-radius: 4px;
    min-width: 28px;
}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0;
}
"""