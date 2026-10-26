import sys

from PySide6.QtWidgets import QApplication

from frontend import MainWindow
from frontend.style import APP_STYLE


def main():
    app = QApplication(sys.argv)

    app.setApplicationName("Simplex Solver")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()