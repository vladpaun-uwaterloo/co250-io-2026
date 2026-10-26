import sys

from PySide6.QtWidgets import QApplication

from frontend import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Simplex Solver")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()