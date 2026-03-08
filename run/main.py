"""MidiGen — Entry point. Launches the PyQt6 application."""

import sys
from pathlib import Path

# Ensure project root is on sys.path so core/gui packages resolve
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("MidiGen")
    app.setOrganizationName("MidiGen")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
