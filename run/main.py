"""MidiGen — Entry point. Launches the PyQt6 application."""

import logging
import sys
import warnings
from pathlib import Path

# Suppress noisy third-party warnings before any imports trigger them
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*Coremltools.*")
warnings.filterwarnings("ignore", message=".*tflite-runtime.*")
warnings.filterwarnings("ignore", message=".*Tensorflow.*")
logging.getLogger("basic_pitch").setLevel(logging.ERROR)

# Ensure project root is on sys.path so core/gui packages resolve
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from PyQt6.QtWidgets import QApplication

from gui.main_window import MainWindow


def main() -> None:
    # Suppress root logger warnings from basic-pitch dependency checks
    logging.getLogger().setLevel(logging.ERROR)

    app = QApplication(sys.argv)
    app.setApplicationName("MidiGen")
    app.setOrganizationName("MidiGen")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
