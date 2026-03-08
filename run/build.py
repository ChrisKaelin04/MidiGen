"""Optional PyInstaller build script for standalone distribution."""

import subprocess
import sys
from pathlib import Path

# Resolve paths relative to project root
_project_root = Path(__file__).resolve().parent.parent


def build() -> None:
    sep = ';' if sys.platform == 'win32' else ':'
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--name', 'MidiGen',
        '--onefile',
        '--windowed',
        '--add-data', f'{_project_root / "core"}{sep}core',
        '--add-data', f'{_project_root / "gui"}{sep}gui',
        str(_project_root / 'run' / 'main.py'),
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("Build complete. Check the dist/ directory.")


if __name__ == "__main__":
    build()
