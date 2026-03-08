"""Optional PyInstaller build script for standalone distribution."""

import subprocess
import sys


def build() -> None:
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--name', 'MidiGen',
        '--onefile',
        '--windowed',
        '--add-data', 'core:core',
        '--add-data', 'gui:gui',
        'main.py',
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("Build complete. Check the dist/ directory.")


if __name__ == "__main__":
    build()
