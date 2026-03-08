"""Main application window — wires all panels, manages app state and pipeline."""

import json
import os
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QMimeData, QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QAction, QDrag
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from core import AudioData, NoteEvent, ProcessingConfig
from core.audio_loader import load_audio, detect_bpm, apply_hpss
from core.transcriber import transcribe
from core.midi_processor import process
from core.midi_exporter import build_midi, save, save_temp
from gui.controls_panel import ControlsPanel
from gui.waveform_view import WaveformView
from gui.midi_preview import MidiPreview

import pretty_midi

_CONFIG_DIR = Path.home() / '.midigen'
_CONFIG_FILE = _CONFIG_DIR / 'config.json'

_DEFAULT_CONFIG = {
    "bpm": 120,
    "confidence_threshold": 0.5,
    "note_low": 36,
    "note_high": 84,
    "filter_ghost_notes": True,
    "dynamic_velocity": False,
    "preserve_durations": True,
    "onset_threshold": 0.5,
    "frame_threshold": 0.3,
    "minimum_note_length_ms": 58,
    "ensemble_passes": 1,
    "use_hpss": False,
    "filter_harmonics": True,
    "theme": "dark",
    "last_directory": str(Path.home()),
}

DARK_THEME = """
QWidget {
    background-color: #1e1e1e;
    color: #f0f0f0;
    font-size: 13px;
}
QMainWindow {
    background-color: #1e1e1e;
}
QGroupBox {
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 12px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #4a9eff;
}
QPushButton {
    background-color: #2d2d2d;
    border: 1px solid #3a3a3a;
    border-radius: 4px;
    padding: 6px 16px;
    color: #f0f0f0;
}
QPushButton:hover {
    background-color: #3a3a3a;
    border-color: #4a9eff;
}
QPushButton:pressed {
    background-color: #4a9eff;
}
QPushButton:disabled {
    color: #666666;
    background-color: #1e1e1e;
    border-color: #2d2d2d;
}
QPushButton#generateBtn {
    background-color: #1a5a1a;
    border-color: #2a8a2a;
    font-weight: bold;
    font-size: 14px;
}
QPushButton#generateBtn:hover {
    background-color: #2a8a2a;
}
QPushButton#generateBtn:disabled {
    background-color: #1e1e1e;
    border-color: #2d2d2d;
    color: #666666;
}
QSpinBox, QComboBox, QLineEdit {
    background-color: #2d2d2d;
    border: 1px solid #3a3a3a;
    border-radius: 3px;
    padding: 3px 6px;
    color: #f0f0f0;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #3a3a3a;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #4a9eff;
    width: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
QCheckBox {
    spacing: 6px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #3a3a3a;
    border-radius: 3px;
    background-color: #2d2d2d;
}
QCheckBox::indicator:checked {
    background-color: #4a9eff;
    border-color: #4a9eff;
}
QMenuBar {
    background-color: #252525;
    border-bottom: 1px solid #3a3a3a;
}
QMenuBar::item:selected {
    background-color: #3a3a3a;
}
QMenu {
    background-color: #2d2d2d;
    border: 1px solid #3a3a3a;
}
QMenu::item:selected {
    background-color: #4a9eff;
}
QStatusBar {
    background-color: #252525;
    border-top: 1px solid #3a3a3a;
    color: #888888;
}
QSplitter::handle {
    background-color: #3a3a3a;
    height: 2px;
}
QLabel#dragLabel {
    border: 2px dashed #3a3a3a;
    border-radius: 6px;
    padding: 8px;
    color: #888888;
}
QLabel#dragLabel:hover {
    border-color: #4a9eff;
    color: #f0f0f0;
}
"""

LIGHT_THEME = """
QWidget {
    background-color: #f5f5f5;
    color: #1e1e1e;
    font-size: 13px;
}
QMainWindow {
    background-color: #f5f5f5;
}
QGroupBox {
    border: 1px solid #cccccc;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 12px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #0066cc;
}
QPushButton {
    background-color: #e8e8e8;
    border: 1px solid #cccccc;
    border-radius: 4px;
    padding: 6px 16px;
    color: #1e1e1e;
}
QPushButton:hover {
    background-color: #d0d0d0;
    border-color: #0066cc;
}
QPushButton:pressed {
    background-color: #0066cc;
    color: #ffffff;
}
QPushButton:disabled {
    color: #999999;
    background-color: #f0f0f0;
    border-color: #dddddd;
}
QPushButton#generateBtn {
    background-color: #d4edda;
    border-color: #28a745;
    font-weight: bold;
    font-size: 14px;
    color: #1e1e1e;
}
QPushButton#generateBtn:hover {
    background-color: #28a745;
    color: #ffffff;
}
QPushButton#generateBtn:disabled {
    background-color: #f0f0f0;
    border-color: #dddddd;
    color: #999999;
}
QSpinBox, QComboBox, QLineEdit {
    background-color: #ffffff;
    border: 1px solid #cccccc;
    border-radius: 3px;
    padding: 3px 6px;
    color: #1e1e1e;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #cccccc;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #0066cc;
    width: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #cccccc;
    border-radius: 3px;
    background-color: #ffffff;
}
QCheckBox::indicator:checked {
    background-color: #0066cc;
    border-color: #0066cc;
}
QMenuBar {
    background-color: #e8e8e8;
    border-bottom: 1px solid #cccccc;
}
QMenuBar::item:selected {
    background-color: #d0d0d0;
}
QMenu {
    background-color: #ffffff;
    border: 1px solid #cccccc;
}
QMenu::item:selected {
    background-color: #0066cc;
    color: #ffffff;
}
QStatusBar {
    background-color: #e8e8e8;
    border-top: 1px solid #cccccc;
    color: #666666;
}
QSplitter::handle {
    background-color: #cccccc;
    height: 2px;
}
QLabel#dragLabel {
    border: 2px dashed #cccccc;
    border-radius: 6px;
    padding: 8px;
    color: #999999;
}
QLabel#dragLabel:hover {
    border-color: #0066cc;
    color: #1e1e1e;
}
"""


class _PipelineWorker(QThread):
    """Run the transcription + processing pipeline in a background thread."""

    finished = pyqtSignal(list, object)  # notes, midi_obj
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, audio: AudioData, config: ProcessingConfig, parent=None) -> None:
        super().__init__(parent)
        self.audio = audio
        self.config = config

    def run(self) -> None:
        try:
            audio = self.audio
            if self.config.use_hpss:
                self.status.emit("Applying harmonic separation (HPSS)...")
                audio = apply_hpss(audio)

            passes = self.config.ensemble_passes
            if passes > 1:
                self.status.emit(f"Transcribing audio ({passes}-pass ensemble)...")
            else:
                self.status.emit("Transcribing audio with basic-pitch...")
            notes = transcribe(
                audio,
                confidence_threshold=self.config.confidence_threshold,
                onset_threshold=self.config.onset_threshold,
                frame_threshold=self.config.frame_threshold,
                minimum_note_length_ms=self.config.minimum_note_length_ms,
                ensemble_passes=passes,
            )

            self.status.emit(f"Processing {len(notes)} detected notes...")
            processed = process(
                notes,
                bpm=self.config.bpm,
                note_low=self.config.note_low,
                note_high=self.config.note_high,
                do_filter_ghosts=self.config.filter_ghost_notes,
                dynamic_velocity=self.config.dynamic_velocity,
                preserve_durations=self.config.preserve_durations,
                do_filter_harmonics=self.config.filter_harmonics,
            )

            self.status.emit("Building MIDI file...")
            midi_obj = build_midi(processed, self.config.bpm)
            save_temp(midi_obj)

            self.finished.emit(processed, midi_obj)
        except Exception as e:
            self.error.emit(str(e))


class DragLabel(QLabel):
    """A label that supports drag-and-drop of a MIDI file."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dragLabel")
        self.setText("Drag .mid file from here")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(140)
        self._midi_path: Path | None = None

    def set_midi_path(self, path: Path | None) -> None:
        self._midi_path = path
        if path:
            self.setText("Drag .mid file")
            self.setToolTip(str(path))
        else:
            self.setText("Drag .mid file from here")
            self.setToolTip("")

    def mousePressEvent(self, event) -> None:
        if self._midi_path and self._midi_path.exists() and event.button() == Qt.MouseButton.LeftButton:
            drag = QDrag(self)
            mime = QMimeData()
            mime.setUrls([QUrl.fromLocalFile(str(self._midi_path))])
            drag.setMimeData(mime)
            drag.exec(Qt.DropAction.CopyAction)


class MainWindow(QMainWindow):
    """Main application window for MidiGen."""

    def __init__(self) -> None:
        super().__init__()
        self._audio_data: AudioData | None = None
        self._full_audio_samples: tuple | None = None  # (samples, sr) before trimming
        self._processed_notes: list[NoteEvent] = []
        self._midi_obj: pretty_midi.PrettyMIDI | None = None
        self._pipeline_worker: _PipelineWorker | None = None
        self._config = self._load_config()

        self.setWindowTitle("MidiGen — Audio to MIDI")
        self.setMinimumSize(800, 700)
        self.resize(950, 800)

        self._setup_menu()
        self._setup_ui()
        self._setup_statusbar()
        self._apply_config()
        self._apply_theme()
        self._connect_signals()

    def _setup_menu(self) -> None:
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")
        open_action = QAction("&Open Audio File...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_file)
        file_menu.addAction(open_action)
        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # View menu
        view_menu = menubar.addMenu("&View")
        self.theme_action = QAction("Switch to Light Mode", self)
        self.theme_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(self.theme_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        about_action = QAction("&About MidiGen", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # File load row
        file_row = QHBoxLayout()
        self.load_btn = QPushButton("Load Audio File")
        self.file_label = QLabel("No file loaded")
        file_row.addWidget(self.load_btn)
        file_row.addWidget(self.file_label, 1)
        main_layout.addLayout(file_row)

        # Main content splitter (vertical)
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Waveform view
        self.waveform_view = WaveformView()
        splitter.addWidget(self.waveform_view)

        # Controls panel
        self.controls = ControlsPanel()
        self.controls.generate_btn.setObjectName("generateBtn")
        splitter.addWidget(self.controls)

        # MIDI preview
        self.midi_preview = MidiPreview()
        splitter.addWidget(self.midi_preview)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 3)

        main_layout.addWidget(splitter, 1)

        # Export row
        export_row = QHBoxLayout()

        self.save_btn = QPushButton("Save .mid File")
        self.save_btn.setEnabled(False)
        export_row.addWidget(self.save_btn)

        self.open_system_btn = QPushButton("Open with System Default")
        self.open_system_btn.setEnabled(False)
        export_row.addWidget(self.open_system_btn)

        export_row.addStretch()

        self.drag_label = DragLabel()
        export_row.addWidget(self.drag_label)

        main_layout.addLayout(export_row)

    def _setup_statusbar(self) -> None:
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _connect_signals(self) -> None:
        self.load_btn.clicked.connect(self._on_open_file)
        self.controls.generate_requested.connect(self._on_generate)
        self.controls.auto_detect_bpm_requested.connect(self._on_auto_detect_bpm)
        self.controls.settings_changed.connect(self._save_config)
        self.save_btn.clicked.connect(self._on_save_midi)
        self.open_system_btn.clicked.connect(self._on_open_system)

    # --- File loading ---

    def _on_open_file(self) -> None:
        last_dir = self._config.get("last_directory", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            last_dir,
            "Audio Files (*.mp3 *.wav *.flac);;All Files (*)",
        )
        if not path:
            return

        file_path = Path(path)
        self._config["last_directory"] = str(file_path.parent)
        self._save_config()

        try:
            self.status_bar.showMessage(f"Loading {file_path.name}...")
            # Load full audio for waveform display and BPM detection
            audio_full = load_audio(file_path)
            self._full_audio_samples = (audio_full.samples.copy(), audio_full.sample_rate)
            self._audio_data = audio_full

            self.file_label.setText(file_path.name)
            self.waveform_view.load_audio(audio_full.samples, audio_full.sample_rate)
            self.controls.set_generate_enabled(True)
            self.status_bar.showMessage(
                f"Loaded: {file_path.name} ({audio_full.duration_sec:.1f}s)"
            )

            # Reset output state
            self._processed_notes = []
            self._midi_obj = None
            self.midi_preview.clear()
            self._set_export_enabled(False)

        except Exception as e:
            QMessageBox.critical(self, "Error Loading File", str(e))
            self.status_bar.showMessage("Error loading file")

    # --- BPM detection ---

    def _on_auto_detect_bpm(self) -> None:
        if self._audio_data is None:
            self.status_bar.showMessage("Load an audio file first")
            return
        try:
            self.status_bar.showMessage("Detecting BPM...")
            bpm = detect_bpm(self._audio_data)
            self.controls.bpm = int(round(bpm))
            self.status_bar.showMessage(f"Detected BPM: {bpm:.1f}")
        except Exception as e:
            QMessageBox.warning(self, "BPM Detection Failed", str(e))
            self.status_bar.showMessage("BPM detection failed")

    # --- Pipeline execution ---

    def _on_generate(self) -> None:
        if self._full_audio_samples is None:
            return

        if self._pipeline_worker is not None and self._pipeline_worker.isRunning():
            self.status_bar.showMessage("Pipeline already running...")
            return

        # Reload audio trimmed to selection
        start = self.waveform_view.start_sec
        end = self.waveform_view.end_sec
        if self._audio_data is None:
            return

        try:
            audio = load_audio(self._audio_data.file_path, start_sec=start, end_sec=end)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load audio selection: {e}")
            return

        config = ProcessingConfig(
            bpm=self.controls.bpm,
            start_sec=start,
            end_sec=end,
            note_low=self.controls.note_low,
            note_high=self.controls.note_high,
            confidence_threshold=self.controls.confidence_threshold,
            filter_ghost_notes=self.controls.filter_ghosts,
            dynamic_velocity=self.controls.dynamic_velocity,
            preserve_durations=self.controls.preserve_durations,
            onset_threshold=self.controls.onset_threshold,
            frame_threshold=self.controls.frame_threshold,
            minimum_note_length_ms=self.controls.minimum_note_length_ms,
            ensemble_passes=self.controls.ensemble_passes,
            use_hpss=self.controls.use_hpss,
            filter_harmonics=self.controls.do_filter_harmonics,
        )

        self.controls.set_generate_enabled(False)
        self._set_export_enabled(False)

        self._pipeline_worker = _PipelineWorker(audio, config, parent=self)
        self._pipeline_worker.status.connect(self.status_bar.showMessage)
        self._pipeline_worker.finished.connect(self._on_pipeline_finished)
        self._pipeline_worker.error.connect(self._on_pipeline_error)
        self._pipeline_worker.start()

    def _on_pipeline_finished(self, notes: list[NoteEvent], midi_obj: pretty_midi.PrettyMIDI) -> None:
        self._processed_notes = notes
        self._midi_obj = midi_obj
        self.controls.set_generate_enabled(True)
        self._set_export_enabled(True)

        self.midi_preview.display(notes, self.controls.dynamic_velocity)
        self.status_bar.showMessage(
            f"Done — {len(notes)} notes generated"
        )

        from core.midi_exporter import get_temp_path
        self.drag_label.set_midi_path(get_temp_path())
        self._pipeline_worker = None

    def _on_pipeline_error(self, error_msg: str) -> None:
        self.controls.set_generate_enabled(True)
        QMessageBox.critical(self, "Pipeline Error", error_msg)
        self.status_bar.showMessage("Pipeline failed")
        self._pipeline_worker = None

    # --- Export ---

    def _set_export_enabled(self, enabled: bool) -> None:
        self.save_btn.setEnabled(enabled)
        self.open_system_btn.setEnabled(enabled)
        if not enabled:
            self.drag_label.set_midi_path(None)

    def _on_save_midi(self) -> None:
        if self._midi_obj is None:
            return
        last_dir = self._config.get("last_directory", str(Path.home()))
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save MIDI File",
            str(Path(last_dir) / "output.mid"),
            "MIDI Files (*.mid);;All Files (*)",
        )
        if path:
            save(self._midi_obj, Path(path))
            self.status_bar.showMessage(f"Saved: {path}")

    def _on_open_system(self) -> None:
        if self._midi_obj is None:
            return
        temp_path = save_temp(self._midi_obj)
        if sys.platform == 'win32':
            os.startfile(str(temp_path))
        else:
            subprocess.run(['xdg-open', str(temp_path)], check=False)
        self.status_bar.showMessage(f"Opened with system default: {temp_path.name}")

    # --- Theme ---

    def _toggle_theme(self) -> None:
        if self._config.get("theme") == "dark":
            self._config["theme"] = "light"
        else:
            self._config["theme"] = "dark"
        self._save_config()
        self._apply_theme()

    def _apply_theme(self) -> None:
        theme = self._config.get("theme", "dark")
        app = QApplication.instance()
        if theme == "dark":
            app.setStyleSheet(DARK_THEME)
            self.theme_action.setText("Switch to Light Mode")
        else:
            app.setStyleSheet(LIGHT_THEME)
            self.theme_action.setText("Switch to Dark Mode")

    # --- Config persistence ---

    def _load_config(self) -> dict:
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        if _CONFIG_FILE.exists():
            try:
                with open(_CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                # Merge with defaults for any missing keys
                merged = {**_DEFAULT_CONFIG, **config}
                return merged
            except (json.JSONDecodeError, OSError):
                pass
        return dict(_DEFAULT_CONFIG)

    def _save_config(self) -> None:
        self._config.update({
            "bpm": self.controls.bpm,
            "confidence_threshold": self.controls.confidence_threshold,
            "note_low": self.controls.note_low,
            "note_high": self.controls.note_high,
            "filter_ghost_notes": self.controls.filter_ghosts,
            "dynamic_velocity": self.controls.dynamic_velocity,
            "preserve_durations": self.controls.preserve_durations,
            "onset_threshold": self.controls.onset_threshold,
            "frame_threshold": self.controls.frame_threshold,
            "minimum_note_length_ms": self.controls.minimum_note_length_ms,
            "ensemble_passes": self.controls.ensemble_passes,
            "use_hpss": self.controls.use_hpss,
            "filter_harmonics": self.controls.do_filter_harmonics,
        })
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(_CONFIG_FILE, 'w') as f:
                json.dump(self._config, f, indent=2)
        except OSError:
            pass

    def _apply_config(self) -> None:
        self.controls.bpm = self._config.get("bpm", 120)
        self.controls.set_confidence(self._config.get("confidence_threshold", 0.5))
        self.controls.set_note_range(
            self._config.get("note_low", 36),
            self._config.get("note_high", 84),
        )
        self.controls.ghost_filter_check.setChecked(
            self._config.get("filter_ghost_notes", True)
        )
        self.controls.dynamic_velocity_check.setChecked(
            self._config.get("dynamic_velocity", False)
        )
        self.controls.preserve_durations_check.setChecked(
            self._config.get("preserve_durations", True)
        )
        self.controls.onset_slider.setValue(
            int(self._config.get("onset_threshold", 0.5) * 100)
        )
        self.controls.frame_slider.setValue(
            int(self._config.get("frame_threshold", 0.3) * 100)
        )
        self.controls.min_note_slider.setValue(
            int(self._config.get("minimum_note_length_ms", 58))
        )
        self.controls.ensemble_spin.setValue(
            self._config.get("ensemble_passes", 1)
        )
        self.controls.hpss_check.setChecked(
            self._config.get("use_hpss", False)
        )
        self.controls.harmonic_filter_check.setChecked(
            self._config.get("filter_harmonics", True)
        )

    def closeEvent(self, event) -> None:
        self._save_config()
        event.accept()

    # --- Help ---

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About MidiGen",
            "MidiGen v0.1\n\n"
            "Audio-to-MIDI transcription using machine learning.\n"
            "Built with PyQt6, basic-pitch, librosa, and pretty_midi.\n\n"
            "For EDM production workflows.",
        )
