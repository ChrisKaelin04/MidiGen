"""Main application window — wires all panels, manages app state and pipeline."""

import json
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
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from core import AudioData, NoteEvent, ProcessingConfig
from core.audio_loader import load_audio, detect_bpm, apply_hpss
from core.transcriber import transcribe
from core.spectral_validator import spectral_validate
from core.midi_processor import process
from core.midi_exporter import build_midi, save, save_temp
from gui.controls_panel import ControlsPanel
from gui.waveform_view import WaveformView

import pretty_midi

_CONFIG_DIR = Path.home() / '.midigen'
_CONFIG_FILE = _CONFIG_DIR / 'config.json'

_DEFAULT_CONFIG = {
    "bpm": 120,
    "confidence_threshold": 0.5,
    "note_low": 36,
    "note_high": 108,
    "filter_ghost_notes": True,
    "dynamic_velocity": False,
    "preserve_durations": True,
    "onset_threshold": 0.5,
    "frame_threshold": 0.3,
    "minimum_note_length_ms": 58,
    "ensemble_passes": 3,
    "use_hpss": False,
    "filter_harmonics": True,
    "quantize_grid": "1/16",
    "melodia_trick": True,
    "velocity_curve": 1.0,
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
    padding: 6px 12px;
    color: #f0f0f0;
}
QPushButton:hover {
    background-color: #3a3a3a;
    border-color: #4a9eff;
}
QPushButton:pressed {
    background-color: #4a9eff;
}
QPushButton:checked {
    background-color: #4a9eff;
    border-color: #4a9eff;
    color: #ffffff;
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
    min-width: 200px;
    padding: 4px 0px;
}
QMenu::item {
    padding: 6px 30px 6px 20px;
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
    padding: 6px 12px;
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
QPushButton:checked {
    background-color: #0066cc;
    border-color: #0066cc;
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
    min-width: 200px;
    padding: 4px 0px;
}
QMenu::item {
    padding: 6px 30px 6px 20px;
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
                melodia_trick=self.config.melodia_trick,
            )

            if self.config.spectral_validate:
                self.status.emit("Validating notes against spectrogram...")
                notes = spectral_validate(
                    audio, notes,
                    energy_floor_db=self.config.spectral_energy_floor_db,
                    overtone_margin_db=self.config.spectral_overtone_margin_db,
                    recovery_min_energy_db=self.config.spectral_recovery_min_db,
                    recovery_min_duration_sec=self.config.spectral_recovery_min_dur,
                    blind_spot_boost_db=self.config.spectral_blind_spot_boost_db,
                    do_recover=self.config.spectral_do_recover,
                    do_resolve_overlaps=self.config.spectral_do_resolve_overlaps,
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
                harmonic_filter_mode=self.config.harmonic_filter_mode,
                quantize_grid=self.config.quantize_grid,
                velocity_curve=self.config.velocity_curve,
                do_merge_fragments=self.config.merge_fragments,
                fragment_gap_tol=self.config.fragment_gap_tol,
                fragment_reattack_ratio=self.config.fragment_reattack_ratio,
                timing_offset_grid=self.config.timing_offset_grid,
                do_fill_patterns=self.config.fill_patterns,
                pattern_min_reps=self.config.pattern_min_reps,
                pattern_fill_threshold=self.config.pattern_fill_threshold,
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
        guide_action = QAction("&Controls Guide", self)
        guide_action.setShortcut("F1")
        guide_action.triggered.connect(self._show_controls_guide)
        help_menu.addAction(guide_action)
        help_menu.addSeparator()
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

        # Main visualization — spectrogram with MIDI overlay
        self.waveform_view = WaveformView()
        main_layout.addWidget(self.waveform_view, 1)

        # Controls panel — compact, fixed height, NOT in splitter
        self.controls = ControlsPanel()
        self.controls.generate_btn.setObjectName("generateBtn")
        self.controls.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum
        )
        main_layout.addWidget(self.controls)

        # Export row
        export_row = QHBoxLayout()

        self.save_btn = QPushButton("Save .mid File")
        self.save_btn.setEnabled(False)
        export_row.addWidget(self.save_btn)

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
        self.controls.range_low_slider.valueChanged.connect(self._on_note_range_changed)
        self.controls.range_high_slider.valueChanged.connect(self._on_note_range_changed)

    def _on_note_range_changed(self) -> None:
        self.waveform_view.set_note_range_zoom(
            self.controls.note_low, self.controls.note_high
        )

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
            self.waveform_view.clear_midi()
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
            quantize_grid=self.controls.quantize_grid,
            melodia_trick=self.controls.melodia_trick,
            velocity_curve=self.controls.velocity_curve,
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

        self.waveform_view.display_midi(
            notes, self.controls.dynamic_velocity,
            selection_start=self.waveform_view.start_sec,
        )
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
            "quantize_grid": self.controls.quantize_grid,
            "melodia_trick": self.controls.melodia_trick,
            "velocity_curve": self.controls.velocity_curve,
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
        note_low = self._config.get("note_low", 36)
        note_high = self._config.get("note_high", 108)
        # Sanity check — ensure within bounds
        if note_low < 24 or note_high > 108 or note_low >= note_high:
            note_low, note_high = 36, 108
        self.controls.set_note_range(note_low, note_high)
        self.waveform_view.set_note_range_zoom(note_low, note_high)
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
        self.controls.ensemble_edit.setText(
            str(self._config.get("ensemble_passes", 1))
        )
        self.controls.hpss_check.setChecked(
            self._config.get("use_hpss", False)
        )
        self.controls.harmonic_filter_check.setChecked(
            self._config.get("filter_harmonics", True)
        )
        grid = self._config.get("quantize_grid", "1/16")
        idx = self.controls.grid_combo.findText(grid)
        if idx >= 0:
            self.controls.grid_combo.setCurrentIndex(idx)
        self.controls.melodia_check.setChecked(
            self._config.get("melodia_trick", True)
        )
        self.controls.velocity_curve_slider.setValue(
            int(self._config.get("velocity_curve", 1.0) * 100)
        )

    def closeEvent(self, event) -> None:
        self._save_config()
        event.accept()

    # --- Help ---

    def _show_controls_guide(self) -> None:
        from PyQt6.QtWidgets import QDialog, QTextBrowser, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("MidiGen Controls Guide")
        dlg.resize(680, 600)
        layout = QVBoxLayout(dlg)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml(_CONTROLS_GUIDE_HTML)
        layout.addWidget(browser)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(dlg.accept)
        layout.addWidget(btn_box)

        dlg.exec()

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About MidiGen",
            "MidiGen v0.1\n\n"
            "Audio-to-MIDI transcription using machine learning.\n"
            "Built with PyQt6, basic-pitch, librosa, and pretty_midi.\n\n"
            "For EDM production workflows.",
        )


_CONTROLS_GUIDE_HTML = """
<h2>MidiGen Controls Guide</h2>

<h3>General Workflow</h3>
<p>
1. <b>Load an audio file</b> (MP3, WAV, or FLAC).<br>
2. Optionally <b>select a time region</b> by dragging the blue handles on the spectrogram.<br>
3. <b>Adjust parameters</b> below to taste.<br>
4. Click <b>Generate MIDI</b>.<br>
5. Review the results in <b>Overlap</b> or <b>MIDI</b> view, tweak, and regenerate as needed.<br>
6. <b>Save</b> or <b>drag</b> the .mid file into your DAW.
</p>

<hr>
<h3>BPM</h3>
<p>
Beats per minute — used for quantizing detected notes to a rhythmic grid.
Type the value directly.
</p>
<ul>
<li><b>Auto-Detect</b> — Analyzes the loaded audio and estimates the tempo using librosa.
Works well on rhythmic material; may be inaccurate on rubato or ambient pieces.</li>
<li><b>Recommended:</b> Use Auto-Detect as a starting point, then round to the nearest
whole number if you know the track's BPM.</li>
</ul>

<hr>
<h3>Confidence Threshold</h3>
<p>
Minimum amplitude a detected note must have to be kept. Filters out quiet artifacts.
The slider goes from 0.10 (keep almost everything) to 1.00 (only the loudest notes).
</p>
<ul>
<li><b>0.30 – 0.50</b> — Good default range for most material.</li>
<li><b>Lower (0.15 – 0.25)</b> — Use for quiet or pad-heavy material where you want every note.</li>
<li><b>Higher (0.60+)</b> — Use to cut through noisy mixes and keep only dominant notes.</li>
</ul>

<hr>
<h3>Note Range</h3>
<p>
A dual-handle slider setting the lowest and highest MIDI notes to keep.
Notes outside this range are discarded after transcription. Drag the left handle
to set the low cutoff and the right handle to set the high cutoff.
The labels on each side update to show the current note names.
</p>
<ul>
<li><b>C2 – C6 (default)</b> — Covers most melodic content.</li>
<li><b>Widen to C1 – C7</b> — For bass-heavy tracks or material with high harmonics.</li>
<li><b>Narrow range</b> — Useful for isolating bass (C1 – C3) or lead melodies (C4 – C7).</li>
</ul>

<hr>
<h3>ML Model Tuning</h3>

<p><b>Onset Sensitivity</b> (0.05 – 0.95)</p>
<p>
Controls how easily the model registers a new note starting. Lower values detect more
note onsets, especially in dense chords. Higher values are more selective.
</p>
<ul>
<li><b>0.50 (default)</b> — Balanced for most material.</li>
<li><b>0.20 – 0.30</b> — Better for dense chords and rapid passages.</li>
<li><b>0.60 – 0.80</b> — Cleaner output on simple melodies, but may miss notes in chords.</li>
</ul>

<p><b>Sustain Sensitivity</b> (0.05 – 0.95)</p>
<p>
Controls how easily the model sustains an already-detected note. Lower values produce
longer notes (good for pads and sustained chords). Higher values cut notes short.
</p>
<ul>
<li><b>0.30 (default)</b> — Good general starting point.</li>
<li><b>0.10 – 0.20</b> — Better for pads, sustained chords, and legato passages.</li>
<li><b>0.50+</b> — Tighter output, better for staccato/percussive material.</li>
</ul>

<p><b>Min Note Length</b> (10 – 300 ms)</p>
<p>
Shortest note the model will output. Filters out extremely brief artifacts.
</p>
<ul>
<li><b>58 ms (default)</b> — Catches most articulations without too much noise.</li>
<li><b>25 – 40 ms</b> — For very fast arpeggios or grace notes.</li>
<li><b>80 – 150 ms</b> — Cleaner output, removes fast artifacts but may miss quick notes.</li>
</ul>

<p><b>Ensemble Passes</b> (1 – 5)</p>
<p>
Runs the ML model multiple times with different sensitivity presets and keeps only
notes that appear consistently across passes (majority vote). This dramatically
reduces false positives (overtones, ghost notes) while keeping real notes.
</p>
<ul>
<li><b>1 (default)</b> — Single pass. Fastest, uses your slider settings directly.</li>
<li><b>3</b> — Good balance of quality and speed. Recommended for most use.</li>
<li><b>5</b> — Maximum quality. Best for complex material, but takes 5x longer.</li>
</ul>
<p><i>Note: In ensemble mode, the Onset/Sustain/Min Note sliders are ignored — the
ensemble uses its own built-in presets ranging from conservative to aggressive.</i></p>

<p><b>Melodia Filter</b></p>
<p>
Uses the Melodia algorithm to clean up pitch contours in the model output.
On by default. Try disabling for dense polyphonic material where Melodia
may incorrectly suppress valid simultaneous notes.
</p>

<hr>
<h3>DSP &amp; Options</h3>

<p><b>HPSS Separation</b></p>
<p>
Applies Harmonic/Percussive Source Separation to strip drums and percussive clicks
before transcription. Essential for real tracks with drums. Not needed for clean
piano or synth recordings.
</p>

<p><b>Filter Overtones</b></p>
<p>
Removes notes that are likely harmonics/overtones of louder fundamental notes
(octaves, fifths, etc). Uses a 1.5x amplitude ratio.
</p>
<ul>
<li><b>ON (default)</b> — Good for melodies, arpeggios, and single-note lines.</li>
<li><b>OFF</b> — Better for chords and pads, where "overtones" may actually be
real notes in the chord.</li>
</ul>

<p><b>Filter Ghost Notes</b></p>
<p>
Removes notes shorter than a 1/16th note (based on BPM) before quantization.
Cleans up very short artifacts. Leave ON for most use.
</p>

<p><b>Dynamic Velocity</b></p>
<p>
When ON, maps the detected amplitude of each note to MIDI velocity (0–127).
When OFF (default), all notes get a fixed velocity of 100.
Useful for expressive performances; less useful for electronic music where
you want consistent velocity.
</p>

<p><b>Preserve Note Lengths</b></p>
<p>
When ON, detected note durations are kept (snapped to the rhythmic grid).
When OFF, all notes are forced to one grid unit length — useful for
re-triggering synths or samplers.
</p>

<p><b>Quantize Grid</b></p>
<p>
Sets the rhythmic resolution for quantization. All note start times and
durations are snapped to this grid.
</p>
<ul>
<li><b>1/8</b> — Eighth notes. Coarse grid, good for slow material.</li>
<li><b>1/16 (default)</b> — Sixteenth notes. Standard resolution for most music.</li>
<li><b>1/32</b> — Thirty-second notes. Fine grid for fast passages.</li>
<li><b>1/8T</b> — Eighth-note triplets.</li>
<li><b>1/16T</b> — Sixteenth-note triplets.</li>
</ul>

<p><b>Velocity Curve</b></p>
<p>
When Dynamic Velocity is ON, this exponent shapes the amplitude-to-velocity
mapping. Only affects the MIDI output, not detection.
</p>
<ul>
<li><b>1.00 (default)</b> — Linear mapping.</li>
<li><b>&lt; 1.0 (e.g. 0.50)</b> — Boosts quiet notes, more dynamic range visible.</li>
<li><b>&gt; 1.0 (e.g. 2.00)</b> — Compresses dynamics, quieter notes become even quieter.</li>
</ul>

<hr>
<h3>View Modes (Spectrogram)</h3>
<ul>
<li><b>Frequencies</b> — Shows the mel spectrogram heatmap only. Always available.</li>
<li><b>Overlap</b> — Shows MIDI notes overlaid on the spectrogram so you can visually
compare detected notes against the frequency content. Available after generating MIDI.</li>
<li><b>MIDI</b> — Shows a piano roll view of MIDI notes only, with pitch labels.
Available after generating MIDI.</li>
</ul>

<hr>
<h3>Quick-Start Presets by Material Type</h3>
<table border="1" cellpadding="4" cellspacing="0">
<tr>
  <th>Material</th><th>Confidence</th><th>Onset</th><th>Sustain</th>
  <th>Overtones</th><th>HPSS</th><th>Ensemble</th>
</tr>
<tr>
  <td>Clean piano / melody</td><td>0.50</td><td>0.50</td><td>0.30</td>
  <td>ON</td><td>OFF</td><td>1</td>
</tr>
<tr>
  <td>Dense chords</td><td>0.23</td><td>0.55</td><td>0.15</td>
  <td>OFF</td><td>OFF</td><td>3</td>
</tr>
<tr>
  <td>Sustained pads</td><td>0.30</td><td>0.40</td><td>0.15</td>
  <td>OFF</td><td>OFF</td><td>1</td>
</tr>
<tr>
  <td>Full mix with drums</td><td>0.40</td><td>0.50</td><td>0.30</td>
  <td>ON</td><td>ON</td><td>3</td>
</tr>
<tr>
  <td>Fast arpeggios</td><td>0.50</td><td>0.50</td><td>0.30</td>
  <td>ON</td><td>OFF</td><td>1</td>
</tr>
</table>
"""
