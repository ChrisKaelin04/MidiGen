"""Controls panel: BPM, note range, confidence, toggles, and Generate button."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from superqt import QRangeSlider


# MIDI note names for display
_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Range of MIDI notes we expose in the UI (C0 = 24 through C8 = 108)
_MIN_MIDI = 24
_MAX_MIDI = 108


def midi_to_name(midi_num: int) -> str:
    """Convert a MIDI note number to a human-readable name like C4, F#2."""
    octave = (midi_num // 12) - 1
    note = _NOTE_NAMES[midi_num % 12]
    return f"{note}{octave}"


def _build_note_list() -> list[tuple[str, int]]:
    """Build a list of (name, midi_number) tuples for the full range."""
    return [(midi_to_name(m), m) for m in range(_MIN_MIDI, _MAX_MIDI + 1)]


class ControlsPanel(QWidget):
    """Panel containing all user-adjustable controls for the processing pipeline."""

    generate_requested = pyqtSignal()
    auto_detect_bpm_requested = pyqtSignal()
    settings_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._note_list = _build_note_list()
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # --- BPM Section ---
        bpm_group = QGroupBox("BPM")
        bpm_layout = QHBoxLayout(bpm_group)

        self.bpm_spin = QSpinBox()
        self.bpm_spin.setRange(40, 300)
        self.bpm_spin.setValue(120)
        self.bpm_spin.setPrefix("BPM: ")

        self.auto_detect_btn = QPushButton("Auto-Detect")
        self.auto_detect_btn.setToolTip("Detect BPM from loaded audio using librosa")

        bpm_layout.addWidget(self.bpm_spin)
        bpm_layout.addWidget(self.auto_detect_btn)
        layout.addWidget(bpm_group)

        # --- Note Range Section ---
        range_group = QGroupBox("Note Range")
        range_layout = QVBoxLayout(range_group)

        # Dual-handle slider
        self.range_slider = QRangeSlider(Qt.Orientation.Horizontal)
        self.range_slider.setRange(_MIN_MIDI, _MAX_MIDI)
        self.range_slider.setValue((_MIN_MIDI + 12, _MAX_MIDI - 24))  # C2 to C6 default

        slider_label_layout = QHBoxLayout()
        self.range_low_label = QLabel(midi_to_name(_MIN_MIDI + 12))
        self.range_high_label = QLabel(midi_to_name(_MAX_MIDI - 24))
        slider_label_layout.addWidget(self.range_low_label)
        slider_label_layout.addStretch()
        slider_label_layout.addWidget(self.range_high_label)

        range_layout.addWidget(self.range_slider)
        range_layout.addLayout(slider_label_layout)

        # Dropdown combos
        combo_layout = QHBoxLayout()
        combo_layout.addWidget(QLabel("Low:"))
        self.low_combo = QComboBox()
        for name, midi_num in self._note_list:
            self.low_combo.addItem(name, midi_num)
        combo_layout.addWidget(self.low_combo)

        combo_layout.addSpacing(16)

        combo_layout.addWidget(QLabel("High:"))
        self.high_combo = QComboBox()
        for name, midi_num in self._note_list:
            self.high_combo.addItem(name, midi_num)
        combo_layout.addWidget(self.high_combo)

        range_layout.addLayout(combo_layout)
        layout.addWidget(range_group)

        # Set default combo selections to match slider
        self._set_combo_by_midi(self.low_combo, 36)   # C2
        self._set_combo_by_midi(self.high_combo, 84)  # C6

        # --- Confidence Threshold Section ---
        conf_group = QGroupBox("Confidence Threshold")
        conf_layout = QHBoxLayout(conf_group)

        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(10, 100)
        self.confidence_slider.setValue(50)

        self.confidence_label = QLabel("0.50")
        self.confidence_label.setMinimumWidth(36)

        conf_layout.addWidget(QLabel("Low"))
        conf_layout.addWidget(self.confidence_slider)
        conf_layout.addWidget(QLabel("High"))
        conf_layout.addWidget(self.confidence_label)
        layout.addWidget(conf_group)

        # --- ML Model Tuning ---
        ml_group = QGroupBox("ML Model Tuning")
        ml_layout = QVBoxLayout(ml_group)

        # Onset threshold
        onset_row = QHBoxLayout()
        onset_row.addWidget(QLabel("Onset sensitivity:"))
        self.onset_slider = QSlider(Qt.Orientation.Horizontal)
        self.onset_slider.setRange(5, 95)
        self.onset_slider.setValue(50)
        self.onset_slider.setToolTip(
            "How easily the model registers new note onsets. "
            "Lower = catches more notes in dense chords."
        )
        self.onset_label = QLabel("0.50")
        self.onset_label.setMinimumWidth(36)
        onset_row.addWidget(self.onset_slider)
        onset_row.addWidget(self.onset_label)
        ml_layout.addLayout(onset_row)

        # Frame threshold
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Sustain sensitivity:"))
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(5, 95)
        self.frame_slider.setValue(30)
        self.frame_slider.setToolTip(
            "How easily the model sustains a detected note. "
            "Lower = longer sustained notes (better for pads/chords)."
        )
        self.frame_label = QLabel("0.30")
        self.frame_label.setMinimumWidth(36)
        frame_row.addWidget(self.frame_slider)
        frame_row.addWidget(self.frame_label)
        ml_layout.addLayout(frame_row)

        # Minimum note length
        notelen_row = QHBoxLayout()
        notelen_row.addWidget(QLabel("Min note length:"))
        self.min_note_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_note_slider.setRange(10, 300)
        self.min_note_slider.setValue(58)
        self.min_note_slider.setToolTip(
            "Shortest note the model will output, in milliseconds. "
            "Lower = catches faster articulations."
        )
        self.min_note_label = QLabel("58 ms")
        self.min_note_label.setMinimumWidth(50)
        notelen_row.addWidget(self.min_note_slider)
        notelen_row.addWidget(self.min_note_label)
        ml_layout.addLayout(notelen_row)

        # Ensemble passes
        ensemble_row = QHBoxLayout()
        ensemble_row.addWidget(QLabel("Ensemble passes:"))
        self.ensemble_spin = QSpinBox()
        self.ensemble_spin.setRange(1, 5)
        self.ensemble_spin.setValue(1)
        self.ensemble_spin.setToolTip(
            "Run the model multiple times with varying sensitivity and keep\n"
            "notes that appear consistently (majority vote).\n"
            "1 = single pass (fast), 3-5 = ensemble (slower but more accurate)."
        )
        ensemble_row.addWidget(self.ensemble_spin)
        ensemble_row.addStretch()
        ml_layout.addLayout(ensemble_row)

        layout.addWidget(ml_group)

        # --- DSP / Filtering ---
        dsp_group = QGroupBox("DSP & Filtering")
        dsp_layout = QVBoxLayout(dsp_group)

        self.hpss_check = QCheckBox("Harmonic separation (HPSS)")
        self.hpss_check.setChecked(False)
        self.hpss_check.setToolTip(
            "Strip percussive content (drums, clicks) before transcription.\n"
            "Essential for real tracks with drums. Not needed for clean piano."
        )
        dsp_layout.addWidget(self.hpss_check)

        self.harmonic_filter_check = QCheckBox("Filter overtones")
        self.harmonic_filter_check.setChecked(True)
        self.harmonic_filter_check.setToolTip(
            "Remove notes that are likely overtones/harmonics of louder\n"
            "fundamental notes (octaves, 5ths, etc)."
        )
        dsp_layout.addWidget(self.harmonic_filter_check)

        layout.addWidget(dsp_group)

        # --- Toggles ---
        toggle_layout = QHBoxLayout()

        self.ghost_filter_check = QCheckBox("Filter ghost notes")
        self.ghost_filter_check.setChecked(True)
        self.ghost_filter_check.setToolTip(
            "Remove notes shorter than 1/16th note before quantization"
        )

        self.dynamic_velocity_check = QCheckBox("Dynamic velocity")
        self.dynamic_velocity_check.setChecked(False)
        self.dynamic_velocity_check.setToolTip(
            "Map detected amplitude to MIDI velocity (otherwise fixed at 100)"
        )

        self.preserve_durations_check = QCheckBox("Preserve note lengths")
        self.preserve_durations_check.setChecked(True)
        self.preserve_durations_check.setToolTip(
            "Keep detected note durations (snapped to grid). "
            "When off, all notes are forced to 1/16th length."
        )

        toggle_layout.addWidget(self.ghost_filter_check)
        toggle_layout.addWidget(self.dynamic_velocity_check)
        toggle_layout.addWidget(self.preserve_durations_check)
        layout.addLayout(toggle_layout)

        # --- Generate Button ---
        self.generate_btn = QPushButton("Generate MIDI")
        self.generate_btn.setMinimumHeight(40)
        self.generate_btn.setEnabled(False)
        layout.addWidget(self.generate_btn)

        layout.addStretch()

    def _connect_signals(self) -> None:
        self.generate_btn.clicked.connect(self.generate_requested.emit)
        self.auto_detect_btn.clicked.connect(self.auto_detect_bpm_requested.emit)

        # Slider <-> combo bidirectional sync
        self.range_slider.valueChanged.connect(self._on_slider_changed)
        self.low_combo.currentIndexChanged.connect(self._on_low_combo_changed)
        self.high_combo.currentIndexChanged.connect(self._on_high_combo_changed)

        # Confidence label update
        self.confidence_slider.valueChanged.connect(self._on_confidence_changed)

        # ML slider label updates
        self.onset_slider.valueChanged.connect(self._on_onset_changed)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        self.min_note_slider.valueChanged.connect(self._on_min_note_changed)

        # Settings changed notifications
        self.bpm_spin.valueChanged.connect(self.settings_changed.emit)
        self.range_slider.valueChanged.connect(self.settings_changed.emit)
        self.confidence_slider.valueChanged.connect(self.settings_changed.emit)
        self.onset_slider.valueChanged.connect(self.settings_changed.emit)
        self.frame_slider.valueChanged.connect(self.settings_changed.emit)
        self.min_note_slider.valueChanged.connect(self.settings_changed.emit)
        self.ensemble_spin.valueChanged.connect(self.settings_changed.emit)
        self.hpss_check.stateChanged.connect(self.settings_changed.emit)
        self.harmonic_filter_check.stateChanged.connect(self.settings_changed.emit)
        self.ghost_filter_check.stateChanged.connect(self.settings_changed.emit)
        self.dynamic_velocity_check.stateChanged.connect(self.settings_changed.emit)
        self.preserve_durations_check.stateChanged.connect(self.settings_changed.emit)

    def _on_slider_changed(self, value: tuple[int, int]) -> None:
        low, high = value
        self.range_low_label.setText(midi_to_name(low))
        self.range_high_label.setText(midi_to_name(high))
        # Sync combos without re-triggering slider update
        self.low_combo.blockSignals(True)
        self.high_combo.blockSignals(True)
        self._set_combo_by_midi(self.low_combo, low)
        self._set_combo_by_midi(self.high_combo, high)
        self.low_combo.blockSignals(False)
        self.high_combo.blockSignals(False)

    def _on_low_combo_changed(self, index: int) -> None:
        midi_num = self.low_combo.currentData()
        if midi_num is None:
            return
        self.range_slider.blockSignals(True)
        _, high = self.range_slider.value()
        self.range_slider.setValue((midi_num, max(midi_num, high)))
        self.range_low_label.setText(midi_to_name(midi_num))
        self.range_slider.blockSignals(False)

    def _on_high_combo_changed(self, index: int) -> None:
        midi_num = self.high_combo.currentData()
        if midi_num is None:
            return
        self.range_slider.blockSignals(True)
        low, _ = self.range_slider.value()
        self.range_slider.setValue((min(midi_num, low), midi_num))
        self.range_high_label.setText(midi_to_name(midi_num))
        self.range_slider.blockSignals(False)

    def _on_confidence_changed(self, value: int) -> None:
        self.confidence_label.setText(f"{value / 100:.2f}")

    def _on_onset_changed(self, value: int) -> None:
        self.onset_label.setText(f"{value / 100:.2f}")

    def _on_frame_changed(self, value: int) -> None:
        self.frame_label.setText(f"{value / 100:.2f}")

    def _on_min_note_changed(self, value: int) -> None:
        self.min_note_label.setText(f"{value} ms")

    def _set_combo_by_midi(self, combo: QComboBox, midi_num: int) -> None:
        index = combo.findData(midi_num)
        if index >= 0:
            combo.setCurrentIndex(index)

    # --- Public API ---

    @property
    def bpm(self) -> int:
        return self.bpm_spin.value()

    @bpm.setter
    def bpm(self, value: int) -> None:
        self.bpm_spin.setValue(value)

    @property
    def note_low(self) -> int:
        low, _ = self.range_slider.value()
        return low

    @property
    def note_high(self) -> int:
        _, high = self.range_slider.value()
        return high

    @property
    def confidence_threshold(self) -> float:
        return self.confidence_slider.value() / 100.0

    @property
    def filter_ghosts(self) -> bool:
        return self.ghost_filter_check.isChecked()

    @property
    def dynamic_velocity(self) -> bool:
        return self.dynamic_velocity_check.isChecked()

    @property
    def preserve_durations(self) -> bool:
        return self.preserve_durations_check.isChecked()

    @property
    def onset_threshold(self) -> float:
        return self.onset_slider.value() / 100.0

    @property
    def frame_threshold(self) -> float:
        return self.frame_slider.value() / 100.0

    @property
    def minimum_note_length_ms(self) -> float:
        return float(self.min_note_slider.value())

    @property
    def ensemble_passes(self) -> int:
        return self.ensemble_spin.value()

    @property
    def use_hpss(self) -> bool:
        return self.hpss_check.isChecked()

    @property
    def do_filter_harmonics(self) -> bool:
        return self.harmonic_filter_check.isChecked()

    def set_note_range(self, low: int, high: int) -> None:
        """Set the note range from MIDI numbers."""
        self.range_slider.setValue((low, high))

    def set_confidence(self, value: float) -> None:
        """Set the confidence threshold (0.0-1.0)."""
        self.confidence_slider.setValue(int(round(value * 100)))

    def set_generate_enabled(self, enabled: bool) -> None:
        """Enable or disable the Generate MIDI button."""
        self.generate_btn.setEnabled(enabled)
