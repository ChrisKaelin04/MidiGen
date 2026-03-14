"""Controls panel: BPM, note range, confidence, toggles, and Generate button."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QIntValidator
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


# MIDI note names for display
_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Range of MIDI notes we expose in the UI (C1 = 24 through C8 = 108)
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


# Quantize grid options displayed in the combo box
_GRID_OPTIONS = ["1/8", "1/16", "1/32", "1/8T", "1/16T"]


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
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(4)

        # === Top row: BPM | Confidence | Note Range ===
        top_row = QHBoxLayout()
        top_row.setSpacing(8)

        # -- BPM (manual entry only) --
        bpm_group = QGroupBox("BPM")
        bpm_group.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        bpm_layout = QHBoxLayout(bpm_group)
        bpm_layout.setContentsMargins(6, 14, 6, 4)

        self.bpm_edit = QLineEdit("120")
        self.bpm_edit.setValidator(QIntValidator(40, 300))
        self.bpm_edit.setFixedWidth(55)
        self.bpm_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.auto_detect_btn = QPushButton("Auto-Detect")
        self.auto_detect_btn.setToolTip("Detect BPM from loaded audio using librosa")

        bpm_layout.addWidget(self.bpm_edit)
        bpm_layout.addWidget(self.auto_detect_btn)
        top_row.addWidget(bpm_group)

        # -- Confidence Threshold --
        conf_group = QGroupBox("Confidence")
        conf_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        conf_layout = QHBoxLayout(conf_group)
        conf_layout.setContentsMargins(6, 14, 6, 4)

        conf_layout.addWidget(QLabel("Low"))
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(10, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.setMaximumHeight(20)
        conf_layout.addWidget(self.confidence_slider)
        conf_layout.addWidget(QLabel("High"))
        self.confidence_label = QLabel("0.50")
        self.confidence_label.setMinimumWidth(32)
        conf_layout.addWidget(self.confidence_label)
        top_row.addWidget(conf_group)

        # -- Note Range (two separate sliders: Low and High) --
        range_group = QGroupBox("Note Range")
        range_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        range_layout = QGridLayout(range_group)
        range_layout.setContentsMargins(6, 14, 6, 4)
        range_layout.setSpacing(2)

        range_layout.addWidget(QLabel("Low:"), 0, 0)
        self.range_low_slider = QSlider(Qt.Orientation.Horizontal)
        self.range_low_slider.setRange(_MIN_MIDI, _MAX_MIDI)
        self.range_low_slider.setValue(_MIN_MIDI + 12)  # C2
        self.range_low_slider.setMaximumHeight(20)
        self.range_low_label = QLabel(midi_to_name(_MIN_MIDI + 12))
        self.range_low_label.setMinimumWidth(32)
        range_layout.addWidget(self.range_low_slider, 0, 1)
        range_layout.addWidget(self.range_low_label, 0, 2)

        range_layout.addWidget(QLabel("High:"), 1, 0)
        self.range_high_slider = QSlider(Qt.Orientation.Horizontal)
        self.range_high_slider.setRange(_MIN_MIDI, _MAX_MIDI)
        self.range_high_slider.setValue(_MAX_MIDI)  # C8
        self.range_high_slider.setMaximumHeight(20)
        self.range_high_label = QLabel(midi_to_name(_MAX_MIDI))
        self.range_high_label.setMinimumWidth(32)
        range_layout.addWidget(self.range_high_slider, 1, 1)
        range_layout.addWidget(self.range_high_label, 1, 2)

        range_layout.setColumnStretch(1, 1)
        top_row.addWidget(range_group, 1)

        layout.addLayout(top_row)

        # === Preprocessing (runs BEFORE transcription — shown first to match pipeline order) ===
        # Future improvement: an optional wizard/baseline setter that asks the
        # user what type of sound it is (synth lead, chord pad, bass, etc.),
        # whether it's polyphonic, and other expectations — then auto-configures
        # all preprocessing and processing settings. For now, manual toggles.
        preproc_group = QGroupBox("Preprocessing (before transcription)")
        preproc_layout = QGridLayout(preproc_group)
        preproc_layout.setContentsMargins(6, 14, 6, 4)
        preproc_layout.setSpacing(4)

        self.hpss_check = QCheckBox("HPSS separation")
        self.hpss_check.setChecked(False)
        self.hpss_check.setToolTip(
            "Strip percussive content (drums, clicks) before transcription.\n"
            "Essential for real tracks with drums. Demucs preferred when available."
        )
        preproc_layout.addWidget(self.hpss_check, 0, 0)

        self.normalize_check = QCheckBox("Normalize loudness")
        self.normalize_check.setChecked(False)
        self.normalize_check.setToolTip(
            "Normalize audio to a consistent loudness level before\n"
            "transcription. Helps when source recordings have wildly\n"
            "different volumes."
        )
        preproc_layout.addWidget(self.normalize_check, 0, 1)

        self.noise_gate_check = QCheckBox("Noise gate")
        self.noise_gate_check.setChecked(False)
        self.noise_gate_check.setToolTip(
            "Silence low-level content (reverb tails, background noise)\n"
            "below a threshold. Gives the model cleaner note boundaries.\n"
            "May hurt sustained/quiet passages — use with care."
        )
        preproc_layout.addWidget(self.noise_gate_check, 0, 2)

        self.pre_emphasis_check = QCheckBox("Pre-emphasis EQ")
        self.pre_emphasis_check.setChecked(False)
        self.pre_emphasis_check.setToolTip(
            "Apply a gentle EQ boost in a target frequency range before\n"
            "transcription. Can help the model detect notes it's weak on."
        )
        preproc_layout.addWidget(self.pre_emphasis_check, 1, 0)

        # Pre-emphasis boost amount (user-settable)
        emph_row = QHBoxLayout()
        emph_row.addWidget(QLabel("Boost:"))
        self.pre_emphasis_boost_spin = QDoubleSpinBox()
        self.pre_emphasis_boost_spin.setRange(0.5, 6.0)
        self.pre_emphasis_boost_spin.setValue(1.5)
        self.pre_emphasis_boost_spin.setSingleStep(0.5)
        self.pre_emphasis_boost_spin.setSuffix(" dB")
        self.pre_emphasis_boost_spin.setFixedWidth(75)
        self.pre_emphasis_boost_spin.setToolTip(
            "Amount of EQ boost in the target frequency range.\n"
            "Keep it subtle (1-2 dB) to avoid distortion."
        )
        emph_row.addWidget(self.pre_emphasis_boost_spin)
        emph_row.addStretch()
        preproc_layout.addLayout(emph_row, 1, 1)

        layout.addWidget(preproc_group)

        # === Middle row: ML Model Tuning (grid) | Post-processing ===
        mid_row = QHBoxLayout()
        mid_row.setSpacing(8)

        # -- ML Model Tuning --
        ml_group = QGroupBox("ML Model Tuning")
        ml_grid = QGridLayout(ml_group)
        ml_grid.setContentsMargins(6, 14, 6, 4)
        ml_grid.setSpacing(4)

        # Row 0: Onset
        ml_grid.addWidget(QLabel("Onset:"), 0, 0)
        self.onset_slider = QSlider(Qt.Orientation.Horizontal)
        self.onset_slider.setRange(5, 95)
        self.onset_slider.setValue(50)
        self.onset_slider.setMaximumHeight(20)
        self.onset_slider.setToolTip(
            "How easily the model registers new note onsets. "
            "Lower = catches more notes in dense chords."
        )
        self.onset_label = QLabel("0.50")
        self.onset_label.setMinimumWidth(32)
        ml_grid.addWidget(self.onset_slider, 0, 1)
        ml_grid.addWidget(self.onset_label, 0, 2)

        # Row 0: Sustain (col 3-5)
        ml_grid.addWidget(QLabel("Sustain:"), 0, 3)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setRange(5, 95)
        self.frame_slider.setValue(30)
        self.frame_slider.setMaximumHeight(20)
        self.frame_slider.setToolTip(
            "How easily the model sustains a detected note. "
            "Lower = longer sustained notes (better for pads/chords)."
        )
        self.frame_label = QLabel("0.30")
        self.frame_label.setMinimumWidth(32)
        ml_grid.addWidget(self.frame_slider, 0, 4)
        ml_grid.addWidget(self.frame_label, 0, 5)

        # Row 1: Min note length
        ml_grid.addWidget(QLabel("Min note:"), 1, 0)
        self.min_note_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_note_slider.setRange(10, 300)
        self.min_note_slider.setValue(58)
        self.min_note_slider.setMaximumHeight(20)
        self.min_note_slider.setToolTip(
            "Shortest note the model will output, in milliseconds."
        )
        self.min_note_label = QLabel("58 ms")
        self.min_note_label.setMinimumWidth(40)
        ml_grid.addWidget(self.min_note_slider, 1, 1)
        ml_grid.addWidget(self.min_note_label, 1, 2)

        # Row 1: Ensemble passes (manual entry only)
        ml_grid.addWidget(QLabel("Ensemble:"), 1, 3)
        self.ensemble_edit = QLineEdit("1")
        self.ensemble_edit.setValidator(QIntValidator(1, 5))
        self.ensemble_edit.setFixedWidth(35)
        self.ensemble_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ensemble_edit.setToolTip(
            "Run the model multiple times with varying sensitivity and keep\n"
            "notes that appear consistently (majority vote).\n"
            "1 = single pass (fast), 3-5 = ensemble (slower but more accurate)."
        )
        ml_grid.addWidget(self.ensemble_edit, 1, 4)

        # Row 2: Melodia filter
        self.melodia_check = QCheckBox("Melodia filter")
        self.melodia_check.setChecked(True)
        self.melodia_check.setToolTip(
            "Use the Melodia algorithm to clean up pitch contours\n"
            "in basic-pitch output. Helps with cleaner monophonic lines.\n"
            "Try disabling for dense polyphonic material."
        )
        ml_grid.addWidget(self.melodia_check, 2, 0, 1, 3)

        ml_grid.setColumnStretch(1, 1)
        ml_grid.setColumnStretch(4, 1)

        mid_row.addWidget(ml_group, 2)

        # -- Post-processing (after transcription) --
        dsp_group = QGroupBox("Post-processing (after transcription)")
        dsp_layout = QGridLayout(dsp_group)
        dsp_layout.setContentsMargins(6, 14, 6, 4)
        dsp_layout.setSpacing(4)

        self.harmonic_filter_check = QCheckBox("Filter overtones")
        self.harmonic_filter_check.setChecked(True)
        self.harmonic_filter_check.setToolTip(
            "Remove notes that are likely overtones/harmonics of louder\n"
            "fundamental notes (octaves, 5ths, etc)."
        )

        self.ghost_filter_check = QCheckBox("Filter ghost notes")
        self.ghost_filter_check.setChecked(True)
        self.ghost_filter_check.setToolTip(
            "Remove notes shorter than one grid unit before quantization"
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
            "When off, all notes are forced to one grid unit length."
        )

        # Quantize grid combo
        grid_row = QHBoxLayout()
        grid_row.addWidget(QLabel("Grid:"))
        self.grid_combo = QComboBox()
        self.grid_combo.addItems(_GRID_OPTIONS)
        self.grid_combo.setCurrentText("1/16")
        self.grid_combo.setFixedWidth(65)
        self.grid_combo.setToolTip(
            "Quantization grid resolution.\n"
            "1/8 = eighth notes, 1/16 = sixteenth (default),\n"
            "1/32 = thirty-second, T = triplet variants."
        )
        grid_row.addWidget(self.grid_combo)
        grid_row.addStretch()

        # Velocity curve slider
        vel_row = QHBoxLayout()
        vel_row.addWidget(QLabel("Vel curve:"))
        self.velocity_curve_slider = QSlider(Qt.Orientation.Horizontal)
        self.velocity_curve_slider.setRange(30, 300)  # 0.30 to 3.00
        self.velocity_curve_slider.setValue(100)  # 1.00 = linear
        self.velocity_curve_slider.setMaximumHeight(20)
        self.velocity_curve_slider.setToolTip(
            "Velocity curve exponent (only with Dynamic Velocity ON).\n"
            "1.0 = linear, <1.0 = boost quiet notes, >1.0 = compress dynamics."
        )
        self.velocity_curve_label = QLabel("1.00")
        self.velocity_curve_label.setMinimumWidth(32)
        vel_row.addWidget(self.velocity_curve_slider, 1)
        vel_row.addWidget(self.velocity_curve_label)

        dsp_layout.addWidget(self.harmonic_filter_check, 0, 0)
        dsp_layout.addWidget(self.ghost_filter_check, 0, 1)
        dsp_layout.addWidget(self.dynamic_velocity_check, 1, 0)
        dsp_layout.addWidget(self.preserve_durations_check, 1, 1)
        dsp_layout.addLayout(grid_row, 2, 0)
        dsp_layout.addLayout(vel_row, 2, 1)

        mid_row.addWidget(dsp_group, 1)

        layout.addLayout(mid_row)

        # === Generate Button ===
        self.generate_btn = QPushButton("Generate MIDI")
        self.generate_btn.setMinimumHeight(36)
        self.generate_btn.setEnabled(False)
        layout.addWidget(self.generate_btn)

    def _connect_signals(self) -> None:
        self.generate_btn.clicked.connect(self.generate_requested.emit)
        self.auto_detect_btn.clicked.connect(self.auto_detect_bpm_requested.emit)

        # Range sliders <-> labels sync
        self.range_low_slider.valueChanged.connect(self._on_range_low_changed)
        self.range_high_slider.valueChanged.connect(self._on_range_high_changed)

        # Confidence label update
        self.confidence_slider.valueChanged.connect(self._on_confidence_changed)

        # ML slider label updates
        self.onset_slider.valueChanged.connect(self._on_onset_changed)
        self.frame_slider.valueChanged.connect(self._on_frame_changed)
        self.min_note_slider.valueChanged.connect(self._on_min_note_changed)

        # Velocity curve label update
        self.velocity_curve_slider.valueChanged.connect(self._on_velocity_curve_changed)

        # Settings changed notifications
        self.bpm_edit.editingFinished.connect(self.settings_changed.emit)
        self.range_low_slider.valueChanged.connect(self.settings_changed.emit)
        self.range_high_slider.valueChanged.connect(self.settings_changed.emit)
        self.confidence_slider.valueChanged.connect(self.settings_changed.emit)
        self.onset_slider.valueChanged.connect(self.settings_changed.emit)
        self.frame_slider.valueChanged.connect(self.settings_changed.emit)
        self.min_note_slider.valueChanged.connect(self.settings_changed.emit)
        self.ensemble_edit.editingFinished.connect(self.settings_changed.emit)
        self.hpss_check.stateChanged.connect(self.settings_changed.emit)
        self.harmonic_filter_check.stateChanged.connect(self.settings_changed.emit)
        self.ghost_filter_check.stateChanged.connect(self.settings_changed.emit)
        self.dynamic_velocity_check.stateChanged.connect(self.settings_changed.emit)
        self.preserve_durations_check.stateChanged.connect(self.settings_changed.emit)
        self.grid_combo.currentTextChanged.connect(lambda _: self.settings_changed.emit())
        self.melodia_check.stateChanged.connect(self.settings_changed.emit)
        self.velocity_curve_slider.valueChanged.connect(self.settings_changed.emit)
        self.normalize_check.stateChanged.connect(self.settings_changed.emit)
        self.noise_gate_check.stateChanged.connect(self.settings_changed.emit)
        self.pre_emphasis_check.stateChanged.connect(self.settings_changed.emit)
        self.pre_emphasis_boost_spin.valueChanged.connect(lambda _: self.settings_changed.emit())

    def _on_range_low_changed(self, value: int) -> None:
        # Don't let low exceed high
        high = self.range_high_slider.value()
        if value > high:
            self.range_low_slider.blockSignals(True)
            self.range_low_slider.setValue(high)
            self.range_low_slider.blockSignals(False)
            value = high
        self.range_low_label.setText(midi_to_name(value))

    def _on_range_high_changed(self, value: int) -> None:
        # Don't let high go below low
        low = self.range_low_slider.value()
        if value < low:
            self.range_high_slider.blockSignals(True)
            self.range_high_slider.setValue(low)
            self.range_high_slider.blockSignals(False)
            value = low
        self.range_high_label.setText(midi_to_name(value))

    def _on_confidence_changed(self, value: int) -> None:
        self.confidence_label.setText(f"{value / 100:.2f}")

    def _on_onset_changed(self, value: int) -> None:
        self.onset_label.setText(f"{value / 100:.2f}")

    def _on_frame_changed(self, value: int) -> None:
        self.frame_label.setText(f"{value / 100:.2f}")

    def _on_min_note_changed(self, value: int) -> None:
        self.min_note_label.setText(f"{value} ms")

    def _on_velocity_curve_changed(self, value: int) -> None:
        self.velocity_curve_label.setText(f"{value / 100:.2f}")

    # --- Public API ---

    @property
    def bpm(self) -> int:
        text = self.bpm_edit.text().strip()
        try:
            return max(40, min(300, int(text)))
        except ValueError:
            return 120

    @bpm.setter
    def bpm(self, value: int) -> None:
        self.bpm_edit.setText(str(value))

    @property
    def note_low(self) -> int:
        return self.range_low_slider.value()

    @property
    def note_high(self) -> int:
        return self.range_high_slider.value()

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
        text = self.ensemble_edit.text().strip()
        try:
            return max(1, min(5, int(text)))
        except ValueError:
            return 1

    @property
    def use_hpss(self) -> bool:
        return self.hpss_check.isChecked()

    @property
    def do_filter_harmonics(self) -> bool:
        return self.harmonic_filter_check.isChecked()

    @property
    def quantize_grid(self) -> str:
        return self.grid_combo.currentText()

    @property
    def melodia_trick(self) -> bool:
        return self.melodia_check.isChecked()

    @property
    def velocity_curve(self) -> float:
        return self.velocity_curve_slider.value() / 100.0

    @property
    def normalize_loudness(self) -> bool:
        return self.normalize_check.isChecked()

    @property
    def noise_gate(self) -> bool:
        return self.noise_gate_check.isChecked()

    @property
    def pre_emphasis(self) -> bool:
        return self.pre_emphasis_check.isChecked()

    @property
    def pre_emphasis_boost_db(self) -> float:
        return self.pre_emphasis_boost_spin.value()

    def set_note_range(self, low: int, high: int) -> None:
        """Set the note range from MIDI numbers."""
        self.range_low_slider.setValue(low)
        self.range_high_slider.setValue(high)

    def set_confidence(self, value: float) -> None:
        """Set the confidence threshold (0.0-1.0)."""
        self.confidence_slider.setValue(int(round(value * 100)))

    def set_generate_enabled(self, enabled: bool) -> None:
        """Enable or disable the Generate MIDI button."""
        self.generate_btn.setEnabled(enabled)
