"""Unified CQT spectrogram + MIDI overlay display with playback and selection."""

import time

from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
import numpy as np
import pyqtgraph as pg
import librosa
import sounddevice as sd

from core import NoteEvent
from gui.controls_panel import midi_to_name


def _build_inferno_cmap() -> pg.ColorMap:
    """Build an inferno-like colormap for the spectrogram."""
    positions = [0.0, 0.15, 0.35, 0.55, 0.75, 0.90, 1.0]
    colors = [
        (0, 0, 4),
        (40, 11, 84),
        (101, 21, 110),
        (182, 55, 84),
        (246, 136, 32),
        (252, 206, 46),
        (252, 252, 164),
    ]
    return pg.ColorMap(positions, colors)


# View modes
MODE_FREQ = 0
MODE_OVERLAP = 1
MODE_MIDI = 2

# CQT parameters — Y axis maps directly to MIDI note numbers
_CQT_MIDI_MIN = 24   # C1
_CQT_MIDI_MAX = 108  # C8
_CQT_N_BINS = _CQT_MIDI_MAX - _CQT_MIDI_MIN  # 84 bins, 1 per semitone


class _SpectrogramWorker(QThread):
    """Background worker to compute CQT spectrogram (note-aligned)."""

    finished = pyqtSignal(np.ndarray, float)  # cqt_db, duration_sec

    def __init__(self, samples: np.ndarray, sample_rate: int, parent=None) -> None:
        super().__init__(parent)
        self.samples = samples
        self.sample_rate = sample_rate

    def run(self) -> None:
        C = np.abs(librosa.cqt(
            y=self.samples, sr=self.sample_rate,
            fmin=librosa.midi_to_hz(_CQT_MIDI_MIN),
            n_bins=_CQT_N_BINS, bins_per_octave=12,
        ))
        C_db = librosa.amplitude_to_db(C, ref=np.max)
        duration = len(self.samples) / self.sample_rate
        self.finished.emit(C_db, duration)


class WaveformView(QWidget):
    """Unified CQT spectrogram + MIDI overlay with playback and view modes."""

    selection_changed = pyqtSignal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._duration = 0.0
        self._worker: _SpectrogramWorker | None = None
        self._cmap = _build_inferno_cmap()

        # Audio state
        self._audio_samples: np.ndarray | None = None
        self._audio_sr: int = 22050

        # MIDI state
        self._midi_notes: list[NoteEvent] = []
        self._dynamic_velocity = False
        self._midi_items: list[pg.BarGraphItem] = []
        self._has_midi = False
        self._midi_audio: np.ndarray | None = None  # synthesized MIDI audio
        self._selection_start: float = 0.0  # offset for MIDI note times

        # Playback state
        self._playing = False
        self._paused = False
        self._play_source = "audio"  # "audio" or "midi"
        self._play_position = 0.0
        self._play_start_wall = 0.0   # wall-clock time when playback started
        self._play_start_offset = 0.0  # audio position at start of playback
        self._play_end_time = 0.0      # audio position where playback ends
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(30)
        self._playback_timer.timeout.connect(self._on_playback_tick)
        self._playback_line: pg.InfiniteLine | None = None

        # Note range for zoom (default: full CQT range)
        self._note_range_low = _CQT_MIDI_MIN
        self._note_range_high = _CQT_MIDI_MAX

        # Current view mode
        self._mode = MODE_FREQ

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Spectrogram/MIDI plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setLabel('left', 'Note')
        self.plot_widget.setMinimumHeight(250)
        self.plot_widget.hideButtons()
        self.plot_widget.setMouseEnabled(x=False, y=False)

        self.image_item = pg.ImageItem()
        self.image_item.setLookupTable(self._cmap.getLookupTable(nPts=256))
        self.image_item.setZValue(0)
        self.plot_widget.addItem(self.image_item)

        # Color bar legend
        self.color_bar = pg.ColorBarItem(
            values=(-80, 0),
            colorMap=self._cmap,
            label='dB',
            interactive=False,
            width=12,
        )
        self.color_bar.setImageItem(self.image_item, insert_in=self.plot_widget.plotItem)

        # Selection region
        self.region = pg.LinearRegionItem(
            values=[0, 1],
            orientation=pg.LinearRegionItem.Vertical,
            brush=pg.mkBrush(74, 158, 255, 40),
            pen=pg.mkPen(color=(74, 158, 255, 200), width=2),
            movable=True,
        )
        self.region.setZValue(5)
        self.plot_widget.addItem(self.region)

        # Playback position line
        self._playback_line = pg.InfiniteLine(
            pos=0, angle=90, movable=False,
            pen=pg.mkPen(color=(255, 255, 255, 200), width=2),
        )
        self._playback_line.setZValue(20)
        self._playback_line.setVisible(False)
        self.plot_widget.addItem(self._playback_line)

        layout.addWidget(self.plot_widget, 1)

        # Playback controls bar
        playback_bar = QHBoxLayout()
        playback_bar.setSpacing(6)

        self._play_btn = QPushButton("Play Audio")
        self._play_btn.setFixedWidth(100)
        self._play_btn.setEnabled(False)
        playback_bar.addWidget(self._play_btn)

        self._play_midi_btn = QPushButton("Play MIDI")
        self._play_midi_btn.setFixedWidth(90)
        self._play_midi_btn.setEnabled(False)
        playback_bar.addWidget(self._play_midi_btn)

        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setFixedWidth(70)
        self._pause_btn.setEnabled(False)
        playback_bar.addWidget(self._pause_btn)

        self._play_slider = QSlider(Qt.Orientation.Horizontal)
        self._play_slider.setRange(0, 1000)
        self._play_slider.setValue(0)
        self._play_slider.setEnabled(False)
        playback_bar.addWidget(self._play_slider, 1)

        self._play_time_label = QLabel("0:00 / 0:00")
        self._play_time_label.setMinimumWidth(80)
        playback_bar.addWidget(self._play_time_label)

        layout.addLayout(playback_bar)

        # Bottom bar: time fields + view mode selector
        bottom_bar = QHBoxLayout()

        bottom_bar.addWidget(QLabel("Start:"))
        self.start_edit = QLineEdit("0:00")
        self.start_edit.setMaximumWidth(70)
        self.start_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bottom_bar.addWidget(self.start_edit)

        bottom_bar.addSpacing(12)

        bottom_bar.addWidget(QLabel("End:"))
        self.end_edit = QLineEdit("0:00")
        self.end_edit.setMaximumWidth(70)
        self.end_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bottom_bar.addWidget(self.end_edit)

        bottom_bar.addSpacing(12)
        self.duration_label = QLabel("")
        bottom_bar.addWidget(self.duration_label)

        bottom_bar.addStretch()

        # View mode buttons
        bottom_bar.addWidget(QLabel("View:"))
        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)

        self._freq_btn = QPushButton("Frequencies")
        self._freq_btn.setCheckable(True)
        self._freq_btn.setChecked(True)
        self._freq_btn.setFixedWidth(110)
        self._btn_group.addButton(self._freq_btn, MODE_FREQ)
        bottom_bar.addWidget(self._freq_btn)

        self._overlap_btn = QPushButton("Overlap")
        self._overlap_btn.setCheckable(True)
        self._overlap_btn.setEnabled(False)
        self._overlap_btn.setFixedWidth(90)
        self._btn_group.addButton(self._overlap_btn, MODE_OVERLAP)
        bottom_bar.addWidget(self._overlap_btn)

        self._midi_btn = QPushButton("MIDI")
        self._midi_btn.setCheckable(True)
        self._midi_btn.setEnabled(False)
        self._midi_btn.setFixedWidth(70)
        self._btn_group.addButton(self._midi_btn, MODE_MIDI)
        bottom_bar.addWidget(self._midi_btn)

        layout.addLayout(bottom_bar)

        self._show_placeholder(True)

    def _connect_signals(self) -> None:
        self.region.sigRegionChangeFinished.connect(self._on_region_changed)
        self.start_edit.editingFinished.connect(self._on_time_edit_changed)
        self.end_edit.editingFinished.connect(self._on_time_edit_changed)
        self._btn_group.idClicked.connect(self._on_mode_changed)
        self._play_btn.clicked.connect(self._on_play_audio)
        self._play_midi_btn.clicked.connect(self._on_play_midi)
        self._pause_btn.clicked.connect(self._on_pause_resume)
        self._play_slider.sliderPressed.connect(self._on_slider_pressed)
        self._play_slider.sliderReleased.connect(self._on_slider_released)

    def _show_placeholder(self, show: bool) -> None:
        if show:
            self.image_item.clear()
            self.region.setVisible(False)
            self.plot_widget.setTitle("Load an audio file to view spectrogram")
        else:
            self.region.setVisible(True)
            self.plot_widget.setTitle("")

    # --- Adaptive note-label Y axis ---

    def _build_adaptive_ticks(self, y_min: int, y_max: int) -> list[list[tuple]]:
        """Build tick labels adapted to the visible note range span.

        Wide ranges show only C notes; narrow ranges show every semitone.
        """
        span = y_max - y_min
        ticks_major = []
        ticks_minor = []

        for midi_num in range(y_min, y_max + 1):
            name = midi_to_name(midi_num)
            pc = midi_num % 12  # pitch class

            if span > 48:
                # 4+ octaves: only C notes
                if pc == 0:
                    ticks_major.append((midi_num, name))
            elif span > 24:
                # 2-4 octaves: C as major, E and G as minor
                if pc == 0:
                    ticks_major.append((midi_num, name))
                elif pc in (4, 7):
                    ticks_minor.append((midi_num, name))
            elif span > 12:
                # 1-2 octaves: all natural notes
                if pc in (0, 2, 4, 5, 7, 9, 11):
                    ticks_major.append((midi_num, name))
            else:
                # < 1 octave: every semitone
                ticks_major.append((midi_num, name))

        return [ticks_major, ticks_minor]

    def _setup_note_ticks(self, y_min: int = _CQT_MIDI_MIN, y_max: int = _CQT_MIDI_MAX) -> None:
        """Set Y axis tick labels adapted to zoom level."""
        ticks = self._build_adaptive_ticks(y_min, y_max)
        left_axis = self.plot_widget.getAxis('left')
        left_axis.setTicks(ticks)

    # --- Audio loading ---

    def load_audio(self, samples: np.ndarray, sample_rate: int) -> None:
        """Load audio data and compute CQT spectrogram asynchronously."""
        self._stop_playback()
        self._audio_samples = samples
        self._audio_sr = sample_rate
        self._show_placeholder(False)
        self.plot_widget.setTitle("Computing spectrogram...")

        self._play_btn.setEnabled(True)
        self._play_slider.setEnabled(True)

        self._worker = _SpectrogramWorker(samples, sample_rate, parent=self)
        self._worker.finished.connect(self._on_spectrogram_ready)
        self._worker.start()

    def _on_spectrogram_ready(self, cqt_db: np.ndarray, duration: float) -> None:
        self._duration = duration
        self._cqt_db = cqt_db

        n_bins, n_frames = cqt_db.shape
        self.image_item.setImage(cqt_db.T, autoLevels=False, levels=[-80, 0])

        x_scale = duration / n_frames if n_frames > 0 else 1.0
        y_scale = 1.0  # 1 pixel = 1 semitone
        transform = pg.QtGui.QTransform()
        transform.translate(0, _CQT_MIDI_MIN)
        transform.scale(x_scale, y_scale)
        self.image_item.setTransform(transform)

        y_lo = self._note_range_low
        y_hi = self._note_range_high
        self.plot_widget.setXRange(0, duration, padding=0)
        self.plot_widget.setYRange(y_lo, y_hi, padding=0)
        self.plot_widget.setTitle("")
        self.plot_widget.setLabel('left', 'Note')
        self._setup_note_ticks(y_lo, y_hi)

        self.region.setRegion([0, duration])
        self._update_time_fields(0, duration)
        self.duration_label.setText(f"Duration: {self._format_time(duration)}")
        self._play_time_label.setText(f"0:00 / {self._format_time(duration)}")

        self._worker = None

    # --- MIDI overlay ---

    def display_midi(self, notes: list[NoteEvent], dynamic_velocity: bool = False,
                     selection_start: float = 0.0) -> None:
        """Store MIDI notes and enable overlay/midi-only modes."""
        self._midi_notes = notes
        self._dynamic_velocity = dynamic_velocity
        self._has_midi = len(notes) > 0
        self._selection_start = selection_start

        # Synthesize MIDI audio for playback
        self._midi_audio = None
        if self._has_midi:
            self._synthesize_midi()

        # Enable mode buttons
        self._overlap_btn.setEnabled(self._has_midi)
        self._midi_btn.setEnabled(self._has_midi)
        self._play_midi_btn.setEnabled(self._has_midi)

        if self._has_midi:
            self._overlap_btn.setChecked(True)
            self._mode = MODE_OVERLAP
            self._refresh_view()

    def _synthesize_midi(self) -> None:
        """Synthesize MIDI notes to audio using pretty_midi for playback."""
        try:
            import pretty_midi
            pm = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano
            for note in self._midi_notes:
                vel = int(max(1, min(127, note.amplitude * 127))) if self._dynamic_velocity else 100
                inst.notes.append(pretty_midi.Note(
                    velocity=vel,
                    pitch=note.pitch,
                    start=max(0.0, note.start_sec),
                    end=max(0.0, note.end_sec),
                ))
            pm.instruments.append(inst)
            try:
                self._midi_audio = pm.fluidsynth(fs=self._audio_sr).astype(np.float32)
            except Exception:
                self._midi_audio = pm.synthesize(fs=self._audio_sr).astype(np.float32)
            peak = np.max(np.abs(self._midi_audio))
            if peak > 0:
                self._midi_audio = self._midi_audio / peak * 0.8
        except Exception:
            self._midi_audio = None

    def clear_midi(self) -> None:
        """Clear MIDI data and revert to frequencies-only mode."""
        self._midi_notes = []
        self._has_midi = False
        self._dynamic_velocity = False
        self._midi_audio = None
        self._selection_start = 0.0

        self._overlap_btn.setEnabled(False)
        self._midi_btn.setEnabled(False)
        self._play_midi_btn.setEnabled(False)

        self._freq_btn.setChecked(True)
        self._mode = MODE_FREQ
        self._refresh_view()

    def _on_mode_changed(self, mode_id: int) -> None:
        self._mode = mode_id
        self._refresh_view()

    def _refresh_view(self) -> None:
        """Redraw based on current view mode."""
        for item in self._midi_items:
            self.plot_widget.removeItem(item)
        self._midi_items.clear()

        if self._mode == MODE_FREQ:
            self._show_spectrogram(True, opacity=1.0)
            self._setup_note_axes()
        elif self._mode == MODE_OVERLAP:
            self._show_spectrogram(True, opacity=0.55)
            self._setup_note_axes()
            self._draw_midi_overlay()
        elif self._mode == MODE_MIDI:
            self._show_spectrogram(False)
            self._draw_midi_piano_roll()

    def _show_spectrogram(self, visible: bool, opacity: float = 1.0) -> None:
        self.image_item.setVisible(visible)
        self.image_item.setOpacity(opacity)
        self.color_bar.setVisible(visible)

    def set_note_range_zoom(self, low: int, high: int) -> None:
        """Zoom the Y-axis to the given MIDI note range."""
        self._note_range_low = max(_CQT_MIDI_MIN, low)
        self._note_range_high = min(_CQT_MIDI_MAX, high)
        if self._duration > 0:
            self._refresh_view()

    def _setup_note_axes(self) -> None:
        """Set Y axis to note-labeled, zoomed to user's note range."""
        self.plot_widget.setLabel('left', 'Note')
        y_lo = self._note_range_low
        y_hi = self._note_range_high
        self._setup_note_ticks(y_lo, y_hi)
        if self._duration > 0:
            self.plot_widget.setYRange(y_lo, y_hi, padding=0)
            self.plot_widget.setXRange(0, self._duration, padding=0)

    def _draw_midi_overlay(self) -> None:
        """Draw MIDI notes on spectrogram — Y axis is MIDI pitch directly."""
        if not self._midi_notes:
            return

        offset = self._selection_start

        for note in self._midi_notes:
            x_pos = max(0.0, note.start_sec + offset)
            duration = note.end_sec - note.start_sec
            if duration <= 0:
                duration = 0.01

            rect = pg.BarGraphItem(
                x=[x_pos],
                y=[note.pitch - 0.4],
                width=[duration],
                height=[0.8],
                brush=self._note_brush(note),
                pen=pg.mkPen(color=(255, 255, 255, 180), width=1),
            )
            rect.setZValue(10)
            self.plot_widget.addItem(rect)
            self._midi_items.append(rect)

    def _draw_midi_piano_roll(self) -> None:
        """Draw MIDI-only piano roll with pitch on Y axis, zoomed to note range."""
        if not self._midi_notes:
            self.plot_widget.setTitle("No MIDI notes to display")
            return

        self.plot_widget.setTitle("")
        offset = self._selection_start

        for note in self._midi_notes:
            x_pos = max(0.0, note.start_sec + offset)
            duration = note.end_sec - note.start_sec
            if duration <= 0:
                duration = 0.01

            rect = pg.BarGraphItem(
                x=[x_pos],
                y=[note.pitch - 0.4],
                width=[duration],
                height=[0.8],
                brush=self._note_brush(note),
                pen=pg.mkPen(color=(255, 255, 255, 100), width=0.5),
            )
            rect.setZValue(10)
            self.plot_widget.addItem(rect)
            self._midi_items.append(rect)

        # Zoom Y to user's note range
        y_lo = self._note_range_low
        y_hi = self._note_range_high
        self.plot_widget.setYRange(y_lo - 1, y_hi + 1, padding=0)
        self.plot_widget.setLabel('left', 'Note')
        ticks = self._build_adaptive_ticks(y_lo - 1, y_hi + 1)
        left_axis = self.plot_widget.getAxis('left')
        left_axis.setTicks(ticks)

        if self._duration > 0:
            self.plot_widget.setXRange(0, self._duration, padding=0)

    def _note_brush(self, note: NoteEvent):
        if self._dynamic_velocity:
            amp = max(0.0, min(1.0, note.amplitude))
            r = int(amp * 255)
            g = int((1.0 - abs(amp - 0.5) * 2) * 150)
            b = int((1.0 - amp) * 255)
            return pg.mkBrush(r, g, b, 200)
        return pg.mkBrush(0, 255, 120, 200)

    # --- Playback ---

    def _on_play_audio(self) -> None:
        if self._audio_samples is None:
            return
        self._stop_playback()
        self._play_source = "audio"
        start = self.start_sec
        end = self.end_sec
        start_sample = int(start * self._audio_sr)
        end_sample = int(end * self._audio_sr)
        samples = self._audio_samples[start_sample:end_sample]
        if len(samples) == 0:
            return
        self._play_start_offset = start
        self._play_end_time = end
        self._play_start_wall = time.perf_counter()
        sd.play(samples, self._audio_sr)
        self._playing = True
        self._paused = False
        self._play_btn.setText("Playing...")
        self._pause_btn.setEnabled(True)
        self._pause_btn.setText("Pause")
        self._playback_line.setVisible(True)
        self._playback_timer.start()

    def _on_play_midi(self) -> None:
        if self._midi_audio is None:
            return
        self._stop_playback()
        self._play_source = "midi"
        midi_dur = len(self._midi_audio) / self._audio_sr
        self._play_start_offset = self._selection_start
        self._play_end_time = self._selection_start + midi_dur
        self._play_start_wall = time.perf_counter()
        sd.play(self._midi_audio, self._audio_sr)
        self._playing = True
        self._paused = False
        self._play_midi_btn.setText("Playing...")
        self._pause_btn.setEnabled(True)
        self._pause_btn.setText("Pause")
        self._playback_line.setVisible(True)
        self._playback_timer.start()

    def _on_pause_resume(self) -> None:
        """Toggle pause/resume. Keeps playback position on pause."""
        if self._paused:
            # Resume from paused position
            self._resume_from(self._play_position)
        elif self._playing:
            # Pause: stop audio but keep position
            sd.stop()
            self._playing = False
            self._paused = True
            self._playback_timer.stop()
            self._pause_btn.setText("Resume")
            self._play_btn.setText("Play Audio")
            self._play_midi_btn.setText("Play MIDI")
            # Keep playback line visible at current position

    def _resume_from(self, pos: float) -> None:
        """Resume playback from a given position."""
        if self._play_source == "audio" and self._audio_samples is not None:
            start_sample = int(pos * self._audio_sr)
            end_sample = int(self._play_end_time * self._audio_sr)
            samples = self._audio_samples[start_sample:end_sample]
            if len(samples) == 0:
                self._stop_playback()
                return
            sd.play(samples, self._audio_sr)
        elif self._play_source == "midi" and self._midi_audio is not None:
            midi_offset = pos - self._selection_start
            start_sample = int(max(0, midi_offset) * self._audio_sr)
            samples = self._midi_audio[start_sample:]
            if len(samples) == 0:
                self._stop_playback()
                return
            sd.play(samples, self._audio_sr)
        else:
            self._stop_playback()
            return

        self._play_start_offset = pos
        self._play_start_wall = time.perf_counter()
        self._playing = True
        self._paused = False
        self._pause_btn.setText("Pause")
        if self._play_source == "audio":
            self._play_btn.setText("Playing...")
        else:
            self._play_midi_btn.setText("Playing...")
        self._playback_line.setVisible(True)
        self._playback_timer.start()

    def _stop_playback(self) -> None:
        """Full stop — reset position and UI."""
        sd.stop()
        self._playing = False
        self._paused = False
        self._playback_timer.stop()
        self._playback_line.setVisible(False)
        self._play_btn.setText("Play Audio")
        self._play_midi_btn.setText("Play MIDI")
        self._pause_btn.setEnabled(False)
        self._pause_btn.setText("Pause")

    def _on_playback_tick(self) -> None:
        if not self._playing:
            self._stop_playback()
            return
        stream = sd.get_stream()
        if stream is None or not stream.active:
            self._stop_playback()
            return
        elapsed = time.perf_counter() - self._play_start_wall
        pos = self._play_start_offset + elapsed

        if pos >= self._play_end_time:
            self._stop_playback()
            return

        self._play_position = pos
        self._playback_line.setValue(pos)
        if self._duration > 0:
            slider_val = int(pos / self._duration * 1000)
            self._play_slider.blockSignals(True)
            self._play_slider.setValue(min(slider_val, 1000))
            self._play_slider.blockSignals(False)
        self._play_time_label.setText(
            f"{self._format_time(pos)} / {self._format_time(self._duration)}"
        )

    def _on_slider_pressed(self) -> None:
        """Pause timer updates while user drags the slider."""
        self._playback_timer.stop()

    def _on_slider_released(self) -> None:
        """Seek to the slider position. If was playing, restart from there."""
        if self._duration <= 0:
            return
        pos = self._play_slider.value() / 1000.0 * self._duration
        self._play_position = pos
        self._playback_line.setValue(pos)
        self._playback_line.setVisible(True)

        if self._playing:
            sd.stop()
            self._resume_from(pos)
        elif self._paused:
            # Update paused position without resuming
            pass

    # --- Time selection ---

    def _on_region_changed(self) -> None:
        start, end = self.region.getRegion()
        start = max(0.0, start)
        end = min(self._duration, end) if self._duration > 0 else end
        self._update_time_fields(start, end)
        self.selection_changed.emit(start, end)

    def _on_time_edit_changed(self) -> None:
        start = self._parse_time(self.start_edit.text())
        end = self._parse_time(self.end_edit.text())
        if start is not None and end is not None and start < end:
            self.region.blockSignals(True)
            self.region.setRegion([start, end])
            self.region.blockSignals(False)
            self.selection_changed.emit(start, end)

    def _update_time_fields(self, start: float, end: float) -> None:
        self.start_edit.blockSignals(True)
        self.end_edit.blockSignals(True)
        self.start_edit.setText(self._format_time(start))
        self.end_edit.setText(self._format_time(end))
        self.start_edit.blockSignals(False)
        self.end_edit.blockSignals(False)

    @property
    def start_sec(self) -> float:
        val = self._parse_time(self.start_edit.text())
        return val if val is not None else 0.0

    @property
    def end_sec(self) -> float:
        val = self._parse_time(self.end_edit.text())
        return val if val is not None else self._duration

    @property
    def duration(self) -> float:
        return self._duration

    @staticmethod
    def _format_time(seconds: float) -> str:
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes}:{secs:02d}"

    @staticmethod
    def _parse_time(text: str) -> float | None:
        try:
            parts = text.strip().split(':')
            if len(parts) == 2:
                minutes = int(parts[0])
                secs = int(parts[1])
                return minutes * 60.0 + secs
        except (ValueError, IndexError):
            pass
        return None

    def clear(self) -> None:
        """Clear everything — spectrogram and MIDI."""
        self._stop_playback()
        self._duration = 0.0
        self._audio_samples = None
        self._show_placeholder(True)
        self.start_edit.setText("0:00")
        self.end_edit.setText("0:00")
        self.duration_label.setText("")
        self._play_btn.setEnabled(False)
        self._play_slider.setEnabled(False)
        self.clear_midi()
