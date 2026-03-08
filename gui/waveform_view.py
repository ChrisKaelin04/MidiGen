"""Spectrogram heatmap display with draggable selection region."""

from PyQt6.QtCore import Qt, pyqtSignal, QThread
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)
import numpy as np
import pyqtgraph as pg
import librosa


class _SpectrogramWorker(QThread):
    """Background worker to compute spectrogram data without blocking the UI."""

    finished = pyqtSignal(np.ndarray, float)  # spectrogram_db, duration_sec

    def __init__(self, samples: np.ndarray, sample_rate: int, parent=None) -> None:
        super().__init__(parent)
        self.samples = samples
        self.sample_rate = sample_rate

    def run(self) -> None:
        S = librosa.feature.melspectrogram(
            y=self.samples, sr=self.sample_rate, n_mels=128, fmax=8000
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        duration = len(self.samples) / self.sample_rate
        self.finished.emit(S_db, duration)


class WaveformView(QWidget):
    """Spectrogram heatmap with a draggable selection region for start/end times."""

    selection_changed = pyqtSignal(float, float)  # start_sec, end_sec

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._duration = 0.0
        self._worker: _SpectrogramWorker | None = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Spectrogram plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setLabel('left', 'Frequency Bin')
        self.plot_widget.setMinimumHeight(150)
        self.plot_widget.hideButtons()
        self.plot_widget.setMouseEnabled(x=False, y=False)

        self.image_item = pg.ImageItem()
        self.plot_widget.addItem(self.image_item)

        # Selection region (LinearRegionItem for draggable start/end handles)
        self.region = pg.LinearRegionItem(
            values=[0, 1],
            orientation=pg.LinearRegionItem.Vertical,
            brush=pg.mkBrush(74, 158, 255, 50),
            pen=pg.mkPen(color=(74, 158, 255, 200), width=2),
            movable=True,
        )
        self.plot_widget.addItem(self.region)

        layout.addWidget(self.plot_widget)

        # Start/End time fields
        time_layout = QHBoxLayout()

        time_layout.addWidget(QLabel("Start:"))
        self.start_edit = QLineEdit("0:00")
        self.start_edit.setMaximumWidth(70)
        self.start_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        time_layout.addWidget(self.start_edit)

        time_layout.addSpacing(16)

        time_layout.addWidget(QLabel("End:"))
        self.end_edit = QLineEdit("0:00")
        self.end_edit.setMaximumWidth(70)
        self.end_edit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        time_layout.addWidget(self.end_edit)

        time_layout.addStretch()

        self.duration_label = QLabel("")
        time_layout.addWidget(self.duration_label)

        layout.addLayout(time_layout)

        # Placeholder when no audio is loaded
        self._show_placeholder(True)

    def _connect_signals(self) -> None:
        self.region.sigRegionChangeFinished.connect(self._on_region_changed)
        self.start_edit.editingFinished.connect(self._on_time_edit_changed)
        self.end_edit.editingFinished.connect(self._on_time_edit_changed)

    def _show_placeholder(self, show: bool) -> None:
        if show:
            self.image_item.clear()
            self.region.setVisible(False)
            self.plot_widget.setTitle("Load an audio file to view spectrogram")
        else:
            self.region.setVisible(True)
            self.plot_widget.setTitle("")

    def load_audio(self, samples: np.ndarray, sample_rate: int) -> None:
        """Load audio data and compute spectrogram asynchronously.

        Args:
            samples: Mono float32 audio samples.
            sample_rate: Sample rate in Hz.
        """
        self._show_placeholder(False)
        self.plot_widget.setTitle("Computing spectrogram...")

        self._worker = _SpectrogramWorker(samples, sample_rate, parent=self)
        self._worker.finished.connect(self._on_spectrogram_ready)
        self._worker.start()

    def _on_spectrogram_ready(self, spectrogram_db: np.ndarray, duration: float) -> None:
        self._duration = duration

        # Set image with correct scaling: x = time, y = mel bins
        n_mels, n_frames = spectrogram_db.shape
        self.image_item.setImage(spectrogram_db.T, autoLevels=True)

        # Scale so x-axis represents seconds
        x_scale = duration / n_frames if n_frames > 0 else 1.0
        y_scale = 1.0
        self.image_item.setTransform(
            pg.QtGui.QTransform.fromScale(x_scale, y_scale)
        )

        self.plot_widget.setXRange(0, duration)
        self.plot_widget.setYRange(0, n_mels)
        self.plot_widget.setTitle("")

        # Set region to full duration by default
        self.region.setRegion([0, duration])
        self._update_time_fields(0, duration)
        self.duration_label.setText(f"Duration: {self._format_time(duration)}")

        self._worker = None

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
        """Format seconds as M:SS."""
        minutes = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{minutes}:{secs:02d}"

    @staticmethod
    def _parse_time(text: str) -> float | None:
        """Parse M:SS or MM:SS format to seconds."""
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
        """Clear the spectrogram display."""
        self._duration = 0.0
        self._show_placeholder(True)
        self.start_edit.setText("0:00")
        self.end_edit.setText("0:00")
        self.duration_label.setText("")
