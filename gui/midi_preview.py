"""Read-only piano roll visualization of MIDI output."""

from PyQt6.QtWidgets import QVBoxLayout, QWidget
import pyqtgraph as pg
import numpy as np

from core import NoteEvent
from gui.controls_panel import midi_to_name


class MidiPreview(QWidget):
    """Piano roll display showing note events as colored rectangles."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('bottom', 'Time', units='s')
        self.plot_widget.setLabel('left', 'Pitch')
        self.plot_widget.setMinimumHeight(150)
        self.plot_widget.hideButtons()
        self.plot_widget.setMouseEnabled(x=True, y=False)
        self.plot_widget.setTitle("Generate MIDI to see piano roll preview")

        layout.addWidget(self.plot_widget)

    def display(self, notes: list[NoteEvent], dynamic_velocity: bool = False) -> None:
        """Render note events as a piano roll.

        Args:
            notes: List of NoteEvent to display.
            dynamic_velocity: If True, color-code notes by velocity/amplitude.
        """
        self.plot_widget.clear()

        if not notes:
            self.plot_widget.setTitle("No notes to display")
            return

        self.plot_widget.setTitle("")

        # Draw each note as a rectangle (BarGraphItem)
        pitches = [n.pitch for n in notes]
        min_pitch = min(pitches)
        max_pitch = max(pitches)

        for note in notes:
            duration = note.end_sec - note.start_sec
            if duration <= 0:
                duration = 0.01  # minimum visible width

            if dynamic_velocity:
                # Map amplitude to color: low = blue, high = red
                amp = max(0.0, min(1.0, note.amplitude))
                r = int(amp * 255)
                g = int((1.0 - abs(amp - 0.5) * 2) * 150)
                b = int((1.0 - amp) * 255)
                color = pg.mkBrush(r, g, b, 180)
            else:
                color = pg.mkBrush(74, 158, 255, 180)

            rect = pg.BarGraphItem(
                x=[note.start_sec],
                y=[note.pitch - 0.4],
                width=[duration],
                height=[0.8],
                brush=color,
                pen=pg.mkPen(color=(255, 255, 255, 100), width=0.5),
            )
            self.plot_widget.addItem(rect)

        # Configure axes
        self.plot_widget.setXRange(
            min(n.start_sec for n in notes) - 0.1,
            max(n.end_sec for n in notes) + 0.1,
        )
        self.plot_widget.setYRange(min_pitch - 1, max_pitch + 1)

        # Set pitch axis tick labels to note names
        pitch_ticks = [
            (p, midi_to_name(p))
            for p in range(min_pitch, max_pitch + 1)
        ]
        left_axis = self.plot_widget.getAxis('left')
        left_axis.setTicks([pitch_ticks])

    def clear(self) -> None:
        """Clear the piano roll display."""
        self.plot_widget.clear()
        self.plot_widget.setTitle("Generate MIDI to see piano roll preview")
