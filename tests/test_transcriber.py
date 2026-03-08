"""Tests for core.transcriber — mock basic-pitch for unit isolation."""

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from core import AudioData, NoteEvent
from core.transcriber import transcribe


@pytest.fixture
def dummy_audio() -> AudioData:
    """Create a minimal AudioData object for testing."""
    from pathlib import Path
    return AudioData(
        samples=np.zeros(22050, dtype=np.float32),
        sample_rate=22050,
        duration_sec=1.0,
        file_path=Path("test.wav"),
    )


def _make_mock_note_events() -> list[tuple]:
    """Create mock basic-pitch note event tuples.

    Format: (start_time, end_time, pitch, amplitude)
    """
    return [
        (0.0, 0.5, 60, 0.9),   # C4, high confidence
        (0.5, 1.0, 64, 0.7),   # E4, medium confidence
        (1.0, 1.5, 67, 0.3),   # G4, low confidence
        (1.5, 2.0, 72, 0.1),   # C5, very low confidence
    ]


class TestTranscribe:
    @patch('basic_pitch.inference.predict')
    def test_returns_note_events(self, mock_predict: MagicMock, dummy_audio: AudioData) -> None:
        mock_predict.return_value = (None, None, _make_mock_note_events())

        result = transcribe(dummy_audio, confidence_threshold=0.0)

        assert len(result) == 4
        assert all(isinstance(n, NoteEvent) for n in result)

    @patch('basic_pitch.inference.predict')
    def test_confidence_filtering(self, mock_predict: MagicMock, dummy_audio: AudioData) -> None:
        mock_predict.return_value = (None, None, _make_mock_note_events())

        result = transcribe(dummy_audio, confidence_threshold=0.5)

        # Only notes with amplitude >= 0.5 should remain
        assert len(result) == 2
        assert result[0].pitch == 60
        assert result[1].pitch == 64

    @patch('basic_pitch.inference.predict')
    def test_strict_confidence(self, mock_predict: MagicMock, dummy_audio: AudioData) -> None:
        mock_predict.return_value = (None, None, _make_mock_note_events())

        result = transcribe(dummy_audio, confidence_threshold=0.8)

        assert len(result) == 1
        assert result[0].pitch == 60

    @patch('basic_pitch.inference.predict')
    def test_note_event_schema(self, mock_predict: MagicMock, dummy_audio: AudioData) -> None:
        mock_predict.return_value = (None, None, _make_mock_note_events())

        result = transcribe(dummy_audio, confidence_threshold=0.0)

        note = result[0]
        assert note.pitch == 60
        assert note.start_sec == 0.0
        assert note.end_sec == 0.5
        assert note.amplitude == 0.9

    @patch('basic_pitch.inference.predict')
    def test_empty_output(self, mock_predict: MagicMock, dummy_audio: AudioData) -> None:
        mock_predict.return_value = (None, None, [])

        result = transcribe(dummy_audio, confidence_threshold=0.5)

        assert result == []

    @patch('basic_pitch.inference.predict')
    def test_does_not_filter_by_note_range(self, mock_predict: MagicMock, dummy_audio: AudioData) -> None:
        """Transcriber should NOT filter by note range — that's midi_processor's job."""
        wide_range_notes = [
            (0.0, 0.5, 20, 0.9),   # Very low note
            (0.5, 1.0, 110, 0.9),  # Very high note
        ]
        mock_predict.return_value = (None, None, wide_range_notes)

        result = transcribe(dummy_audio, confidence_threshold=0.0)

        assert len(result) == 2
        assert result[0].pitch == 20
        assert result[1].pitch == 110
