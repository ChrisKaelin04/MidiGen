"""Tests for core.midi_exporter — verify MIDI file output."""

from pathlib import Path

import pretty_midi
import pytest

from core import NoteEvent
from core.midi_exporter import build_midi, save, get_temp_path


def _make_notes() -> list[NoteEvent]:
    """Create a simple list of synthetic notes."""
    return [
        NoteEvent(pitch=60, start_sec=0.0, end_sec=0.125, amplitude=100 / 127),
        NoteEvent(pitch=64, start_sec=0.125, end_sec=0.25, amplitude=80 / 127),
        NoteEvent(pitch=67, start_sec=0.25, end_sec=0.375, amplitude=60 / 127),
    ]


class TestBuildMidi:
    def test_returns_pretty_midi(self) -> None:
        midi = build_midi(_make_notes(), bpm=120.0)
        assert isinstance(midi, pretty_midi.PrettyMIDI)

    def test_single_instrument(self) -> None:
        midi = build_midi(_make_notes(), bpm=120.0)
        assert len(midi.instruments) == 1
        assert midi.instruments[0].program == 0

    def test_correct_note_count(self) -> None:
        midi = build_midi(_make_notes(), bpm=120.0)
        assert len(midi.instruments[0].notes) == 3

    def test_pitches_match(self) -> None:
        midi = build_midi(_make_notes(), bpm=120.0)
        pitches = [n.pitch for n in midi.instruments[0].notes]
        assert pitches == [60, 64, 67]

    def test_velocities_mapped(self) -> None:
        midi = build_midi(_make_notes(), bpm=120.0)
        velocities = [n.velocity for n in midi.instruments[0].notes]
        assert velocities[0] == 100
        assert velocities[1] == 80
        assert velocities[2] == 60

    def test_timing(self) -> None:
        midi = build_midi(_make_notes(), bpm=120.0)
        notes = midi.instruments[0].notes
        assert abs(notes[0].start - 0.0) < 0.01
        assert abs(notes[1].start - 0.125) < 0.01
        assert abs(notes[2].start - 0.25) < 0.01

    def test_empty_notes(self) -> None:
        midi = build_midi([], bpm=120.0)
        assert len(midi.instruments[0].notes) == 0


class TestSave:
    def test_save_creates_file(self, tmp_path: Path) -> None:
        midi = build_midi(_make_notes(), bpm=120.0)
        out_path = tmp_path / "output.mid"
        result = save(midi, out_path)
        assert result == out_path
        assert out_path.exists()

    def test_saved_file_readable(self, tmp_path: Path) -> None:
        midi = build_midi(_make_notes(), bpm=120.0)
        out_path = tmp_path / "output.mid"
        save(midi, out_path)

        # Reload and verify
        loaded = pretty_midi.PrettyMIDI(str(out_path))
        assert len(loaded.instruments) == 1
        assert len(loaded.instruments[0].notes) == 3

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        midi = build_midi(_make_notes(), bpm=120.0)
        out_path = tmp_path / "subdir" / "deep" / "output.mid"
        save(midi, out_path)
        assert out_path.exists()

    def test_roundtrip_pitches(self, tmp_path: Path) -> None:
        notes = _make_notes()
        midi = build_midi(notes, bpm=120.0)
        out_path = tmp_path / "output.mid"
        save(midi, out_path)

        loaded = pretty_midi.PrettyMIDI(str(out_path))
        loaded_pitches = sorted(n.pitch for n in loaded.instruments[0].notes)
        expected_pitches = sorted(n.pitch for n in notes)
        assert loaded_pitches == expected_pitches


class TestTempPath:
    def test_temp_path_is_in_midigen_dir(self) -> None:
        path = get_temp_path()
        assert '.midigen' in str(path)
        assert path.name == 'output.mid'
