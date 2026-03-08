"""Build a PrettyMIDI object from NoteEvents and export to .mid files."""

from pathlib import Path

import pretty_midi

from core import NoteEvent


_PPQN = 960
_CONFIG_DIR = Path.home() / '.midigen'
_TEMP_DIR = _CONFIG_DIR / 'tmp'


def build_midi(notes: list[NoteEvent], bpm: float) -> pretty_midi.PrettyMIDI:
    """Build a single-track PrettyMIDI object from processed note events.

    Args:
        notes: Processed NoteEvent list with final pitches, timings, and amplitudes.
        bpm: BPM used for the MIDI file tempo.

    Returns:
        A PrettyMIDI object with one piano track containing all notes.
    """
    midi = pretty_midi.PrettyMIDI(initial_tempo=bpm, resolution=_PPQN)
    instrument = pretty_midi.Instrument(program=0, name='Piano')

    for note in notes:
        velocity = max(1, min(127, int(round(note.amplitude * 127))))
        midi_note = pretty_midi.Note(
            velocity=velocity,
            pitch=note.pitch,
            start=note.start_sec,
            end=note.end_sec,
        )
        instrument.notes.append(midi_note)

    midi.instruments.append(instrument)
    return midi


def save(midi: pretty_midi.PrettyMIDI, path: Path) -> Path:
    """Save a PrettyMIDI object to a .mid file.

    Args:
        midi: The PrettyMIDI object to save.
        path: Destination file path.

    Returns:
        The path the file was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
    return path


def get_temp_path() -> Path:
    """Get the temporary output path for drag-drop export.

    Creates the temp directory if it doesn't exist.

    Returns:
        Path to ~/.midigen/tmp/output.mid
    """
    _TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return _TEMP_DIR / 'output.mid'


def save_temp(midi: pretty_midi.PrettyMIDI) -> Path:
    """Save a PrettyMIDI object to the temp location for drag-drop.

    Args:
        midi: The PrettyMIDI object to save.

    Returns:
        Path to the saved temp file.
    """
    return save(midi, get_temp_path())
