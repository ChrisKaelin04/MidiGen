"""Process NoteEvents: filter, quantize, and apply velocity.

Each function is a pure transformation on a list of NoteEvent,
independently testable with no side effects.
"""

from core import NoteEvent


# Harmonic series intervals in semitones from the fundamental.
# 2nd harmonic = octave (+12), 3rd = octave+5th (+19), 4th = 2 octaves (+24),
# 5th = 2 octaves+major 3rd (+28), 6th = 2 octaves+5th (+31)
_HARMONIC_INTERVALS = [12, 19, 24, 28, 31]


_GRID_DIVISORS = {
    "1/8":  2.0,
    "1/16": 4.0,
    "1/32": 8.0,
    "1/8T": 3.0,
    "1/16T": 6.0,
}


def _grid_duration(bpm: float, grid: str = "1/16") -> float:
    """Return the duration of one grid unit in seconds at the given BPM."""
    beat_sec = 60.0 / bpm
    divisor = _GRID_DIVISORS.get(grid, 4.0)
    return beat_sec / divisor


def _sixteenth_duration(bpm: float) -> float:
    """Return the duration of one 1/16th note in seconds at the given BPM."""
    return _grid_duration(bpm, "1/16")


def filter_note_range(notes: list[NoteEvent], low: int, high: int) -> list[NoteEvent]:
    """Drop notes outside the MIDI note range [low, high] inclusive.

    Args:
        notes: Input note events.
        low: Lowest allowed MIDI note number.
        high: Highest allowed MIDI note number.

    Returns:
        Filtered list keeping only notes where low <= pitch <= high.
    """
    return [n for n in notes if low <= n.pitch <= high]


def filter_ghost_notes(notes: list[NoteEvent], bpm: float) -> list[NoteEvent]:
    """Drop notes shorter than one 1/16th note duration.

    This runs BEFORE quantization so it operates on raw detected durations.

    Args:
        notes: Input note events.
        bpm: Current BPM for calculating 1/16th note length.

    Returns:
        Filtered list with ghost notes removed.
    """
    min_duration = _sixteenth_duration(bpm)
    return [n for n in notes if (n.end_sec - n.start_sec) >= min_duration]


def quantize_onsets(notes: list[NoteEvent], bpm: float, grid: str = "1/16") -> list[NoteEvent]:
    """Snap note start times to the nearest grid position.

    Args:
        notes: Input note events.
        bpm: Current BPM for calculating grid positions.
        grid: Grid resolution ("1/8", "1/16", "1/32", "1/8T", "1/16T").

    Returns:
        New list with quantized start times (end times adjusted by same delta).
    """
    step = _grid_duration(bpm, grid)
    result: list[NoteEvent] = []
    for n in notes:
        grid_index = round(n.start_sec / step)
        quantized_start = grid_index * step
        delta = quantized_start - n.start_sec
        result.append(NoteEvent(
            pitch=n.pitch,
            start_sec=quantized_start,
            end_sec=n.end_sec + delta,
            amplitude=n.amplitude,
        ))
    return result


def set_durations(notes: list[NoteEvent], bpm: float, grid: str = "1/16") -> list[NoteEvent]:
    """Set all note durations to exactly one grid unit.

    Args:
        notes: Input note events.
        bpm: Current BPM for calculating grid length.
        grid: Grid resolution.

    Returns:
        New list with uniform durations.
    """
    dur = _grid_duration(bpm, grid)
    return [
        NoteEvent(
            pitch=n.pitch,
            start_sec=n.start_sec,
            end_sec=n.start_sec + dur,
            amplitude=n.amplitude,
        )
        for n in notes
    ]


def filter_harmonics(notes: list[NoteEvent], onset_tol: float = 0.05) -> list[NoteEvent]:
    """Remove notes that are likely overtones of louder fundamental notes.

    For each pair of simultaneous notes (onsets within tolerance), if the
    higher note's pitch matches a harmonic interval of the lower note AND
    the lower note has greater or equal amplitude, the higher note is
    removed as a probable overtone.

    Args:
        notes: Input note events.
        onset_tol: Maximum onset time difference to consider notes simultaneous.

    Returns:
        Filtered list with likely overtone notes removed.
    """
    if len(notes) <= 1:
        return list(notes)

    # Sort by onset then pitch for consistent processing
    sorted_notes = sorted(notes, key=lambda n: (n.start_sec, n.pitch))
    is_harmonic = [False] * len(sorted_notes)

    for i, note_i in enumerate(sorted_notes):
        if is_harmonic[i]:
            continue
        for j in range(i + 1, len(sorted_notes)):
            note_j = sorted_notes[j]
            # Stop searching once onsets are too far apart
            if note_j.start_sec - note_i.start_sec > onset_tol:
                break
            if is_harmonic[j]:
                continue

            interval = note_j.pitch - note_i.pitch
            if interval in _HARMONIC_INTERVALS:
                # Higher note matches a harmonic of the lower note.
                # Only flag as overtone if the fundamental is significantly
                # louder — real chord voicings often include octaves/5ths
                # at similar amplitude, so we need a strong ratio.
                if note_i.amplitude >= note_j.amplitude * 1.5:
                    is_harmonic[j] = True

    return [n for n, flagged in zip(sorted_notes, is_harmonic) if not flagged]


def snap_durations(notes: list[NoteEvent], bpm: float, grid: str = "1/16") -> list[NoteEvent]:
    """Snap note end times to the nearest grid position.

    Preserves detected note lengths but rounds them to clean grid values.
    Ensures a minimum duration of one grid unit.

    Args:
        notes: Input note events (start times should already be quantized).
        bpm: Current BPM for calculating grid positions.
        grid: Grid resolution.

    Returns:
        New list with end times snapped to grid, minimum one grid unit duration.
    """
    step = _grid_duration(bpm, grid)
    result: list[NoteEvent] = []
    for n in notes:
        raw_duration = n.end_sec - n.start_sec
        grid_count = max(1, round(raw_duration / step))
        snapped_end = n.start_sec + grid_count * step
        result.append(NoteEvent(
            pitch=n.pitch,
            start_sec=n.start_sec,
            end_sec=snapped_end,
            amplitude=n.amplitude,
        ))
    return result


def apply_velocity(notes: list[NoteEvent], dynamic: bool,
                   velocity_curve: float = 1.0) -> list[NoteEvent]:
    """Map amplitude to MIDI velocity with optional curve shaping.

    In dynamic mode, amplitude is shaped by the velocity curve exponent:
      shaped = amplitude ^ velocity_curve
    - curve=1.0: linear (default)
    - curve<1.0: boosts quiet notes (e.g. 0.5 = square root)
    - curve>1.0: compresses quiet notes (e.g. 2.0 = squared)

    In fixed mode, all velocities are set to 100 (amplitude set to 100/127).

    Args:
        notes: Input note events.
        dynamic: If True, use amplitude-based velocity; if False, use fixed velocity.
        velocity_curve: Exponent for shaping the amplitude-to-velocity mapping.

    Returns:
        New list with amplitude values set for velocity mapping.
    """
    if dynamic:
        result = []
        for n in notes:
            clamped = max(0.0, min(1.0, n.amplitude))
            # Apply curve shaping
            if velocity_curve != 1.0 and clamped > 0:
                clamped = clamped ** velocity_curve
            # Ensure at least 1/127 so velocity is never 0
            if clamped < 1.0 / 127.0:
                clamped = 1.0 / 127.0
            result.append(NoteEvent(
                pitch=n.pitch,
                start_sec=n.start_sec,
                end_sec=n.end_sec,
                amplitude=clamped,
            ))
        return result
    else:
        fixed_amp = 100.0 / 127.0
        return [
            NoteEvent(
                pitch=n.pitch,
                start_sec=n.start_sec,
                end_sec=n.end_sec,
                amplitude=fixed_amp,
            )
            for n in notes
        ]


def process(notes: list[NoteEvent], bpm: float, note_low: int, note_high: int,
            do_filter_ghosts: bool, dynamic_velocity: bool,
            preserve_durations: bool = True,
            do_filter_harmonics: bool = True,
            quantize_grid: str = "1/16",
            velocity_curve: float = 1.0) -> list[NoteEvent]:
    """Run the full processing pipeline on a list of note events.

    Steps executed in order:
    1. Filter harmonics / overtones (if enabled)
    2. Filter by note range
    3. Filter ghost notes (if enabled)
    4. Quantize onsets to grid
    5. Duration handling:
       - preserve_durations=True: snap end times to grid (keeps detected lengths)
       - preserve_durations=False: force all durations to one grid unit
    6. Apply velocity mapping with optional curve

    Args:
        notes: Raw note events from transcription.
        bpm: BPM for quantization grid.
        note_low: Lowest MIDI note to keep.
        note_high: Highest MIDI note to keep.
        do_filter_ghosts: Whether to remove ghost notes.
        dynamic_velocity: Whether to use dynamic or fixed velocity.
        preserve_durations: Whether to keep detected note lengths.
        do_filter_harmonics: Whether to remove likely overtone notes.
        quantize_grid: Grid resolution ("1/8", "1/16", "1/32", "1/8T", "1/16T").
        velocity_curve: Exponent for velocity shaping (1.0=linear).

    Returns:
        Processed list of NoteEvent ready for MIDI export.
    """
    result = notes
    if do_filter_harmonics:
        result = filter_harmonics(result)
    result = filter_note_range(result, note_low, note_high)
    if do_filter_ghosts:
        result = filter_ghost_notes(result, bpm)
    result = quantize_onsets(result, bpm, quantize_grid)
    if preserve_durations:
        result = snap_durations(result, bpm, quantize_grid)
    else:
        result = set_durations(result, bpm, quantize_grid)
    result = apply_velocity(result, dynamic_velocity, velocity_curve)
    return result
