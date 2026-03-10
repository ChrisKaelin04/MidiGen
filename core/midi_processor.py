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


def merge_fragments(notes: list[NoteEvent], gap_tol: float = 0.05,
                    reattack_ratio: float = 0.0) -> list[NoteEvent]:
    """Merge adjacent note fragments with the same pitch into single notes.

    When the ML model splits a sustained note into multiple short detections,
    this stitches them back together.  Two notes are merged when they share
    the same pitch, the gap between the end of one and the start of the
    next is within *gap_tol* seconds, AND the next note does not look like
    a fresh attack (re-attack detection).

    A re-attack is detected when the next note's amplitude is significantly
    higher than the current note's amplitude (indicating a new key strike
    rather than a decaying continuation).  Set *reattack_ratio* to 0.0 to
    disable re-attack detection (merge everything within gap_tol).

    Args:
        notes: Input note events (raw, pre-quantization).
        gap_tol: Maximum gap in seconds between fragments to merge.
        reattack_ratio: If > 0, skip merging when next amplitude exceeds
            current amplitude by this ratio (e.g. 1.5 means next note must
            be < 1.5x current amplitude to merge).  0 disables the check.

    Returns:
        New list with fragments merged into longer notes.
    """
    if len(notes) <= 1:
        return list(notes)

    sorted_notes = sorted(notes, key=lambda n: (n.pitch, n.start_sec))
    result: list[NoteEvent] = []
    i = 0
    while i < len(sorted_notes):
        current = sorted_notes[i]
        # Absorb subsequent fragments of the same pitch
        while (i + 1 < len(sorted_notes)
               and sorted_notes[i + 1].pitch == current.pitch
               and sorted_notes[i + 1].start_sec - current.end_sec <= gap_tol):
            nxt = sorted_notes[i + 1]
            # Re-attack detection: if next note is much louder, it's a new
            # strike, not a fragment continuation
            if (reattack_ratio > 0
                    and current.amplitude > 0
                    and nxt.amplitude / current.amplitude >= reattack_ratio):
                break
            i += 1
            current = NoteEvent(
                pitch=current.pitch,
                start_sec=current.start_sec,
                end_sec=nxt.end_sec,
                amplitude=max(current.amplitude, nxt.amplitude),
            )
        result.append(current)
        i += 1
    return result


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


def _group_by_onset(notes: list[NoteEvent], onset_tol: float = 0.05) -> list[list[NoteEvent]]:
    """Group notes into onset groups where all onsets are within tolerance."""
    if not notes:
        return []
    sorted_notes = sorted(notes, key=lambda n: n.start_sec)
    groups: list[list[NoteEvent]] = [[sorted_notes[0]]]
    for note in sorted_notes[1:]:
        if note.start_sec - groups[-1][0].start_sec <= onset_tol:
            groups[-1].append(note)
        else:
            groups.append([note])
    return groups


def filter_harmonics_adaptive(notes: list[NoteEvent], onset_tol: float = 0.05,
                               chord_threshold: int = 3) -> list[NoteEvent]:
    """Context-aware harmonic filter: filters melody lines, preserves chords.

    Groups notes by onset time. For groups with few notes (melody/bass lines),
    applies the standard harmonic filter to remove overtones. For groups with
    many simultaneous notes (chords), keeps all notes since harmonic intervals
    are intentional voicing, not overtones.

    Args:
        notes: Input note events.
        onset_tol: Maximum onset time difference to group notes together.
        chord_threshold: Groups with more notes than this are treated as
            chords and skip harmonic filtering.

    Returns:
        Filtered list with overtones removed from melody lines only.
    """
    if len(notes) <= 1:
        return list(notes)

    groups = _group_by_onset(notes, onset_tol)
    result: list[NoteEvent] = []
    for group in groups:
        if len(group) > chord_threshold:
            result.extend(group)  # chord — keep all
        else:
            result.extend(filter_harmonics(group, onset_tol))  # melody — filter
    return result


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


def _find_regular_chains(
    groups: list[list[NoteEvent]],
    group_onsets: list[float],
    interval: float,
    tolerance: float,
    min_reps: int,
    used: set[int] | None = None,
) -> list[list[list[NoteEvent]]]:
    """Find sequences of onset groups spaced at a regular interval.

    Walks through groups left-to-right, building chains where each next
    group's onset is within *tolerance* of the expected position.  A chain
    is only kept if it has at least *min_reps* groups.  Each group can
    belong to at most one chain.

    Args:
        used: Optional set of group indices already claimed by other chains.
            Newly claimed indices are added in-place.

    Returns:
        List of chains, where each chain is a list of onset groups.
    """
    chains: list[list[list[NoteEvent]]] = []
    if used is None:
        used = set()

    for i in range(len(groups)):
        if i in used:
            continue
        chain = [groups[i]]
        chain_idx = [i]
        expected = group_onsets[i] + interval

        for j in range(i + 1, len(groups)):
            if j in used:
                continue
            if abs(group_onsets[j] - expected) <= tolerance:
                chain.append(groups[j])
                chain_idx.append(j)
                expected = group_onsets[j] + interval
            elif group_onsets[j] > expected + tolerance:
                # Past the window — stop searching for this chain
                break

        if len(chain) >= min_reps:
            chains.append(chain)
            used.update(chain_idx)

    return chains


def fill_repeated_patterns(
    notes: list[NoteEvent],
    bpm: float,
    quantize_grid: str = "1/16",
    onset_tol: float = 0.01,
    min_reps: int = 4,
    fill_threshold: float = 0.75,
) -> list[NoteEvent]:
    """Fill missing notes in repeating chord/note patterns.

    After quantization, onset groups that repeat at regular intervals are
    identified.  Within each repeating chain, a *consensus* pitch set is
    built from pitches that appear in at least *fill_threshold* of the
    repetitions.  Any repetition missing a consensus pitch gets a new note
    inserted with the group's average amplitude and duration.

    This is especially useful for EDM where chord progressions repeat in
    4/8/16-bar loops and the ML model occasionally drops a note.

    Args:
        notes: Quantized note events.
        bpm: Beats per minute (used to determine the grid step for
            interval detection).
        quantize_grid: Grid resolution that was used for quantization.
        onset_tol: Maximum onset time difference (seconds) to group
            notes together (should be small since notes are quantized).
        min_reps: Minimum number of repetitions to consider a pattern.
        fill_threshold: Fraction of repetitions a pitch must appear in
            to be part of the consensus (0.0–1.0).

    Returns:
        Original notes plus any newly filled notes.
    """
    if not notes or bpm <= 0:
        return list(notes)

    grid_step = _grid_duration(bpm, quantize_grid)

    # Group quantized notes by onset
    groups = _group_by_onset(notes, onset_tol)
    if len(groups) < min_reps:
        return list(notes)

    group_onsets = [g[0].start_sec for g in groups]

    # Find the dominant interval between consecutive groups
    intervals: list[float] = []
    for i in range(len(group_onsets) - 1):
        raw = group_onsets[i + 1] - group_onsets[i]
        # Round to nearest grid step
        rounded = round(raw / grid_step) * grid_step
        if rounded > 0:
            intervals.append(rounded)

    if not intervals:
        return list(notes)

    # Find chains at all observed intervals (not just the dominant one).
    # This handles pieces with multiple pattern sections at different rates.
    # A shared `used` set prevents a group from appearing in multiple chains.
    from collections import Counter
    interval_counts = Counter(intervals)
    tolerance = grid_step * 0.5
    chains: list[list[list[NoteEvent]]] = []
    used: set[int] = set()
    # Try each distinct interval, most common first
    for iv, _count in interval_counts.most_common():
        if iv <= 0:
            continue
        new_chains = _find_regular_chains(
            groups, group_onsets, iv, tolerance, min_reps, used,
        )
        chains.extend(new_chains)

    # For each chain, build a consensus pitch set and fill gaps
    added: list[NoteEvent] = []
    for chain in chains:
        # Count how often each pitch appears across repetitions
        pitch_counts: dict[int, int] = {}
        for group in chain:
            for note in group:
                pitch_counts[note.pitch] = pitch_counts.get(note.pitch, 0) + 1

        n_reps = len(chain)
        consensus = {
            p for p, c in pitch_counts.items()
            if c / n_reps >= fill_threshold
        }

        # Fill missing consensus pitches
        for group in chain:
            group_pitches = {n.pitch for n in group}
            missing = consensus - group_pitches
            if not missing:
                continue

            # Use group's average amplitude and duration for filled notes
            avg_amp = sum(n.amplitude for n in group) / len(group)
            avg_dur = sum(n.end_sec - n.start_sec for n in group) / len(group)
            onset = group[0].start_sec

            for pitch in missing:
                added.append(NoteEvent(
                    pitch=pitch,
                    start_sec=onset,
                    end_sec=onset + avg_dur,
                    amplitude=avg_amp,
                ))

    return list(notes) + added


def apply_timing_offset(notes: list[NoteEvent], offset_sec: float) -> list[NoteEvent]:
    """Shift all note times by a fixed offset.

    Positive values shift notes later (to the right). Notes that would
    start before 0.0 are clamped to 0.0.

    Args:
        notes: Input note events.
        offset_sec: Time offset in seconds. Positive = later, negative = earlier.

    Returns:
        New list with shifted start and end times.
    """
    if offset_sec == 0.0:
        return list(notes)
    result: list[NoteEvent] = []
    for n in notes:
        new_start = max(0.0, n.start_sec + offset_sec)
        new_end = max(new_start, n.end_sec + offset_sec)
        result.append(NoteEvent(
            pitch=n.pitch,
            start_sec=new_start,
            end_sec=new_end,
            amplitude=n.amplitude,
        ))
    return result


def process(notes: list[NoteEvent], bpm: float, note_low: int, note_high: int,
            do_filter_ghosts: bool, dynamic_velocity: bool,
            preserve_durations: bool = True,
            do_filter_harmonics: bool = True,
            harmonic_filter_mode: str = "adaptive",
            quantize_grid: str = "1/16",
            velocity_curve: float = 1.0,
            do_merge_fragments: bool = False,
            fragment_gap_tol: float = 0.05,
            fragment_reattack_ratio: float = 0.0,
            timing_offset_grid: float = 0.0,
            do_fill_patterns: bool = False,
            pattern_min_reps: int = 4,
            pattern_fill_threshold: float = 0.75) -> list[NoteEvent]:
    """Run the full processing pipeline on a list of note events.

    Steps executed in order:
    1. Apply timing offset (shift all notes in time)
    2. Merge fragmented note detections (if enabled)
    3. Filter harmonics / overtones (if enabled)
    4. Filter by note range
    5. Filter ghost notes (if enabled)
    6. Quantize onsets to grid
    7. Duration handling:
       - preserve_durations=True: snap end times to grid (keeps detected lengths)
       - preserve_durations=False: force all durations to one grid unit
    8. Fill repeated patterns (if enabled) — after quantization for clean intervals
    9. Apply velocity mapping with optional curve

    Args:
        notes: Raw note events from transcription.
        bpm: BPM for quantization grid.
        note_low: Lowest MIDI note to keep.
        note_high: Highest MIDI note to keep.
        do_filter_ghosts: Whether to remove ghost notes.
        dynamic_velocity: Whether to use dynamic or fixed velocity.
        preserve_durations: Whether to keep detected note lengths.
        do_filter_harmonics: Whether to remove likely overtone notes (legacy).
        harmonic_filter_mode: "off", "on" (always filter), or "adaptive"
            (filter melody lines, preserve chords). Takes priority over
            do_filter_harmonics when set to non-default.
        quantize_grid: Grid resolution ("1/8", "1/16", "1/32", "1/8T", "1/16T").
        velocity_curve: Exponent for velocity shaping (1.0=linear).
        do_merge_fragments: Whether to stitch adjacent same-pitch fragments.
        fragment_gap_tol: Maximum gap (seconds) between fragments to merge.
        fragment_reattack_ratio: Amplitude ratio to detect re-attacks (0=disabled).
        timing_offset_grid: Timing shift in grid steps (positive = later).
            Computed to seconds using bpm and quantize_grid.
        do_fill_patterns: Whether to fill missing notes in repeating patterns.
        pattern_min_reps: Minimum repetitions to consider a pattern.
        pattern_fill_threshold: Fraction of reps a pitch must appear in to fill.

    Returns:
        Processed list of NoteEvent ready for MIDI export.
    """
    result = notes
    if timing_offset_grid != 0.0:
        offset_sec = timing_offset_grid * _grid_duration(bpm, quantize_grid)
        result = apply_timing_offset(result, offset_sec)
    if do_merge_fragments:
        result = merge_fragments(result, gap_tol=fragment_gap_tol,
                                 reattack_ratio=fragment_reattack_ratio)
    # Harmonic filtering: mode takes priority, fall back to bool
    if harmonic_filter_mode == "adaptive":
        result = filter_harmonics_adaptive(result)
    elif harmonic_filter_mode == "on" or (harmonic_filter_mode == "adaptive" and do_filter_harmonics):
        result = filter_harmonics(result)
    # "off" — skip filtering entirely
    result = filter_note_range(result, note_low, note_high)
    if do_filter_ghosts:
        result = filter_ghost_notes(result, bpm)
    result = quantize_onsets(result, bpm, quantize_grid)
    if preserve_durations:
        result = snap_durations(result, bpm, quantize_grid)
    else:
        result = set_durations(result, bpm, quantize_grid)
    if do_fill_patterns:
        result = fill_repeated_patterns(
            result, bpm, quantize_grid,
            min_reps=pattern_min_reps,
            fill_threshold=pattern_fill_threshold,
        )
    result = apply_velocity(result, dynamic_velocity, velocity_curve)
    return result
