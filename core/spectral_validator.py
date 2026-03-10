"""Spectral validation: use CQT spectrogram as ground truth to validate and recover notes.

Instead of guessing which notes are overtones by amplitude ratio, we check
whether a frequency actually has independent energy in the spectrogram.
This correctly preserves intentionally played octaves, fifths, etc.

All thresholds are RELATIVE to the piece's own energy profile, so the same
settings work on quiet recordings and loud ones.

Pipeline: raw basic-pitch notes → spectral_validate() → cleaned notes
(runs BEFORE midi_processor steps like harmonic filter, quantize, etc.)
"""

import numpy as np
import librosa

from core import AudioData, NoteEvent


# CQT parameters — 1 bin per semitone, MIDI 24 (C1) to 108 (C8)
_MIDI_LOW = 24
_MIDI_HIGH = 108
_N_BINS = _MIDI_HIGH - _MIDI_LOW  # 84 bins

# Known model blind spots — pitches basic-pitch consistently misses.
# These get extra-sensitive recovery thresholds.
_BLIND_SPOT_PITCHES = {66, 71}  # F#4, B4


def compute_cqt(
    audio: AudioData,
    hop_length: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a Constant-Q Transform of the audio.

    Returns energy (magnitude) per semitone per time frame.

    Args:
        audio: AudioData with mono float32 samples.
        hop_length: Hop length in samples (controls time resolution).

    Returns:
        cqt_mag: 2D array of shape (n_bins, n_frames) — magnitude in dB.
        frame_times: 1D array of shape (n_frames,) — time in seconds per frame.
    """
    sr = audio.sample_rate
    fmin = librosa.midi_to_hz(_MIDI_LOW)

    cqt_complex = librosa.cqt(
        y=audio.samples,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=_N_BINS,
        bins_per_octave=12,
    )

    # Convert to dB magnitude (more perceptually meaningful)
    cqt_mag = librosa.amplitude_to_db(np.abs(cqt_complex), ref=np.max)

    frame_times = librosa.frames_to_time(
        np.arange(cqt_mag.shape[1]),
        sr=sr,
        hop_length=hop_length,
    )

    return cqt_mag, frame_times


def _pitch_to_bin(pitch: int) -> int | None:
    """Convert MIDI pitch to CQT bin index. Returns None if out of range."""
    b = pitch - _MIDI_LOW
    if 0 <= b < _N_BINS:
        return b
    return None


def _frames_for_note(
    start_sec: float,
    end_sec: float,
    frame_times: np.ndarray,
) -> tuple[int, int]:
    """Return (start_frame, end_frame) indices for a note's time range."""
    start_idx = int(np.searchsorted(frame_times, start_sec))
    end_idx = int(np.searchsorted(frame_times, end_sec))
    # Ensure at least 1 frame
    if end_idx <= start_idx:
        end_idx = min(start_idx + 1, len(frame_times))
    return start_idx, end_idx


def cqt_energy_at(
    cqt_mag: np.ndarray,
    pitch: int,
    start_sec: float,
    end_sec: float,
    frame_times: np.ndarray,
) -> float:
    """Get mean CQT energy (dB) at a specific pitch over a time range.

    Returns -80 if pitch is out of CQT range.
    """
    b = _pitch_to_bin(pitch)
    if b is None:
        return -80.0

    f0, f1 = _frames_for_note(start_sec, end_sec, frame_times)
    if f0 >= cqt_mag.shape[1]:
        return -80.0

    f1 = min(f1, cqt_mag.shape[1])
    return float(np.mean(cqt_mag[b, f0:f1]))


def expected_overtone_energy(
    cqt_mag: np.ndarray,
    fundamental_pitch: int,
    start_sec: float,
    end_sec: float,
    frame_times: np.ndarray,
) -> dict[int, float]:
    """Estimate how much energy at harmonic pitches is explained by a fundamental.

    Returns:
        Dict mapping harmonic pitch -> expected energy (dB) from this fundamental.
    """
    harmonic_intervals = [12, 19, 24, 28, 31]
    harmonic_decay_db = [10, 16, 20, 24, 28]

    fund_energy = cqt_energy_at(cqt_mag, fundamental_pitch, start_sec, end_sec, frame_times)
    result = {}
    for interval, decay in zip(harmonic_intervals, harmonic_decay_db):
        h_pitch = fundamental_pitch + interval
        if _pitch_to_bin(h_pitch) is not None:
            result[h_pitch] = fund_energy - decay
    return result


def _compute_reference_energy(
    cqt_mag: np.ndarray,
    frame_times: np.ndarray,
    notes: list[NoteEvent],
) -> float:
    """Compute the median energy of confident notes as a reference baseline.

    This gives us a relative threshold: "what does a real note look like
    in this specific piece?" rather than using absolute dB values.
    """
    if not notes:
        return -40.0  # fallback

    energies = []
    for n in notes:
        e = cqt_energy_at(cqt_mag, n.pitch, n.start_sec, n.end_sec, frame_times)
        energies.append(e)

    return float(np.median(energies))


def validate_notes(
    cqt_mag: np.ndarray,
    frame_times: np.ndarray,
    notes: list[NoteEvent],
    energy_floor_db: float = -50.0,
    overtone_margin_db: float = 6.0,
    relative_floor: bool = True,
    floor_offset_db: float = 30.0,
) -> tuple[list[NoteEvent], list[NoteEvent]]:
    """Validate notes against spectral energy. Reject false positives.

    Two checks per note:
    1. Energy check: does this pitch have energy above the noise floor?
       In relative mode, the floor is computed from the piece's own
       note energies (median - floor_offset_db), so quiet and loud
       tracks are handled equally.
    2. Overtone check: if a lower simultaneous note could explain this
       note's energy as an overtone, does the actual energy exceed the
       expected overtone level by at least overtone_margin_db?

    Args:
        cqt_mag: CQT magnitude array (dB).
        frame_times: Time axis for CQT.
        notes: Notes to validate.
        energy_floor_db: Absolute minimum energy to consider a note real.
            Used as fallback when relative_floor is False.
        overtone_margin_db: How much louder than expected overtone energy
            a note must be to count as independently played.
        relative_floor: If True, compute energy floor relative to the
            piece's median note energy.
        floor_offset_db: How far below median note energy to set the floor.

    Returns:
        (kept_notes, rejected_notes) — two lists.
    """
    if not notes:
        return [], []

    # Compute energy for each note
    note_energies = []
    for n in notes:
        e = cqt_energy_at(cqt_mag, n.pitch, n.start_sec, n.end_sec, frame_times)
        note_energies.append(e)

    # Set the energy floor relative to this piece's note energies
    if relative_floor and note_energies:
        ref_energy = float(np.median(note_energies))
        floor = ref_energy - floor_offset_db
    else:
        floor = energy_floor_db

    kept = []
    rejected = []

    for i, note in enumerate(notes):
        energy = note_energies[i]

        # Check 1: is there any meaningful energy at this pitch?
        if energy < floor:
            rejected.append(note)
            continue

        # Check 2: could this be an overtone of a lower simultaneous note?
        is_overtone = False
        for j, other in enumerate(notes):
            if j == i or other.pitch >= note.pitch:
                continue
            # Check temporal overlap
            overlap_start = max(note.start_sec, other.start_sec)
            overlap_end = min(note.end_sec, other.end_sec)
            if overlap_end <= overlap_start:
                continue

            # Get expected overtone energy from this fundamental
            overtone_map = expected_overtone_energy(
                cqt_mag, other.pitch, overlap_start, overlap_end, frame_times
            )
            if note.pitch in overtone_map:
                expected = overtone_map[note.pitch]
                if energy < expected + overtone_margin_db:
                    is_overtone = True
                    break

        if is_overtone:
            rejected.append(note)
        else:
            kept.append(note)

    return kept, rejected


def walk_sustain(
    cqt_mag: np.ndarray,
    pitch: int,
    start_frame: int,
    frame_times: np.ndarray,
    drop_db: float = 12.0,
    min_energy_db: float = -50.0,
) -> float:
    """Walk forward from an onset to estimate note duration from spectral energy.

    Starts at start_frame and walks until energy drops by drop_db from
    the initial level, or falls below min_energy_db.

    Returns:
        Duration in seconds.
    """
    b = _pitch_to_bin(pitch)
    if b is None or start_frame >= cqt_mag.shape[1]:
        return 0.0

    initial_energy = cqt_mag[b, start_frame]
    if initial_energy < min_energy_db:
        return 0.0

    end_frame = start_frame + 1
    n_frames = cqt_mag.shape[1]

    while end_frame < n_frames:
        e = cqt_mag[b, end_frame]
        if e < initial_energy - drop_db or e < min_energy_db:
            break
        end_frame += 1

    if end_frame >= n_frames:
        duration = frame_times[-1] - frame_times[start_frame]
    else:
        duration = frame_times[end_frame] - frame_times[start_frame]

    return max(0.0, duration)


def _detect_onsets_at_bin(
    cqt_mag: np.ndarray,
    b: int,
    onset_rise_db: float = 10.0,
    min_energy_db: float = -25.0,
) -> list[int]:
    """Detect onset frames in a single CQT bin by looking for sudden energy rises.

    An onset is a frame where energy jumps by at least onset_rise_db compared
    to the preceding frames AND the energy is above min_energy_db.

    Returns list of frame indices.
    """
    n_frames = cqt_mag.shape[1]
    if n_frames < 3:
        return []

    onsets = []
    lookback = 3
    for f in range(lookback, n_frames):
        current = cqt_mag[b, f]
        if current < min_energy_db:
            continue
        prior = float(np.mean(cqt_mag[b, max(0, f - lookback):f]))
        if current - prior >= onset_rise_db:
            onsets.append(f)
    return onsets


def _find_sustained_regions(
    cqt_mag: np.ndarray,
    b: int,
    frame_times: np.ndarray,
    min_energy_db: float,
    min_duration_sec: float,
) -> list[tuple[int, int]]:
    """Find contiguous regions where energy stays above threshold.

    This catches swelling notes that onset detection misses — any region
    with sustained energy above the threshold counts, regardless of how
    the energy arrived there.

    Returns list of (start_frame, end_frame) tuples.
    """
    n_frames = cqt_mag.shape[1]
    regions = []
    start = None

    for f in range(n_frames):
        if cqt_mag[b, f] >= min_energy_db:
            if start is None:
                start = f
        else:
            if start is not None:
                # Check duration
                if f - 1 >= 0:
                    dur = frame_times[min(f, n_frames - 1)] - frame_times[start]
                    if dur >= min_duration_sec:
                        regions.append((start, f))
                start = None

    # Handle region that extends to end
    if start is not None:
        dur = frame_times[-1] - frame_times[start] if n_frames > 0 else 0
        if dur >= min_duration_sec:
            regions.append((start, n_frames))

    return regions


def recover_notes(
    cqt_mag: np.ndarray,
    frame_times: np.ndarray,
    existing_notes: list[NoteEvent],
    ref_energy: float = -15.0,
    min_duration_sec: float = 0.08,
    blind_spot_boost_db: float = 8.0,
    recovery_threshold_offset_db: float = 20.0,
    target_pitches: set[int] | None = None,
) -> list[NoteEvent]:
    """Find spectral energy not explained by existing notes and propose new notes.

    Uses AVERAGE energy over sustained regions (not onset detection alone)
    so that swelling notes and soft chord tones are caught. The energy
    threshold is relative to the piece's confident note energy.

    Also cross-references with existing notes at the same onset time:
    if other confident notes start at the same time, a candidate at a
    harmonic interval is more likely to be a real chord tone.

    Args:
        cqt_mag: CQT magnitude array (dB).
        frame_times: Time axis.
        existing_notes: Already validated notes.
        ref_energy: Reference energy (median of confident notes) — used
            to set relative thresholds. Caller should compute this.
        min_duration_sec: Minimum sustain duration to keep a candidate.
        blind_spot_boost_db: Lower thresholds by this much for blind spot pitches.
        recovery_threshold_offset_db: How far below ref_energy to set the
            recovery threshold (larger = more sensitive).
        target_pitches: If set, only recover at these specific pitches.
            None means recover at all pitches.

    Returns:
        List of newly recovered NoteEvent candidates.
    """
    n_bins, n_frames = cqt_mag.shape
    if n_frames < 4:
        return []

    base_threshold = ref_energy - recovery_threshold_offset_db

    # Build an occupancy map
    occupied = np.zeros((n_bins, n_frames), dtype=bool)
    for note in existing_notes:
        b = _pitch_to_bin(note.pitch)
        if b is None:
            continue
        f0, f1 = _frames_for_note(note.start_sec, note.end_sec, frame_times)
        f1 = min(f1, n_frames)
        f0 = max(0, f0 - 1)
        f1 = min(n_frames, f1 + 1)
        occupied[b, f0:f1] = True

    # Build onset map of existing notes for cross-referencing
    existing_onsets: dict[float, list[NoteEvent]] = {}
    for note in existing_notes:
        # Round to nearest frame time for grouping
        f0 = int(np.searchsorted(frame_times, note.start_sec))
        key = round(note.start_sec, 3)
        existing_onsets.setdefault(key, []).append(note)

    # Determine which bins to scan
    if target_pitches is not None:
        scan_bins = []
        for p in target_pitches:
            b = _pitch_to_bin(p)
            if b is not None:
                scan_bins.append(b)
    else:
        scan_bins = list(range(n_bins))

    recovered = []
    for b in scan_bins:
        pitch = b + _MIDI_LOW
        is_blind_spot = pitch in _BLIND_SPOT_PITCHES

        threshold = base_threshold
        if is_blind_spot:
            threshold -= blind_spot_boost_db

        # Find sustained energy regions at this pitch
        regions = _find_sustained_regions(
            cqt_mag, b, frame_times, threshold, min_duration_sec,
        )

        for start_f, end_f in regions:
            # Skip if fully occupied by existing notes
            region_occupied = occupied[b, start_f:end_f]
            if region_occupied.all():
                continue

            # Find the unoccupied portion
            unoccupied_frames = np.where(~region_occupied)[0] + start_f
            if len(unoccupied_frames) == 0:
                continue

            # Get the contiguous unoccupied sub-region
            first_free = int(unoccupied_frames[0])
            last_free = int(unoccupied_frames[-1])

            onset_sec = float(frame_times[first_free])
            end_sec = float(frame_times[min(last_free + 1, n_frames - 1)])
            duration = end_sec - onset_sec

            if duration < min_duration_sec:
                continue

            # Compute average energy over the candidate region
            avg_energy = float(np.mean(cqt_mag[b, first_free:last_free + 1]))
            if avg_energy < threshold:
                continue

            # Cross-reference: is there a simultaneous existing note?
            # This boosts confidence for chord tones
            has_simultaneous = False
            onset_tol = 0.06
            for note in existing_notes:
                if abs(note.start_sec - onset_sec) <= onset_tol:
                    has_simultaneous = True
                    break

            # For non-blind-spot pitches, require simultaneous notes
            # (reduces false positives from resonances)
            if not is_blind_spot and not has_simultaneous:
                if target_pitches is None:
                    continue

            # Estimate amplitude from energy relative to reference
            amplitude = float(np.clip((avg_energy + 80.0) / 80.0, 0.05, 1.0))

            recovered.append(NoteEvent(
                pitch=pitch,
                start_sec=onset_sec,
                end_sec=end_sec,
                amplitude=amplitude,
            ))

            # Mark as occupied
            occupied[b, first_free:last_free + 1] = True

    return recovered


def resolve_overlaps(
    notes: list[NoteEvent],
    onset_tol: float = 0.03,
) -> list[NoteEvent]:
    """Resolve overlapping notes of the same pitch.

    When two notes of the same pitch overlap, one of these happened:
    1. The earlier note should have ended before the new one starts
       (basic-pitch overestimated its duration).
    2. The later note is a false re-detection of the still-sounding note.

    Decision logic:
    - If the later note starts very close to the earlier note's start
      (within onset_tol), keep the louder one.
    - Otherwise, trim the earlier note to end at the later note's start.

    Args:
        notes: Input note events.
        onset_tol: Tolerance for considering onsets as "same time".

    Returns:
        Cleaned note list with overlaps resolved.
    """
    if len(notes) <= 1:
        return list(notes)

    # Group by pitch
    by_pitch: dict[int, list[NoteEvent]] = {}
    for n in notes:
        by_pitch.setdefault(n.pitch, []).append(n)

    result = []
    for pitch, group in by_pitch.items():
        sorted_group = sorted(group, key=lambda n: n.start_sec)
        cleaned: list[NoteEvent] = []

        for note in sorted_group:
            if not cleaned:
                cleaned.append(note)
                continue

            prev = cleaned[-1]

            # Check overlap: does prev extend past note's start?
            if prev.end_sec <= note.start_sec:
                cleaned.append(note)
                continue

            # Overlap detected
            if abs(note.start_sec - prev.start_sec) <= onset_tol:
                # Near-simultaneous onsets: keep the louder, longer one
                if note.amplitude > prev.amplitude:
                    cleaned[-1] = note
                elif (note.amplitude == prev.amplitude
                      and (note.end_sec - note.start_sec) > (prev.end_sec - prev.start_sec)):
                    cleaned[-1] = note
            else:
                # Different onset times: trim the earlier note
                trimmed = NoteEvent(
                    pitch=prev.pitch,
                    start_sec=prev.start_sec,
                    end_sec=note.start_sec,
                    amplitude=prev.amplitude,
                )
                cleaned[-1] = trimmed
                cleaned.append(note)

        result.extend(cleaned)

    return result


def spectral_validate(
    audio: AudioData,
    notes: list[NoteEvent],
    energy_floor_db: float = -60.0,
    overtone_margin_db: float = -3.0,
    floor_offset_db: float = 30.0,
    recovery_threshold_offset_db: float = 20.0,
    min_recovery_duration_sec: float = 0.08,
    blind_spot_boost_db: float = 8.0,
    do_validate: bool = True,
    do_recover: bool = False,
    do_resolve_overlaps: bool = True,
    hop_length: int = 512,
) -> list[NoteEvent]:
    """Top-level entry point: validate notes against the spectrogram.

    All thresholds are computed RELATIVE to the piece's own energy profile
    (median of confident note energies). This means the same settings work
    for quiet recordings and loud ones.

    1. Compute CQT from audio
    2. Compute reference energy from confident notes
    3. Optionally validate existing notes (reject false positives)
    4. Optionally recover missing notes from sustained energy regions
    5. Optionally resolve overlapping same-pitch notes

    Args:
        audio: AudioData with mono samples.
        notes: Raw notes from basic-pitch (pre-processing).
        energy_floor_db: Absolute fallback floor (used if relative fails).
        overtone_margin_db: Margin for overtone disambiguation.
            Negative = more conservative (fewer rejections).
        floor_offset_db: How far below median note energy to set validation floor.
        recovery_threshold_offset_db: How far below median for recovery.
        min_recovery_duration_sec: Minimum duration for recovered notes.
        blind_spot_boost_db: Extra sensitivity for known blind spot pitches.
        do_validate: Whether to validate existing notes.
        do_recover: Whether to attempt note recovery.
        do_resolve_overlaps: Whether to fix same-pitch overlaps.
        hop_length: CQT hop length (time resolution).

    Returns:
        Validated and cleaned list of NoteEvent.
    """
    cqt_mag, frame_times = compute_cqt(audio, hop_length=hop_length)

    # Compute reference energy from the confident notes
    ref_energy = _compute_reference_energy(cqt_mag, frame_times, notes)

    # Step 1: validate existing notes (with relative thresholds)
    if do_validate:
        kept, _rejected = validate_notes(
            cqt_mag, frame_times, notes,
            energy_floor_db=energy_floor_db,
            overtone_margin_db=overtone_margin_db,
            relative_floor=True,
            floor_offset_db=floor_offset_db,
        )
    else:
        kept = list(notes)

    # Step 2: recover missing notes from sustained spectral energy
    if do_recover:
        recovered = recover_notes(
            cqt_mag, frame_times, kept,
            ref_energy=ref_energy,
            min_duration_sec=min_recovery_duration_sec,
            blind_spot_boost_db=blind_spot_boost_db,
            recovery_threshold_offset_db=recovery_threshold_offset_db,
            target_pitches=_BLIND_SPOT_PITCHES,
        )
        kept = kept + recovered

    # Step 3: resolve overlaps
    if do_resolve_overlaps:
        kept = resolve_overlaps(kept)

    return kept
