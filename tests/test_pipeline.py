"""End-to-end pipeline tests using fixture pairs.

Fixtures live flat in tests/fixtures/. Audio and MIDI files are matched
by filename stem (case-insensitive):

    piano_simple.wav  <-->  Piano_Simple.mid

Processing config comes from defaults.json, optionally overridden by
a per-file {stem}.json.

Instead of demanding exact note-for-note match (unrealistic for ML
transcription), we measure precision, recall, and F1 score using
mir_eval's note-level metrics with configurable tolerances.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pretty_midi
import pytest

from core import ProcessingConfig
from core.audio_loader import load_audio, apply_hpss
from core.transcriber import transcribe
from core.midi_processor import process
from core.midi_exporter import build_midi


FIXTURES_DIR = Path(__file__).parent / 'fixtures'
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac'}

# Tolerances for note matching
ONSET_TOLERANCE_SEC = 0.05   # 50ms onset tolerance
PITCH_TOLERANCE = 0          # exact pitch match required

# Minimum accuracy thresholds to pass
MIN_PRECISION = 0.50
MIN_RECALL = 0.50
MIN_F1 = 0.50


class FixturePair(NamedTuple):
    name: str
    audio_path: Path
    midi_path: Path


@dataclass
class MatchResult:
    """Results of comparing detected vs expected notes."""
    total_detected: int
    total_expected: int
    true_positives: int
    precision: float
    recall: float
    f1: float
    missed_pitches: list[int]
    extra_pitches: list[int]


def _load_defaults() -> dict:
    """Load the shared defaults.json config."""
    defaults_path = FIXTURES_DIR / 'defaults.json'
    if defaults_path.exists():
        with open(defaults_path) as f:
            return json.load(f)
    return {
        "bpm": 120,
        "start_sec": 0,
        "end_sec": 0,
        "confidence_threshold": 0.5,
        "note_low": 36,
        "note_high": 84,
        "filter_ghost_notes": True,
        "dynamic_velocity": False,
        "preserve_durations": True,
        "onset_threshold": 0.5,
        "frame_threshold": 0.3,
        "minimum_note_length_ms": 58,
        "ensemble_passes": 1,
        "use_hpss": False,
        "filter_harmonics": True,
    }


def _load_config_for(stem: str) -> dict:
    """Load defaults, then overlay any per-file {stem}.json overrides."""
    config = _load_defaults()
    for f in FIXTURES_DIR.iterdir():
        if f.suffix == '.json' and f.stem.lower() == stem.lower() and f.name.lower() != 'defaults.json':
            with open(f) as fh:
                overrides = json.load(fh)
            config.update(overrides)
            break
    return config


def _match_notes(
    detected: list[pretty_midi.Note],
    expected: list[pretty_midi.Note],
    onset_tol: float = ONSET_TOLERANCE_SEC,
) -> MatchResult:
    """Match detected notes against expected notes with tolerance.

    A detected note is a true positive if there exists an unmatched expected
    note with the same pitch and onset within onset_tol seconds.
    """
    expected_matched = [False] * len(expected)
    detected_matched = [False] * len(detected)

    # Greedy matching: for each detected note, find closest unmatched expected
    for di, dn in enumerate(detected):
        best_idx = -1
        best_delta = float('inf')
        for ei, en in enumerate(expected):
            if expected_matched[ei]:
                continue
            if dn.pitch != en.pitch:
                continue
            delta = abs(dn.start - en.start)
            if delta <= onset_tol and delta < best_delta:
                best_delta = delta
                best_idx = ei
        if best_idx >= 0:
            detected_matched[di] = True
            expected_matched[best_idx] = True

    tp = sum(detected_matched)
    total_det = len(detected)
    total_exp = len(expected)

    precision = tp / total_det if total_det > 0 else 1.0
    recall = tp / total_exp if total_exp > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    missed = [expected[i].pitch for i in range(total_exp) if not expected_matched[i]]
    extra = [detected[i].pitch for i in range(total_det) if not detected_matched[i]]

    return MatchResult(
        total_detected=total_det,
        total_expected=total_exp,
        true_positives=tp,
        precision=precision,
        recall=recall,
        f1=f1,
        missed_pitches=missed,
        extra_pitches=extra,
    )


def _find_fixture_pairs() -> list[FixturePair]:
    """Find all audio+MIDI pairs matched by case-insensitive stem."""
    if not FIXTURES_DIR.exists():
        return []

    audio_by_stem: dict[str, Path] = {}
    midi_by_stem: dict[str, Path] = {}

    for f in FIXTURES_DIR.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() in AUDIO_EXTENSIONS:
            audio_by_stem[f.stem.lower()] = f
        elif f.suffix.lower() == '.mid':
            midi_by_stem[f.stem.lower()] = f

    pairs = []
    for stem in sorted(audio_by_stem.keys()):
        if stem in midi_by_stem:
            pairs.append(FixturePair(
                name=stem,
                audio_path=audio_by_stem[stem],
                midi_path=midi_by_stem[stem],
            ))
    return pairs


def _pair_ids(pairs: list[FixturePair]) -> list[str]:
    return [p.name for p in pairs]


_PAIRS = _find_fixture_pairs()


@pytest.mark.parametrize("pair", _PAIRS, ids=_pair_ids(_PAIRS))
def test_pipeline_fixture(pair: FixturePair) -> None:
    """Run the full pipeline on a fixture and check accuracy metrics."""
    meta = _load_config_for(pair.name)

    config = ProcessingConfig(
        bpm=meta['bpm'],
        start_sec=meta['start_sec'],
        end_sec=meta['end_sec'],
        note_low=meta['note_low'],
        note_high=meta['note_high'],
        confidence_threshold=meta['confidence_threshold'],
        filter_ghost_notes=meta['filter_ghost_notes'],
        dynamic_velocity=meta['dynamic_velocity'],
        preserve_durations=meta.get('preserve_durations', True),
        onset_threshold=meta.get('onset_threshold', 0.5),
        frame_threshold=meta.get('frame_threshold', 0.3),
        minimum_note_length_ms=meta.get('minimum_note_length_ms', 58),
        ensemble_passes=meta.get('ensemble_passes', 1),
        use_hpss=meta.get('use_hpss', False),
        filter_harmonics=meta.get('filter_harmonics', True),
        harmonic_filter_mode=meta.get('harmonic_filter_mode', 'adaptive'),
        merge_fragments=meta.get('merge_fragments', False),
        fragment_gap_tol=meta.get('fragment_gap_tol', 0.05),
        fragment_reattack_ratio=meta.get('fragment_reattack_ratio', 1.5),
        timing_offset_grid=meta.get('timing_offset_grid', 0.0),
        fill_patterns=meta.get('fill_patterns', False),
        pattern_min_reps=meta.get('pattern_min_reps', 4),
        pattern_fill_threshold=meta.get('pattern_fill_threshold', 0.75),
    )

    end_sec = config.end_sec if config.end_sec > 0 else None
    audio = load_audio(pair.audio_path, start_sec=config.start_sec, end_sec=end_sec)

    if config.use_hpss:
        audio = apply_hpss(audio)

    notes = transcribe(
        audio,
        confidence_threshold=config.confidence_threshold,
        onset_threshold=config.onset_threshold,
        frame_threshold=config.frame_threshold,
        minimum_note_length_ms=config.minimum_note_length_ms,
        ensemble_passes=config.ensemble_passes,
    )
    processed = process(
        notes,
        bpm=config.bpm,
        note_low=config.note_low,
        note_high=config.note_high,
        do_filter_ghosts=config.filter_ghost_notes,
        dynamic_velocity=config.dynamic_velocity,
        preserve_durations=config.preserve_durations,
        do_filter_harmonics=config.filter_harmonics,
        harmonic_filter_mode=config.harmonic_filter_mode,
        do_merge_fragments=config.merge_fragments,
        fragment_gap_tol=config.fragment_gap_tol,
        fragment_reattack_ratio=config.fragment_reattack_ratio,
        timing_offset_grid=config.timing_offset_grid,
        do_fill_patterns=config.fill_patterns,
        pattern_min_reps=config.pattern_min_reps,
        pattern_fill_threshold=config.pattern_fill_threshold,
    )
    result_midi = build_midi(processed, config.bpm)

    # Load expected MIDI (gather notes from all tracks)
    expected_midi = pretty_midi.PrettyMIDI(str(pair.midi_path))
    expected_notes_raw = []
    for inst in expected_midi.instruments:
        expected_notes_raw.extend(inst.notes)

    result_notes = sorted(result_midi.instruments[0].notes, key=lambda n: (n.start, n.pitch))
    expected_notes = sorted(expected_notes_raw, key=lambda n: (n.start, n.pitch))

    # Compute accuracy
    match = _match_notes(result_notes, expected_notes)

    # Print detailed report for diagnostics
    print(f"\n{'='*60}")
    print(f"  {pair.name}")
    print(f"  Detected: {match.total_detected}  Expected: {match.total_expected}  "
          f"Matched: {match.true_positives}")
    print(f"  Precision: {match.precision:.2%}  Recall: {match.recall:.2%}  "
          f"F1: {match.f1:.2%}")
    if match.missed_pitches:
        from gui.controls_panel import midi_to_name
        missed_names = [midi_to_name(p) for p in match.missed_pitches[:10]]
        print(f"  Missed notes: {', '.join(missed_names)}"
              f"{'...' if len(match.missed_pitches) > 10 else ''}")
    if match.extra_pitches:
        from gui.controls_panel import midi_to_name
        extra_names = [midi_to_name(p) for p in match.extra_pitches[:10]]
        print(f"  Extra notes:  {', '.join(extra_names)}"
              f"{'...' if len(match.extra_pitches) > 10 else ''}")
    print(f"{'='*60}")

    # Assert minimum quality thresholds
    assert match.precision >= MIN_PRECISION, (
        f"[{pair.name}] Precision {match.precision:.2%} below threshold {MIN_PRECISION:.0%}"
    )
    assert match.recall >= MIN_RECALL, (
        f"[{pair.name}] Recall {match.recall:.2%} below threshold {MIN_RECALL:.0%}"
    )
    assert match.f1 >= MIN_F1, (
        f"[{pair.name}] F1 {match.f1:.2%} below threshold {MIN_F1:.0%}"
    )
