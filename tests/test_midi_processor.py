"""Tests for core.midi_processor — pure logic, no audio needed.

This is the most critical test module: all functions are tested
with synthetic NoteEvent lists.
"""

import pytest

from core import NoteEvent
from core.midi_processor import (
    _sixteenth_duration,
    apply_velocity,
    filter_ghost_notes,
    filter_harmonics,
    filter_note_range,
    process,
    quantize_onsets,
    set_durations,
    snap_durations,
)


# --- Helpers ---

def _note(pitch: int = 60, start: float = 0.0, end: float = 0.5, amp: float = 0.8) -> NoteEvent:
    return NoteEvent(pitch=pitch, start_sec=start, end_sec=end, amplitude=amp)


# --- Tests ---

class TestSixteenthDuration:
    def test_120_bpm(self) -> None:
        # At 120 BPM: beat = 0.5s, 1/16th = 0.125s
        assert abs(_sixteenth_duration(120.0) - 0.125) < 1e-9

    def test_60_bpm(self) -> None:
        # At 60 BPM: beat = 1.0s, 1/16th = 0.25s
        assert abs(_sixteenth_duration(60.0) - 0.25) < 1e-9

    def test_140_bpm(self) -> None:
        expected = 60.0 / 140.0 / 4.0
        assert abs(_sixteenth_duration(140.0) - expected) < 1e-9


class TestFilterNoteRange:
    def test_keeps_notes_in_range(self) -> None:
        notes = [_note(pitch=p) for p in [36, 48, 60, 72, 84]]
        result = filter_note_range(notes, low=48, high=72)
        assert [n.pitch for n in result] == [48, 60, 72]

    def test_boundary_notes_kept(self) -> None:
        notes = [_note(pitch=36), _note(pitch=84)]
        result = filter_note_range(notes, low=36, high=84)
        assert len(result) == 2

    def test_all_filtered_out(self) -> None:
        notes = [_note(pitch=30), _note(pitch=90)]
        result = filter_note_range(notes, low=36, high=84)
        assert len(result) == 0

    def test_empty_input(self) -> None:
        result = filter_note_range([], low=36, high=84)
        assert result == []


class TestFilterGhostNotes:
    def test_removes_short_notes(self) -> None:
        # At 120 BPM, 1/16th = 0.125s
        short = _note(start=0.0, end=0.05)   # 50ms < 125ms
        long_ = _note(start=0.0, end=0.2)    # 200ms > 125ms
        result = filter_ghost_notes([short, long_], bpm=120.0)
        assert len(result) == 1
        assert result[0].end_sec == 0.2

    def test_exact_boundary_kept(self) -> None:
        # Note exactly at 1/16th duration should be kept (>= check)
        dur = _sixteenth_duration(120.0)
        note = _note(start=0.0, end=dur)
        result = filter_ghost_notes([note], bpm=120.0)
        assert len(result) == 1

    def test_different_bpm(self) -> None:
        # At 60 BPM, 1/16th = 0.25s
        note = _note(start=0.0, end=0.2)  # 200ms < 250ms
        result = filter_ghost_notes([note], bpm=60.0)
        assert len(result) == 0

    def test_empty_input(self) -> None:
        result = filter_ghost_notes([], bpm=120.0)
        assert result == []


class TestQuantizeOnsets:
    def test_snaps_to_nearest_grid(self) -> None:
        # At 120 BPM, grid = 0.125s. Note at 0.06 should snap to 0.0 (nearer) or 0.125
        # 0.06 is closer to 0.0 (delta=0.06) than 0.125 (delta=0.065)
        note = _note(start=0.06, end=0.2)
        result = quantize_onsets([note], bpm=120.0)
        assert abs(result[0].start_sec - 0.0) < 1e-9

    def test_snaps_up(self) -> None:
        # Note at 0.07 is closer to 0.125 (delta=0.055) than 0.0 (delta=0.07)
        note = _note(start=0.07, end=0.2)
        result = quantize_onsets([note], bpm=120.0)
        assert abs(result[0].start_sec - 0.125) < 1e-9

    def test_already_on_grid(self) -> None:
        note = _note(start=0.25, end=0.5)
        result = quantize_onsets([note], bpm=120.0)
        assert abs(result[0].start_sec - 0.25) < 1e-9

    def test_end_time_adjusted_by_same_delta(self) -> None:
        note = _note(start=0.06, end=0.2)
        result = quantize_onsets([note], bpm=120.0)
        # Delta is -0.06, so end should be 0.2 - 0.06 = 0.14
        assert abs(result[0].end_sec - 0.14) < 1e-9

    def test_multiple_notes(self) -> None:
        notes = [_note(start=0.06), _note(start=0.13), _note(start=0.26)]
        result = quantize_onsets(notes, bpm=120.0)
        assert abs(result[0].start_sec - 0.0) < 1e-9
        assert abs(result[1].start_sec - 0.125) < 1e-9
        assert abs(result[2].start_sec - 0.25) < 1e-9

    def test_empty_input(self) -> None:
        result = quantize_onsets([], bpm=120.0)
        assert result == []


class TestSetDurations:
    def test_all_durations_equal_sixteenth(self) -> None:
        notes = [_note(start=0.0, end=0.3), _note(start=0.5, end=1.0)]
        result = set_durations(notes, bpm=120.0)
        expected_dur = _sixteenth_duration(120.0)
        for n in result:
            actual_dur = n.end_sec - n.start_sec
            assert abs(actual_dur - expected_dur) < 1e-9

    def test_start_times_unchanged(self) -> None:
        notes = [_note(start=0.3), _note(start=0.8)]
        result = set_durations(notes, bpm=120.0)
        assert result[0].start_sec == 0.3
        assert result[1].start_sec == 0.8

    def test_different_bpm(self) -> None:
        notes = [_note(start=0.0, end=1.0)]
        result = set_durations(notes, bpm=60.0)
        expected_dur = _sixteenth_duration(60.0)
        assert abs(result[0].end_sec - result[0].start_sec - expected_dur) < 1e-9

    def test_empty_input(self) -> None:
        result = set_durations([], bpm=120.0)
        assert result == []


class TestSnapDurations:
    def test_snaps_end_to_grid(self) -> None:
        # At 120 BPM, grid = 0.125s. Note with 0.3s duration -> 2 grid units = 0.25s
        note = _note(start=0.0, end=0.3)
        result = snap_durations([note], bpm=120.0)
        grid = _sixteenth_duration(120.0)
        assert abs(result[0].end_sec - 2 * grid) < 1e-9

    def test_minimum_one_sixteenth(self) -> None:
        # Very short note should get minimum 1 grid unit
        note = _note(start=0.0, end=0.01)
        result = snap_durations([note], bpm=120.0)
        grid = _sixteenth_duration(120.0)
        assert abs(result[0].end_sec - result[0].start_sec - grid) < 1e-9

    def test_preserves_long_note(self) -> None:
        # At 120 BPM, grid = 0.125s. Note with 1.0s duration -> 8 grid units = 1.0s
        note = _note(start=0.0, end=1.0)
        result = snap_durations([note], bpm=120.0)
        assert abs(result[0].end_sec - 1.0) < 1e-9

    def test_rounds_to_nearest(self) -> None:
        # 0.19s duration at 120 BPM: 0.19/0.125 = 1.52 -> rounds to 2 -> 0.25s
        note = _note(start=0.0, end=0.19)
        result = snap_durations([note], bpm=120.0)
        grid = _sixteenth_duration(120.0)
        assert abs(result[0].end_sec - 2 * grid) < 1e-9

    def test_empty_input(self) -> None:
        result = snap_durations([], bpm=120.0)
        assert result == []


class TestFilterHarmonics:
    def test_removes_octave_overtone(self) -> None:
        # C4 (60) with louder amplitude, C5 (72) as overtone
        fundamental = _note(pitch=60, start=0.0, amp=0.9)
        overtone = _note(pitch=72, start=0.0, amp=0.5)
        result = filter_harmonics([fundamental, overtone])
        assert len(result) == 1
        assert result[0].pitch == 60

    def test_removes_fifth_overtone(self) -> None:
        # C4 (60) fundamental, G5 (79 = 60+19) as 3rd harmonic
        fundamental = _note(pitch=60, start=0.0, amp=0.8)
        overtone = _note(pitch=79, start=0.0, amp=0.4)
        result = filter_harmonics([fundamental, overtone])
        assert len(result) == 1
        assert result[0].pitch == 60

    def test_keeps_non_harmonic_notes(self) -> None:
        # C4 and E4 — not a harmonic relationship
        note_c = _note(pitch=60, start=0.0, amp=0.8)
        note_e = _note(pitch=64, start=0.0, amp=0.7)
        result = filter_harmonics([note_c, note_e])
        assert len(result) == 2

    def test_keeps_notes_at_different_times(self) -> None:
        # Same harmonic interval but at different times — not an overtone
        note1 = _note(pitch=60, start=0.0, amp=0.9)
        note2 = _note(pitch=72, start=1.0, amp=0.5)
        result = filter_harmonics([note1, note2])
        assert len(result) == 2

    def test_keeps_loud_upper_note(self) -> None:
        # If the upper note is much louder, it's probably real, not an overtone
        low = _note(pitch=60, start=0.0, amp=0.2)
        high = _note(pitch=72, start=0.0, amp=0.9)
        # Fundamental amplitude (0.2) < overtone * 0.5 (0.45), so NOT filtered
        result = filter_harmonics([low, high])
        assert len(result) == 2

    def test_multiple_harmonics_of_same_fundamental(self) -> None:
        # C4 with octave (C5) and double octave+5th (G6) overtones
        fund = _note(pitch=60, start=0.0, amp=0.9)
        h2 = _note(pitch=72, start=0.0, amp=0.4)   # +12
        h3 = _note(pitch=79, start=0.0, amp=0.3)   # +19
        h4 = _note(pitch=84, start=0.0, amp=0.2)   # +24
        result = filter_harmonics([fund, h2, h3, h4])
        assert len(result) == 1
        assert result[0].pitch == 60

    def test_empty_input(self) -> None:
        assert filter_harmonics([]) == []

    def test_single_note(self) -> None:
        result = filter_harmonics([_note(pitch=60)])
        assert len(result) == 1


class TestApplyVelocity:
    def test_dynamic_maps_amplitude(self) -> None:
        notes = [_note(amp=0.0), _note(amp=0.5), _note(amp=1.0)]
        result = apply_velocity(notes, dynamic=True)
        # amp=0.0 should clamp to minimum (~1/127)
        assert result[0].amplitude >= 1.0 / 127.0
        # amp=0.5 stays at 0.5
        assert abs(result[1].amplitude - 0.5) < 1e-9
        # amp=1.0 stays at 1.0
        assert abs(result[2].amplitude - 1.0) < 1e-9

    def test_fixed_sets_all_to_100_over_127(self) -> None:
        notes = [_note(amp=0.2), _note(amp=0.9)]
        result = apply_velocity(notes, dynamic=False)
        expected = 100.0 / 127.0
        for n in result:
            assert abs(n.amplitude - expected) < 1e-9

    def test_dynamic_clamps_above_1(self) -> None:
        note = _note(amp=1.5)
        result = apply_velocity([note], dynamic=True)
        assert result[0].amplitude <= 1.0

    def test_empty_input(self) -> None:
        assert apply_velocity([], dynamic=True) == []
        assert apply_velocity([], dynamic=False) == []


class TestProcess:
    def test_full_pipeline_force_sixteenth(self) -> None:
        notes = [
            _note(pitch=60, start=0.06, end=0.2, amp=0.8),
            _note(pitch=30, start=0.0, end=0.5, amp=0.9),   # out of range
            _note(pitch=72, start=0.13, end=0.14, amp=0.7),  # ghost note (10ms)
        ]
        result = process(
            notes, bpm=120.0, note_low=36, note_high=84,
            do_filter_ghosts=True, dynamic_velocity=False,
            preserve_durations=False,
        )
        # pitch=30 filtered by range, pitch=72 filtered as ghost
        assert len(result) == 1
        assert result[0].pitch == 60
        # Quantized to grid, 1/16th duration, fixed velocity
        assert abs(result[0].start_sec - 0.0) < 1e-9
        dur = _sixteenth_duration(120.0)
        assert abs(result[0].end_sec - result[0].start_sec - dur) < 1e-9
        assert abs(result[0].amplitude - 100.0 / 127.0) < 1e-9

    def test_full_pipeline_preserve_durations(self) -> None:
        notes = [
            _note(pitch=60, start=0.06, end=1.06, amp=0.8),  # ~1s duration
        ]
        result = process(
            notes, bpm=120.0, note_low=36, note_high=84,
            do_filter_ghosts=True, dynamic_velocity=False,
            preserve_durations=True,
        )
        assert len(result) == 1
        # Start quantized to 0.0, duration snapped to grid (~1.0s = 8 grid units)
        assert abs(result[0].start_sec - 0.0) < 1e-9
        assert abs(result[0].end_sec - 1.0) < 1e-9

    def test_ghost_filter_disabled(self) -> None:
        ghost = _note(pitch=60, start=0.0, end=0.01, amp=0.8)
        result = process(
            [ghost], bpm=120.0, note_low=36, note_high=84,
            do_filter_ghosts=False, dynamic_velocity=False,
            preserve_durations=False,
        )
        assert len(result) == 1
