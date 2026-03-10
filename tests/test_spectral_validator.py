"""Tests for core.spectral_validator — spectral validation and overlap resolution.

Uses synthetic CQT data (no audio files needed) to test each function
independently. Also tests exact note correctness (pitch + timing), not
just note counts.
"""

import numpy as np
import pytest

from core import AudioData, NoteEvent
from core.spectral_validator import (
    _MIDI_HIGH,
    _MIDI_LOW,
    _N_BINS,
    _BLIND_SPOT_PITCHES,
    _compute_reference_energy,
    _detect_onsets_at_bin,
    _find_sustained_regions,
    _frames_for_note,
    _pitch_to_bin,
    compute_cqt,
    cqt_energy_at,
    expected_overtone_energy,
    recover_notes,
    resolve_overlaps,
    spectral_validate,
    validate_notes,
    walk_sustain,
)


# --- Helpers ---

def _note(pitch: int = 60, start: float = 0.0, end: float = 0.5, amp: float = 0.8) -> NoteEvent:
    return NoteEvent(pitch=pitch, start_sec=start, end_sec=end, amplitude=amp)


def _make_cqt(
    n_frames: int = 100,
    frame_duration: float = 0.023,  # ~512/22050
    default_db: float = -80.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic CQT with all bins at default_db."""
    cqt_mag = np.full((_N_BINS, n_frames), default_db, dtype=np.float64)
    frame_times = np.arange(n_frames) * frame_duration
    return cqt_mag, frame_times


def _set_energy(
    cqt_mag: np.ndarray,
    frame_times: np.ndarray,
    pitch: int,
    start_sec: float,
    end_sec: float,
    energy_db: float,
) -> None:
    """Set energy at a specific pitch and time range in the CQT."""
    b = _pitch_to_bin(pitch)
    if b is None:
        return
    f0, f1 = _frames_for_note(start_sec, end_sec, frame_times)
    f1 = min(f1, cqt_mag.shape[1])
    cqt_mag[b, f0:f1] = energy_db


# --- Unit tests ---

class TestPitchToBin:
    def test_midi_low_is_bin_zero(self) -> None:
        assert _pitch_to_bin(_MIDI_LOW) == 0

    def test_midi_high_minus_one(self) -> None:
        assert _pitch_to_bin(_MIDI_HIGH - 1) == _N_BINS - 1

    def test_out_of_range_low(self) -> None:
        assert _pitch_to_bin(_MIDI_LOW - 1) is None

    def test_out_of_range_high(self) -> None:
        assert _pitch_to_bin(_MIDI_HIGH) is None

    def test_middle_c(self) -> None:
        assert _pitch_to_bin(60) == 60 - _MIDI_LOW


class TestFramesForNote:
    def test_basic_range(self) -> None:
        frame_times = np.array([0.0, 0.023, 0.046, 0.069, 0.092])
        f0, f1 = _frames_for_note(0.02, 0.07, frame_times)
        assert f0 == 1
        assert f1 == 4

    def test_zero_duration_gets_one_frame(self) -> None:
        frame_times = np.array([0.0, 0.023, 0.046])
        f0, f1 = _frames_for_note(0.023, 0.023, frame_times)
        assert f1 > f0


class TestCqtEnergyAt:
    def test_returns_set_energy(self) -> None:
        cqt, ft = _make_cqt(50, default_db=-80.0)
        _set_energy(cqt, ft, pitch=60, start_sec=0.0, end_sec=0.5, energy_db=-10.0)
        e = cqt_energy_at(cqt, 60, 0.0, 0.5, ft)
        assert e == pytest.approx(-10.0, abs=1.0)

    def test_silent_region(self) -> None:
        cqt, ft = _make_cqt(50)
        e = cqt_energy_at(cqt, 60, 0.0, 0.5, ft)
        assert e == pytest.approx(-80.0, abs=1.0)

    def test_out_of_range_pitch(self) -> None:
        cqt, ft = _make_cqt(50)
        e = cqt_energy_at(cqt, 10, 0.0, 0.5, ft)
        assert e == -80.0


class TestExpectedOvertoneEnergy:
    def test_returns_harmonic_pitches(self) -> None:
        cqt, ft = _make_cqt(50)
        _set_energy(cqt, ft, pitch=48, start_sec=0.0, end_sec=0.5, energy_db=-10.0)
        result = expected_overtone_energy(cqt, 48, 0.0, 0.5, ft)
        assert 60 in result
        assert 67 in result
        assert 72 in result

    def test_decay_ordering(self) -> None:
        cqt, ft = _make_cqt(50)
        _set_energy(cqt, ft, pitch=48, start_sec=0.0, end_sec=0.5, energy_db=-10.0)
        result = expected_overtone_energy(cqt, 48, 0.0, 0.5, ft)
        assert result[60] > result[67] > result[72]


class TestComputeReferenceEnergy:
    def test_returns_median_of_note_energies(self) -> None:
        cqt, ft = _make_cqt(100)
        _set_energy(cqt, ft, pitch=60, start_sec=0.0, end_sec=0.5, energy_db=-10.0)
        _set_energy(cqt, ft, pitch=64, start_sec=0.0, end_sec=0.5, energy_db=-20.0)
        _set_energy(cqt, ft, pitch=67, start_sec=0.0, end_sec=0.5, energy_db=-15.0)

        notes = [
            _note(pitch=60, start=0.0, end=0.5),
            _note(pitch=64, start=0.0, end=0.5),
            _note(pitch=67, start=0.0, end=0.5),
        ]
        ref = _compute_reference_energy(cqt, ft, notes)
        assert ref == pytest.approx(-15.0, abs=1.0)  # median of -10, -15, -20

    def test_empty_notes_returns_fallback(self) -> None:
        cqt, ft = _make_cqt(50)
        ref = _compute_reference_energy(cqt, ft, [])
        assert ref == -40.0


class TestValidateNotes:
    def test_keeps_note_with_energy(self) -> None:
        cqt, ft = _make_cqt(100)
        _set_energy(cqt, ft, pitch=60, start_sec=0.0, end_sec=0.5, energy_db=-10.0)

        notes = [_note(pitch=60, start=0.0, end=0.5)]
        kept, rejected = validate_notes(cqt, ft, notes, relative_floor=False, energy_floor_db=-50.0)
        assert len(kept) == 1
        assert len(rejected) == 0

    def test_rejects_note_without_energy(self) -> None:
        cqt, ft = _make_cqt(100)
        notes = [_note(pitch=60, start=0.0, end=0.5)]
        kept, rejected = validate_notes(cqt, ft, notes, relative_floor=False, energy_floor_db=-50.0)
        assert len(kept) == 0
        assert len(rejected) == 1

    def test_relative_floor_adapts_to_piece(self) -> None:
        """Relative floor should keep notes that are close to median energy."""
        cqt, ft = _make_cqt(100)
        # 3 notes: 2 strong, 1 weaker but still within 30dB of median
        _set_energy(cqt, ft, pitch=60, start_sec=0.0, end_sec=0.5, energy_db=-10.0)
        _set_energy(cqt, ft, pitch=64, start_sec=0.0, end_sec=0.5, energy_db=-12.0)
        _set_energy(cqt, ft, pitch=67, start_sec=0.0, end_sec=0.5, energy_db=-35.0)

        notes = [
            _note(pitch=60, start=0.0, end=0.5),
            _note(pitch=64, start=0.0, end=0.5),
            _note(pitch=67, start=0.0, end=0.5),
        ]
        # Median energy ~ -12, floor = -12 - 30 = -42. -35 > -42, so kept.
        kept, rejected = validate_notes(cqt, ft, notes, relative_floor=True, floor_offset_db=30.0)
        assert len(kept) == 3

    def test_relative_floor_rejects_very_weak(self) -> None:
        """A note far below median should still be rejected."""
        cqt, ft = _make_cqt(100)
        _set_energy(cqt, ft, pitch=60, start_sec=0.0, end_sec=0.5, energy_db=-10.0)
        _set_energy(cqt, ft, pitch=64, start_sec=0.0, end_sec=0.5, energy_db=-10.0)
        # This note is at -75 dB, 65 dB below median of -10
        _set_energy(cqt, ft, pitch=67, start_sec=0.0, end_sec=0.5, energy_db=-75.0)

        notes = [
            _note(pitch=60, start=0.0, end=0.5),
            _note(pitch=64, start=0.0, end=0.5),
            _note(pitch=67, start=0.0, end=0.5),
        ]
        kept, rejected = validate_notes(cqt, ft, notes, relative_floor=True, floor_offset_db=30.0)
        kept_pitches = {n.pitch for n in kept}
        assert 60 in kept_pitches
        assert 64 in kept_pitches
        assert 67 not in kept_pitches

    def test_rejects_overtone_of_louder_fundamental(self) -> None:
        cqt, ft = _make_cqt(100)
        _set_energy(cqt, ft, pitch=48, start_sec=0.0, end_sec=0.5, energy_db=-5.0)
        _set_energy(cqt, ft, pitch=60, start_sec=0.0, end_sec=0.5, energy_db=-18.0)

        notes = [
            _note(pitch=48, start=0.0, end=0.5, amp=0.9),
            _note(pitch=60, start=0.0, end=0.5, amp=0.4),
        ]
        kept, rejected = validate_notes(cqt, ft, notes, relative_floor=False,
                                         energy_floor_db=-50.0, overtone_margin_db=6.0)
        kept_pitches = {n.pitch for n in kept}
        rejected_pitches = {n.pitch for n in rejected}
        assert 48 in kept_pitches
        assert 60 in rejected_pitches

    def test_keeps_genuinely_played_octave(self) -> None:
        cqt, ft = _make_cqt(100)
        _set_energy(cqt, ft, pitch=48, start_sec=0.0, end_sec=0.5, energy_db=-5.0)
        _set_energy(cqt, ft, pitch=60, start_sec=0.0, end_sec=0.5, energy_db=-3.0)

        notes = [
            _note(pitch=48, start=0.0, end=0.5, amp=0.9),
            _note(pitch=60, start=0.0, end=0.5, amp=0.8),
        ]
        kept, rejected = validate_notes(cqt, ft, notes, relative_floor=False,
                                         energy_floor_db=-50.0, overtone_margin_db=6.0)
        kept_pitches = {n.pitch for n in kept}
        assert 48 in kept_pitches
        assert 60 in kept_pitches

    def test_keeps_chord_with_harmonic_intervals(self) -> None:
        cqt, ft = _make_cqt(100)
        chord_pitches = [48, 55, 60, 64, 67]
        for p in chord_pitches:
            _set_energy(cqt, ft, pitch=p, start_sec=0.0, end_sec=0.5, energy_db=-8.0)

        notes = [_note(pitch=p, start=0.0, end=0.5, amp=0.7) for p in chord_pitches]
        kept, rejected = validate_notes(cqt, ft, notes, relative_floor=False,
                                         energy_floor_db=-50.0, overtone_margin_db=6.0)
        kept_pitches = sorted(n.pitch for n in kept)
        assert kept_pitches == chord_pitches

    def test_validates_exact_pitches(self) -> None:
        cqt, ft = _make_cqt(100)
        _set_energy(cqt, ft, pitch=60, start_sec=0.0, end_sec=0.5, energy_db=-10.0)
        _set_energy(cqt, ft, pitch=64, start_sec=0.0, end_sec=0.5, energy_db=-12.0)

        notes = [
            _note(pitch=60, start=0.0, end=0.5),
            _note(pitch=64, start=0.0, end=0.5),
            _note(pitch=81, start=0.0, end=0.5),  # no energy
        ]
        kept, rejected = validate_notes(cqt, ft, notes, relative_floor=False, energy_floor_db=-50.0)
        assert sorted(n.pitch for n in kept) == [60, 64]
        assert rejected[0].pitch == 81

    def test_empty_input(self) -> None:
        cqt, ft = _make_cqt(50)
        kept, rejected = validate_notes(cqt, ft, [])
        assert kept == []
        assert rejected == []

    def test_non_overlapping_notes_not_treated_as_overtones(self) -> None:
        cqt, ft = _make_cqt(200)
        _set_energy(cqt, ft, pitch=48, start_sec=0.0, end_sec=0.5, energy_db=-5.0)
        _set_energy(cqt, ft, pitch=60, start_sec=1.0, end_sec=1.5, energy_db=-10.0)

        notes = [
            _note(pitch=48, start=0.0, end=0.5),
            _note(pitch=60, start=1.0, end=1.5),
        ]
        kept, rejected = validate_notes(cqt, ft, notes, relative_floor=False, energy_floor_db=-50.0)
        assert len(kept) == 2


class TestWalkSustain:
    def test_walks_until_energy_drops(self) -> None:
        cqt, ft = _make_cqt(100, frame_duration=0.023)
        b = _pitch_to_bin(60)
        cqt[b, 0:50] = -10.0
        cqt[b, 50:] = -80.0

        dur = walk_sustain(cqt, 60, start_frame=0, frame_times=ft)
        expected = ft[50] - ft[0]
        assert dur == pytest.approx(expected, abs=0.001)

    def test_walks_to_end_if_sustained(self) -> None:
        cqt, ft = _make_cqt(100, frame_duration=0.023)
        b = _pitch_to_bin(60)
        cqt[b, :] = -10.0

        dur = walk_sustain(cqt, 60, start_frame=0, frame_times=ft)
        expected = ft[-1] - ft[0]
        assert dur == pytest.approx(expected, abs=0.001)

    def test_zero_duration_for_silent_onset(self) -> None:
        cqt, ft = _make_cqt(100)
        dur = walk_sustain(cqt, 60, start_frame=0, frame_times=ft, min_energy_db=-50.0)
        assert dur == 0.0

    def test_out_of_range_pitch(self) -> None:
        cqt, ft = _make_cqt(100)
        dur = walk_sustain(cqt, 10, start_frame=0, frame_times=ft)
        assert dur == 0.0


class TestDetectOnsetsAtBin:
    def test_detects_sudden_rise(self) -> None:
        cqt, ft = _make_cqt(50)
        b = _pitch_to_bin(60)
        cqt[b, 10:30] = -10.0

        onsets = _detect_onsets_at_bin(cqt, b, onset_rise_db=10.0, min_energy_db=-25.0)
        assert len(onsets) >= 1
        assert any(abs(f - 10) <= 1 for f in onsets)

    def test_no_onset_for_gradual_rise(self) -> None:
        cqt, ft = _make_cqt(50)
        b = _pitch_to_bin(60)
        for f in range(50):
            cqt[b, f] = -80.0 + (70.0 * f / 49.0)

        onsets = _detect_onsets_at_bin(cqt, b, onset_rise_db=15.0, min_energy_db=-25.0)
        assert len(onsets) == 0

    def test_no_onset_below_energy_threshold(self) -> None:
        cqt, ft = _make_cqt(50)
        b = _pitch_to_bin(60)
        cqt[b, 10:30] = -50.0

        onsets = _detect_onsets_at_bin(cqt, b, onset_rise_db=10.0, min_energy_db=-25.0)
        assert len(onsets) == 0


class TestFindSustainedRegions:
    def test_finds_sustained_region(self) -> None:
        cqt, ft = _make_cqt(100, frame_duration=0.023)
        b = _pitch_to_bin(60)
        cqt[b, 10:50] = -10.0  # ~0.92s sustained

        regions = _find_sustained_regions(cqt, b, ft, min_energy_db=-20.0, min_duration_sec=0.1)
        assert len(regions) >= 1
        start_f, end_f = regions[0]
        assert start_f == 10
        assert end_f == 50

    def test_ignores_short_spike(self) -> None:
        cqt, ft = _make_cqt(100, frame_duration=0.023)
        b = _pitch_to_bin(60)
        cqt[b, 10:12] = -10.0  # ~46ms, below 100ms threshold

        regions = _find_sustained_regions(cqt, b, ft, min_energy_db=-20.0, min_duration_sec=0.1)
        assert len(regions) == 0

    def test_finds_multiple_regions(self) -> None:
        cqt, ft = _make_cqt(100, frame_duration=0.023)
        b = _pitch_to_bin(60)
        cqt[b, 10:30] = -10.0
        cqt[b, 50:70] = -10.0

        regions = _find_sustained_regions(cqt, b, ft, min_energy_db=-20.0, min_duration_sec=0.1)
        assert len(regions) == 2

    def test_catches_swelling_note(self) -> None:
        """A note that gradually rises above threshold should be found."""
        cqt, ft = _make_cqt(100, frame_duration=0.023)
        b = _pitch_to_bin(60)
        # Gradual swell from -30 to -5 over frames 10-50
        for f in range(10, 50):
            cqt[b, f] = -30.0 + (25.0 * (f - 10) / 39.0)

        regions = _find_sustained_regions(cqt, b, ft, min_energy_db=-20.0, min_duration_sec=0.1)
        # Should find the portion above -20 dB
        assert len(regions) >= 1


class TestRecoverNotes:
    def test_recovers_unoccupied_sustained_energy(self) -> None:
        """Sustained energy at an unoccupied pitch should be recovered."""
        cqt, ft = _make_cqt(100)
        b = _pitch_to_bin(64)
        cqt[b, 20:60] = -10.0

        existing = []
        recovered = recover_notes(cqt, ft, existing, ref_energy=-10.0,
                                   recovery_threshold_offset_db=20.0,
                                   target_pitches={64})
        recovered_pitches = [n.pitch for n in recovered]
        assert 64 in recovered_pitches

    def test_does_not_recover_already_covered(self) -> None:
        cqt, ft = _make_cqt(100)
        b = _pitch_to_bin(60)
        cqt[b, 10:60] = -10.0

        existing = [_note(pitch=60, start=0.0, end=1.5)]
        recovered = recover_notes(cqt, ft, existing, ref_energy=-10.0,
                                   target_pitches={60})
        recovered_pitches = [n.pitch for n in recovered]
        assert 60 not in recovered_pitches

    def test_blind_spot_lower_threshold(self) -> None:
        """Known blind spots should use lower thresholds."""
        cqt, ft = _make_cqt(100)
        for pitch in [71, 66]:
            b = _pitch_to_bin(pitch)
            cqt[b, 20:60] = -28.0  # below -10 - 20 = -30 but above -30 - 8 = -38

        existing = []
        recovered = recover_notes(
            cqt, ft, existing,
            ref_energy=-10.0,
            recovery_threshold_offset_db=20.0,
            blind_spot_boost_db=8.0,
            target_pitches={66, 71},
        )
        recovered_pitches = set(n.pitch for n in recovered)
        assert 71 in recovered_pitches
        assert 66 in recovered_pitches

    def test_blind_spot_exact_pitches(self) -> None:
        assert 66 in _BLIND_SPOT_PITCHES
        assert 71 in _BLIND_SPOT_PITCHES

    def test_respects_min_duration(self) -> None:
        cqt, ft = _make_cqt(100, frame_duration=0.023)
        b = _pitch_to_bin(64)
        cqt[b, 10] = -10.0  # single frame ~23ms

        existing = []
        recovered = recover_notes(cqt, ft, existing, ref_energy=-10.0,
                                   min_duration_sec=0.08, target_pitches={64})
        assert len(recovered) == 0

    def test_recovered_note_has_correct_pitch(self) -> None:
        cqt, ft = _make_cqt(100)
        b = _pitch_to_bin(67)
        cqt[b, 20:50] = -10.0

        recovered = recover_notes(cqt, ft, [], ref_energy=-10.0,
                                   target_pitches={67})
        matching = [n for n in recovered if n.pitch == 67]
        assert len(matching) >= 1

    def test_empty_cqt(self) -> None:
        cqt = np.full((_N_BINS, 0), -80.0)
        ft = np.array([])
        assert recover_notes(cqt, ft, []) == []


class TestResolveOverlaps:
    def test_trims_earlier_note_on_overlap(self) -> None:
        notes = [
            _note(pitch=60, start=0.0, end=1.0, amp=0.8),
            _note(pitch=60, start=0.7, end=1.5, amp=0.7),
        ]
        result = resolve_overlaps(notes)
        assert len(result) == 2
        first = sorted(result, key=lambda n: n.start_sec)[0]
        assert first.end_sec == pytest.approx(0.7)

    def test_near_simultaneous_keeps_louder(self) -> None:
        notes = [
            _note(pitch=60, start=0.0, end=0.5, amp=0.5),
            _note(pitch=60, start=0.02, end=0.6, amp=0.9),
        ]
        result = resolve_overlaps(notes, onset_tol=0.03)
        assert len(result) == 1
        assert result[0].amplitude == pytest.approx(0.9)

    def test_no_overlap_untouched(self) -> None:
        notes = [
            _note(pitch=60, start=0.0, end=0.5, amp=0.8),
            _note(pitch=60, start=0.6, end=1.0, amp=0.7),
        ]
        result = resolve_overlaps(notes)
        assert len(result) == 2

    def test_different_pitches_not_affected(self) -> None:
        notes = [
            _note(pitch=60, start=0.0, end=1.0, amp=0.8),
            _note(pitch=64, start=0.5, end=1.5, amp=0.7),
        ]
        result = resolve_overlaps(notes)
        assert len(result) == 2

    def test_triple_overlap_cascade(self) -> None:
        notes = [
            _note(pitch=60, start=0.0, end=1.0, amp=0.8),
            _note(pitch=60, start=0.5, end=1.5, amp=0.7),
            _note(pitch=60, start=1.2, end=2.0, amp=0.6),
        ]
        result = resolve_overlaps(notes)
        result_sorted = sorted(result, key=lambda n: n.start_sec)
        assert len(result_sorted) == 3
        assert result_sorted[0].end_sec == pytest.approx(0.5)
        assert result_sorted[1].end_sec == pytest.approx(1.2)
        assert result_sorted[2].end_sec == pytest.approx(2.0)

    def test_exact_same_note_keeps_louder(self) -> None:
        notes = [
            _note(pitch=60, start=0.0, end=0.5, amp=0.4),
            _note(pitch=60, start=0.0, end=0.5, amp=0.8),
        ]
        result = resolve_overlaps(notes, onset_tol=0.03)
        assert len(result) == 1
        assert result[0].amplitude == pytest.approx(0.8)

    def test_empty_input(self) -> None:
        assert resolve_overlaps([]) == []

    def test_single_note(self) -> None:
        result = resolve_overlaps([_note()])
        assert len(result) == 1

    def test_preserves_correct_pitches_after_resolution(self) -> None:
        notes = [
            _note(pitch=60, start=0.0, end=1.0),
            _note(pitch=60, start=0.5, end=1.5),
            _note(pitch=64, start=0.0, end=1.0),
            _note(pitch=64, start=0.8, end=1.8),
        ]
        result = resolve_overlaps(notes)
        pitches = sorted(n.pitch for n in result)
        assert pitches == [60, 60, 64, 64]


class TestComputeCqt:
    def test_returns_correct_shape(self) -> None:
        sr = 22050
        duration = 1.0
        samples = np.random.randn(int(sr * duration)).astype(np.float32) * 0.01
        audio = AudioData(
            samples=samples, sample_rate=sr,
            duration_sec=duration, file_path="test.wav",
        )
        cqt_mag, ft = compute_cqt(audio)
        assert cqt_mag.shape[0] == _N_BINS
        assert len(ft) == cqt_mag.shape[1]

    def test_sine_wave_peaks_at_correct_bin(self) -> None:
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        freq = 440.0  # A4 = MIDI 69
        samples = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)
        audio = AudioData(
            samples=samples, sample_rate=sr,
            duration_sec=duration, file_path="test.wav",
        )
        cqt_mag, ft = compute_cqt(audio)

        mean_energy = cqt_mag.mean(axis=1)
        peak_bin = int(np.argmax(mean_energy))
        expected_bin = 69 - _MIDI_LOW

        assert abs(peak_bin - expected_bin) <= 1


class TestSpectralValidateIntegration:
    def test_validates_real_note_rejects_phantom(self) -> None:
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        samples = (np.sin(2 * np.pi * 440.0 * t) * 0.5).astype(np.float32)
        audio = AudioData(
            samples=samples, sample_rate=sr,
            duration_sec=duration, file_path="test.wav",
        )

        notes = [
            _note(pitch=69, start=0.0, end=1.0, amp=0.8),
            _note(pitch=50, start=0.0, end=1.0, amp=0.5),
        ]

        result = spectral_validate(
            audio, notes,
            do_recover=False,
            do_resolve_overlaps=False,
        )

        result_pitches = [n.pitch for n in result]
        assert 69 in result_pitches
        assert 50 not in result_pitches

    def test_resolves_overlap_in_full_pipeline(self) -> None:
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        samples = (np.sin(2 * np.pi * 440.0 * t) * 0.5).astype(np.float32)
        audio = AudioData(
            samples=samples, sample_rate=sr,
            duration_sec=duration, file_path="test.wav",
        )

        notes = [
            _note(pitch=69, start=0.0, end=1.5, amp=0.8),
            _note(pitch=69, start=1.0, end=2.0, amp=0.7),
        ]

        result = spectral_validate(
            audio, notes,
            do_recover=False,
            do_resolve_overlaps=True,
        )

        a69 = sorted([n for n in result if n.pitch == 69], key=lambda n: n.start_sec)
        if len(a69) == 2:
            assert a69[0].end_sec <= a69[1].start_sec + 0.001

    def test_chord_preservation(self) -> None:
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        freqs = [261.63, 329.63, 392.0]  # C4, E4, G4
        samples = sum(
            np.sin(2 * np.pi * f * t) * 0.3 for f in freqs
        ).astype(np.float32)
        audio = AudioData(
            samples=samples, sample_rate=sr,
            duration_sec=duration, file_path="test.wav",
        )

        notes = [
            _note(pitch=60, start=0.0, end=1.0, amp=0.7),
            _note(pitch=64, start=0.0, end=1.0, amp=0.7),
            _note(pitch=67, start=0.0, end=1.0, amp=0.7),
        ]

        result = spectral_validate(
            audio, notes,
            do_recover=False,
            do_resolve_overlaps=False,
        )

        result_pitches = sorted(n.pitch for n in result)
        assert result_pitches == [60, 64, 67]
