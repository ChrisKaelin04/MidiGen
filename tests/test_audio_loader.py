"""Tests for core.audio_loader — loading, trimming, format support, and preprocessing."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from core import AudioData
from core.audio_loader import (
    load_audio,
    normalize_loudness,
    apply_noise_gate,
    apply_pre_emphasis,
)


@pytest.fixture
def synthetic_wav(tmp_path: Path) -> Path:
    """Create a short synthetic WAV file (1 second, 22050 Hz, mono)."""
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    # 440 Hz sine wave
    samples = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    path = tmp_path / "test.wav"
    sf.write(str(path), samples, sr)
    return path


@pytest.fixture
def stereo_wav(tmp_path: Path) -> Path:
    """Create a short stereo WAV file."""
    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    left = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    right = np.sin(2 * np.pi * 880 * t).astype(np.float32)
    stereo = np.column_stack([left, right])
    path = tmp_path / "stereo.wav"
    sf.write(str(path), stereo, sr)
    return path


def _make_audio(samples: np.ndarray, sr: int = 22050) -> AudioData:
    """Helper to create AudioData from a numpy array."""
    return AudioData(
        samples=samples.astype(np.float32),
        sample_rate=sr,
        duration_sec=len(samples) / sr,
        file_path=Path("test.wav"),
    )


class TestLoadAudio:
    def test_load_wav_returns_audio_data(self, synthetic_wav: Path) -> None:
        audio = load_audio(synthetic_wav)
        assert audio.sample_rate == 22050
        assert audio.samples.ndim == 1  # mono
        assert audio.duration_sec > 0
        assert audio.file_path == synthetic_wav

    def test_load_wav_duration(self, synthetic_wav: Path) -> None:
        audio = load_audio(synthetic_wav)
        assert abs(audio.duration_sec - 1.0) < 0.05

    def test_trim_start_end(self, synthetic_wav: Path) -> None:
        audio = load_audio(synthetic_wav, start_sec=0.25, end_sec=0.75)
        assert abs(audio.duration_sec - 0.5) < 0.05

    def test_trim_start_only(self, synthetic_wav: Path) -> None:
        audio = load_audio(synthetic_wav, start_sec=0.5)
        assert abs(audio.duration_sec - 0.5) < 0.05

    def test_stereo_to_mono(self, stereo_wav: Path) -> None:
        audio = load_audio(stereo_wav)
        assert audio.samples.ndim == 1

    def test_unsupported_format(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "test.ogg"
        bad_file.write_text("not audio")
        with pytest.raises(ValueError, match="Unsupported audio format"):
            load_audio(bad_file)

    def test_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_audio(tmp_path / "nonexistent.wav")

    def test_samples_are_float(self, synthetic_wav: Path) -> None:
        audio = load_audio(synthetic_wav)
        assert audio.samples.dtype == np.float32


class TestNormalizeLoudness:
    def test_output_is_audio_data(self) -> None:
        samples = np.sin(np.linspace(0, 2 * np.pi * 440, 22050)).astype(np.float32) * 0.5
        audio = _make_audio(samples)
        result = normalize_loudness(audio)
        assert isinstance(result, AudioData)
        assert result.sample_rate == audio.sample_rate
        assert result.duration_sec == audio.duration_sec

    def test_quiet_signal_gets_louder(self) -> None:
        samples = np.sin(np.linspace(0, 2 * np.pi * 440, 22050)).astype(np.float32) * 0.01
        audio = _make_audio(samples)
        result = normalize_loudness(audio, target_lufs=-14.0)
        assert np.sqrt(np.mean(result.samples ** 2)) > np.sqrt(np.mean(samples ** 2))

    def test_loud_signal_gets_quieter(self) -> None:
        samples = np.sin(np.linspace(0, 2 * np.pi * 440, 22050)).astype(np.float32) * 0.95
        audio = _make_audio(samples)
        result = normalize_loudness(audio, target_lufs=-20.0)
        assert np.sqrt(np.mean(result.samples ** 2)) < np.sqrt(np.mean(samples ** 2))

    def test_silence_unchanged(self) -> None:
        samples = np.zeros(22050, dtype=np.float32)
        audio = _make_audio(samples)
        result = normalize_loudness(audio)
        assert np.allclose(result.samples, 0.0)

    def test_output_clipped_to_range(self) -> None:
        samples = np.sin(np.linspace(0, 2 * np.pi * 440, 22050)).astype(np.float32) * 0.001
        audio = _make_audio(samples)
        result = normalize_loudness(audio, target_lufs=-5.0)
        assert np.all(result.samples >= -1.0)
        assert np.all(result.samples <= 1.0)

    def test_dtype_preserved(self) -> None:
        samples = np.sin(np.linspace(0, 2 * np.pi * 440, 22050)).astype(np.float32) * 0.5
        audio = _make_audio(samples)
        result = normalize_loudness(audio)
        assert result.samples.dtype == np.float32

    def test_different_targets_produce_different_levels(self) -> None:
        samples = np.sin(np.linspace(0, 2 * np.pi * 440, 22050)).astype(np.float32) * 0.5
        audio = _make_audio(samples)
        loud = normalize_loudness(audio, target_lufs=-10.0)
        quiet = normalize_loudness(audio, target_lufs=-25.0)
        rms_loud = np.sqrt(np.mean(loud.samples ** 2))
        rms_quiet = np.sqrt(np.mean(quiet.samples ** 2))
        assert rms_loud > rms_quiet


class TestNoiseGate:
    def test_output_is_audio_data(self) -> None:
        samples = np.sin(np.linspace(0, 2 * np.pi * 440, 22050)).astype(np.float32) * 0.5
        audio = _make_audio(samples)
        result = apply_noise_gate(audio)
        assert isinstance(result, AudioData)

    def test_loud_signal_passes_through(self) -> None:
        samples = np.sin(np.linspace(0, 2 * np.pi * 440, 22050)).astype(np.float32) * 0.5
        audio = _make_audio(samples)
        result = apply_noise_gate(audio, threshold_db=-40.0)
        # Signal at 0.5 amplitude is about -6 dB, well above -40 dB
        rms_in = np.sqrt(np.mean(samples ** 2))
        rms_out = np.sqrt(np.mean(result.samples ** 2))
        assert rms_out > rms_in * 0.9

    def test_quiet_signal_attenuated(self) -> None:
        samples = np.sin(np.linspace(0, 2 * np.pi * 440, 22050)).astype(np.float32) * 0.001
        audio = _make_audio(samples)
        result = apply_noise_gate(audio, threshold_db=-20.0)
        # Signal at 0.001 amplitude is about -60 dB, below -20 dB threshold
        rms_out = np.sqrt(np.mean(result.samples ** 2))
        rms_in = np.sqrt(np.mean(samples ** 2))
        assert rms_out < rms_in * 0.5

    def test_mixed_signal_gates_quiet_parts(self) -> None:
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        # First half loud, second half very quiet
        loud = np.sin(2 * np.pi * 440 * t[:sr // 2]) * 0.5
        quiet = np.sin(2 * np.pi * 440 * t[sr // 2:]) * 0.0005
        samples = np.concatenate([loud, quiet]).astype(np.float32)
        audio = _make_audio(samples)
        result = apply_noise_gate(audio, threshold_db=-30.0)
        # Second half should be heavily attenuated
        second_half_rms = np.sqrt(np.mean(result.samples[sr // 2:] ** 2))
        first_half_rms = np.sqrt(np.mean(result.samples[:sr // 2] ** 2))
        assert second_half_rms < first_half_rms * 0.2

    def test_dtype_preserved(self) -> None:
        samples = np.sin(np.linspace(0, 2 * np.pi * 440, 22050)).astype(np.float32) * 0.5
        audio = _make_audio(samples)
        result = apply_noise_gate(audio)
        assert result.samples.dtype == np.float32


class TestPreEmphasis:
    def test_output_is_audio_data(self) -> None:
        samples = np.sin(np.linspace(0, 2 * np.pi * 440, 22050)).astype(np.float32) * 0.5
        audio = _make_audio(samples)
        result = apply_pre_emphasis(audio)
        assert isinstance(result, AudioData)

    def test_boosts_target_frequency(self) -> None:
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        # Signal at 480 Hz (inside default 440-520 Hz boost range)
        samples = (np.sin(2 * np.pi * 480 * t) * 0.3).astype(np.float32)
        audio = _make_audio(samples, sr=sr)
        result = apply_pre_emphasis(audio, boost_db=2.0, low_hz=440.0, high_hz=520.0)
        # The boosted signal should have more energy
        rms_in = np.sqrt(np.mean(samples ** 2))
        rms_out = np.sqrt(np.mean(result.samples ** 2))
        assert rms_out > rms_in

    def test_does_not_boost_outside_band(self) -> None:
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        # Signal at 200 Hz (outside 440-520 Hz range)
        samples = (np.sin(2 * np.pi * 200 * t) * 0.3).astype(np.float32)
        audio = _make_audio(samples, sr=sr)
        result = apply_pre_emphasis(audio, boost_db=2.0, low_hz=440.0, high_hz=520.0)
        # Should be nearly identical (bandpass filter won't extract 200 Hz)
        rms_in = np.sqrt(np.mean(samples ** 2))
        rms_out = np.sqrt(np.mean(result.samples ** 2))
        assert abs(rms_out - rms_in) / rms_in < 0.05

    def test_small_boost_is_subtle(self) -> None:
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        samples = (np.sin(2 * np.pi * 480 * t) * 0.3).astype(np.float32)
        audio = _make_audio(samples, sr=sr)
        result = apply_pre_emphasis(audio, boost_db=1.5, low_hz=440.0, high_hz=520.0)
        # 1.5 dB boost should be subtle — less than 20% RMS increase
        rms_in = np.sqrt(np.mean(samples ** 2))
        rms_out = np.sqrt(np.mean(result.samples ** 2))
        ratio = rms_out / rms_in
        assert 1.0 < ratio < 1.25

    def test_output_clipped_to_range(self) -> None:
        sr = 22050
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        samples = (np.sin(2 * np.pi * 480 * t) * 0.99).astype(np.float32)
        audio = _make_audio(samples, sr=sr)
        result = apply_pre_emphasis(audio, boost_db=6.0, low_hz=440.0, high_hz=520.0)
        assert np.all(result.samples >= -1.0)
        assert np.all(result.samples <= 1.0)

    def test_invalid_frequency_range_returns_unchanged(self) -> None:
        samples = np.sin(np.linspace(0, 2 * np.pi * 440, 22050)).astype(np.float32) * 0.5
        audio = _make_audio(samples)
        result = apply_pre_emphasis(audio, low_hz=600.0, high_hz=400.0)
        assert np.allclose(result.samples, audio.samples)

    def test_dtype_preserved(self) -> None:
        samples = np.sin(np.linspace(0, 2 * np.pi * 440, 22050)).astype(np.float32) * 0.5
        audio = _make_audio(samples)
        result = apply_pre_emphasis(audio)
        assert result.samples.dtype == np.float32
