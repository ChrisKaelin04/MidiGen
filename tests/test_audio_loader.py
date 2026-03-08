"""Tests for core.audio_loader — loading, trimming, and format support."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from core.audio_loader import load_audio


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
