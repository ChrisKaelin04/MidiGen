"""Load audio files and return AudioData for the processing pipeline."""

from pathlib import Path

import librosa
import numpy as np

from core import AudioData


_SUPPORTED_EXTENSIONS = {'.mp3', '.wav', '.flac'}
_DEFAULT_SR = 22050


def load_audio(file_path: Path, start_sec: float = 0.0, end_sec: float | None = None) -> AudioData:
    """Load an audio file, convert to mono, and optionally trim to a time range.

    Args:
        file_path: Path to the audio file (MP3, WAV, or FLAC).
        start_sec: Start time in seconds for trimming. Defaults to 0.
        end_sec: End time in seconds for trimming. None means end of file.

    Returns:
        AudioData with mono float32 samples at 22050 Hz.

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if file_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format '{file_path.suffix}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
        )

    # librosa loads as mono float32 by default
    samples, sr = librosa.load(str(file_path), sr=_DEFAULT_SR, mono=True)

    # Trim to selection if specified
    if start_sec > 0.0 or end_sec is not None:
        start_sample = int(start_sec * sr)
        end_sample = int(end_sec * sr) if end_sec is not None else len(samples)
        start_sample = max(0, min(start_sample, len(samples)))
        end_sample = max(start_sample, min(end_sample, len(samples)))
        samples = samples[start_sample:end_sample]

    duration_sec = len(samples) / sr

    return AudioData(
        samples=samples,
        sample_rate=sr,
        duration_sec=duration_sec,
        file_path=file_path,
    )


def apply_hpss(audio: AudioData) -> AudioData:
    """Apply Harmonic/Percussive Source Separation, returning only harmonics.

    Strips percussive transients (drums, clicks, noise) from the audio,
    leaving melodic/harmonic content for cleaner transcription.

    Args:
        audio: AudioData to process.

    Returns:
        New AudioData containing only the harmonic component.
    """
    stft = librosa.stft(audio.samples)
    harmonic_stft, _ = librosa.decompose.hpss(stft)
    harmonic_samples = librosa.istft(harmonic_stft, length=len(audio.samples))

    return AudioData(
        samples=harmonic_samples.astype(np.float32),
        sample_rate=audio.sample_rate,
        duration_sec=audio.duration_sec,
        file_path=audio.file_path,
    )


def detect_bpm(audio: AudioData) -> float:
    """Detect BPM of audio using librosa's beat tracker.

    Args:
        audio: AudioData to analyze.

    Returns:
        Estimated BPM as a float.
    """
    tempo, _ = librosa.beat.beat_track(y=audio.samples, sr=audio.sample_rate)
    # librosa may return an ndarray with one element or a scalar
    if isinstance(tempo, np.ndarray):
        return float(tempo[0])
    return float(tempo)
