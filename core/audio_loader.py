"""Load audio files and return AudioData for the processing pipeline.

Includes preprocessing functions that run BEFORE transcription:
- HPSS: strip percussive transients (legacy — Demucs preferred for real tracks)
- Loudness normalization: consistent input level for basic-pitch
- Noise gate: clean up reverb tails and low-level noise
- Pre-emphasis EQ: gentle boost in frequency ranges where basic-pitch is weak
"""

from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, sosfilt

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


def normalize_loudness(audio: AudioData, target_lufs: float = -14.0) -> AudioData:
    """Normalize audio loudness to a consistent level.

    Uses RMS-based normalization as an approximation of LUFS. This gives
    basic-pitch a consistent input level regardless of how loud or quiet
    the source recording is, which helps with confidence thresholds.

    Args:
        audio: AudioData to normalize.
        target_lufs: Target loudness in approximate LUFS. Default -14.0
            is a good level for transcription (not too hot, not too quiet).

    Returns:
        New AudioData with normalized loudness.
    """
    samples = audio.samples
    rms = np.sqrt(np.mean(samples ** 2))
    if rms < 1e-8:
        return audio  # silence, nothing to normalize

    # Convert target LUFS to linear RMS (approximate: LUFS ≈ 20*log10(RMS) + 3.01)
    target_rms = 10 ** ((target_lufs - 3.01) / 20.0)
    gain = target_rms / rms

    # Apply gain with clipping protection
    normalized = np.clip(samples * gain, -1.0, 1.0).astype(np.float32)

    return AudioData(
        samples=normalized,
        sample_rate=audio.sample_rate,
        duration_sec=audio.duration_sec,
        file_path=audio.file_path,
    )


def apply_noise_gate(
    audio: AudioData,
    threshold_db: float = -40.0,
    attack_ms: float = 5.0,
    release_ms: float = 50.0,
) -> AudioData:
    """Apply a noise gate to silence low-level content.

    Attenuates audio frames below the threshold to zero, cleaning up
    reverb tails, background noise, and bleed that can confuse
    transcription. Uses simple envelope following with attack/release
    smoothing to avoid clicks.

    Args:
        audio: AudioData to gate.
        threshold_db: Gate threshold in dB. Frames below this are silenced.
        attack_ms: Attack time in ms (how fast the gate opens).
        release_ms: Release time in ms (how fast the gate closes).

    Returns:
        New AudioData with noise gate applied.
    """
    samples = audio.samples
    sr = audio.sample_rate

    # Compute frame-level RMS envelope
    frame_length = int(sr * 0.01)  # 10ms frames
    hop = frame_length // 2

    # Compute per-sample amplitude envelope using a sliding RMS
    envelope = np.zeros_like(samples)
    for i in range(0, len(samples) - frame_length, hop):
        rms = np.sqrt(np.mean(samples[i:i + frame_length] ** 2))
        rms_db = 20 * np.log10(rms + 1e-10)
        # Binary gate: open (1.0) or closed (0.0)
        gate_val = 1.0 if rms_db >= threshold_db else 0.0
        envelope[i:i + hop] = gate_val

    # Fill any remaining samples
    remainder_start = len(samples) - (len(samples) % hop)
    if remainder_start < len(samples):
        envelope[remainder_start:] = envelope[max(0, remainder_start - 1)]

    # Smooth the envelope with attack/release to avoid clicks
    attack_samples = max(1, int(sr * attack_ms / 1000.0))
    release_samples = max(1, int(sr * release_ms / 1000.0))

    smoothed = np.zeros_like(envelope)
    smoothed[0] = envelope[0]
    for i in range(1, len(envelope)):
        if envelope[i] > smoothed[i - 1]:
            # Gate opening (attack)
            coeff = 1.0 - np.exp(-1.0 / attack_samples)
            smoothed[i] = smoothed[i - 1] + coeff * (envelope[i] - smoothed[i - 1])
        else:
            # Gate closing (release)
            coeff = 1.0 - np.exp(-1.0 / release_samples)
            smoothed[i] = smoothed[i - 1] + coeff * (envelope[i] - smoothed[i - 1])

    gated = (samples * smoothed).astype(np.float32)

    return AudioData(
        samples=gated,
        sample_rate=audio.sample_rate,
        duration_sec=audio.duration_sec,
        file_path=audio.file_path,
    )


def apply_pre_emphasis(
    audio: AudioData,
    boost_db: float = 1.5,
    low_hz: float = 440.0,
    high_hz: float = 520.0,
) -> AudioData:
    """Apply a gentle EQ boost in a target frequency range.

    basic-pitch has known weak spots around B4 (494 Hz) and F#4 (370 Hz).
    A subtle boost in these regions can push borderline detections above
    the confidence threshold without distorting the overall spectrum.

    Uses a bandpass filter to isolate the target band, scales it, and
    adds it back to the original signal.

    Args:
        audio: AudioData to process.
        boost_db: Amount of boost in dB. Keep this small (1-2 dB).
        low_hz: Lower edge of the boost band in Hz.
        high_hz: Upper edge of the boost band in Hz.

    Returns:
        New AudioData with pre-emphasis applied.
    """
    samples = audio.samples
    sr = audio.sample_rate
    nyquist = sr / 2.0

    # Validate frequency range
    if low_hz >= high_hz or high_hz >= nyquist:
        return audio

    # Design bandpass filter (second-order sections for stability)
    sos = butter(
        N=2,
        Wn=[low_hz / nyquist, high_hz / nyquist],
        btype='bandpass',
        output='sos',
    )

    # Extract the band
    band = sosfilt(sos, samples).astype(np.float32)

    # Compute linear gain from dB
    gain = 10 ** (boost_db / 20.0) - 1.0  # subtract 1 because we ADD to original

    # Add boosted band back to original
    emphasized = np.clip(samples + band * gain, -1.0, 1.0).astype(np.float32)

    return AudioData(
        samples=emphasized,
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
