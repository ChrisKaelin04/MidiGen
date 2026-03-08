# MidiGen

A standalone desktop GUI application for converting audio files (MP3, WAV, FLAC) to MIDI using machine learning transcription. Focused on electronic dance music production workflows.

## Features

- Polyphonic audio-to-MIDI transcription via [Basic Pitch](https://github.com/spotify/basic-pitch) (Spotify)
- BPM auto-detection and manual entry with 1/16th note quantization grid
- Configurable note range filter (displayed as note names, not raw Hz)
- Adjustable ML confidence threshold
- Ghost note filtering and dynamic/fixed velocity options
- Start/end time selection via waveform spectrogram view
- Read-only MIDI piano roll preview
- Export as .mid file, open with system default, or drag into FL Studio

## Requirements

- Python 3.10+
- See `requirements.txt` for Python dependencies
- Linux only: `ffmpeg` required for MP3 support (`sudo apt install ffmpeg`)

## Running

**Windows:**
```
run.bat
```

**Linux:**
```
bash run.sh
```

Or directly:
```
pip install -r requirements.txt
python main.py
```

## Project Structure

See `PROMPT_FOR_OPUS.md` for the full architecture specification.
See `CHECKLIST.md` for implementation progress tracking.

## Testing

```
pytest tests/
```

To add a test fixture (audio + expected MIDI pair):
```
python tests/add_fixture.py --audio path/to/clip.mp3 --midi path/to/expected.mid
```
