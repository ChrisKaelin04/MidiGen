# MidiGen — Implementation Prompt for Claude Opus

## Instructions for Opus

Generate all files completely — do not truncate, summarize, or use placeholder comments like `# ... rest of code`. If the full response is too long for one reply, stop at a clean file boundary and wait for the user to say "continue" before proceeding with the next file.

## Project Overview

Build a standalone, cross-platform (Windows/Linux) desktop GUI application called **MidiGen** that converts audio files (MP3, WAV, FLAC) to MIDI using machine learning transcription. The primary use case is electronic dance music (EDM) production. The user is the primary developer and will extend the app over time — code must be modular, clean, and independently testable per module.

---

## Tech Stack

| Concern | Library |
|---|---|
| GUI | PyQt6 |
| Audio transcription | basic-pitch (Spotify) — polyphonic ML |
| Audio loading / BPM detection | librosa |
| MIDI manipulation | pretty_midi |
| Waveform / spectrogram display | pyqtgraph |
| Testing | pytest |
| Config persistence | JSON (~/.midigen/config.json) |

**Python version: 3.10+**

Distribution: NOT bundled. App requires user to have Python 3.10+ installed. Provide `run.bat` (Windows) and `run.sh` (Linux) launchers, plus an optional `build.py` PyInstaller script for future standalone distribution.

---

## Module Structure

```
MidiGen/
├── main.py                    # Entry point, launches QApplication
├── requirements.txt           # Pinned major versions
├── run.bat                    # Windows launcher: python main.py
├── run.sh                     # Linux launcher: python3 main.py
├── build.py                   # Optional PyInstaller build script
├── core/
│   ├── __init__.py
│   ├── audio_loader.py        # Load audio files, trim to selection, return numpy array
│   ├── transcriber.py         # Wrap basic-pitch, return list of NoteEvent dataclasses
│   ├── midi_processor.py      # Quantize to 1/16 grid, filter ghosts, handle velocity
│   └── midi_exporter.py       # Build PrettyMIDI object, save .mid, expose drag-drop path
├── gui/
│   ├── __init__.py
│   ├── main_window.py         # QMainWindow, wires all panels, manages app state
│   ├── waveform_view.py       # Spectrogram heatmap (pyqtgraph) + draggable selection region
│   ├── controls_panel.py      # BPM, note range, confidence, toggle checkboxes
│   └── midi_preview.py        # Read-only piano roll visualization of output MIDI
├── tests/
│   ├── __init__.py
│   ├── fixtures/
│   │   └── README.md          # Instructions for adding fixture pairs
│   ├── add_fixture.py         # CLI helper to register new audio+MIDI fixture pairs
│   ├── test_audio_loader.py
│   ├── test_midi_processor.py # Most important: pure logic, no audio needed
│   ├── test_transcriber.py    # Mock basic-pitch for unit isolation
│   ├── test_midi_exporter.py
│   └── test_pipeline.py       # End-to-end tests using fixture pairs
└── CHECKLIST.md
```

---

## Data Transfer Objects

Define these dataclasses in `core/__init__.py` or a `core/models.py`:

```python
@dataclass
class AudioData:
    samples: np.ndarray       # mono float32 array
    sample_rate: int
    duration_sec: float
    file_path: Path

@dataclass
class NoteEvent:
    pitch: int                # MIDI note number (0-127)
    start_sec: float
    end_sec: float
    amplitude: float          # 0.0-1.0 from basic-pitch confidence/amplitude

@dataclass
class ProcessingConfig:
    bpm: float
    start_sec: float
    end_sec: float
    note_low: int             # MIDI note number (e.g. 36 = C2)
    note_high: int            # MIDI note number (e.g. 84 = C6)
    confidence_threshold: float
    filter_ghost_notes: bool
    dynamic_velocity: bool
```

---

## Processing Pipeline

Triggered when user clicks "Generate MIDI". Each step is a discrete function call — no step has a side effect on another module's state.

### Step 1: Load Audio (`audio_loader.py`)
- Load file with librosa at 22050 Hz, mono
- Trim to `config.start_sec` / `config.end_sec`
- Support MP3, WAV, FLAC (librosa + soundfile + audioread cover all three)
- Return `AudioData`

### Step 2: Transcribe (`transcriber.py`)
- Pass `AudioData.samples` and `AudioData.sample_rate` to basic-pitch
- Apply `config.confidence_threshold` to filter low-confidence predictions
- Return `list[NoteEvent]` — do NOT filter by note range here, do it in Step 3
- Reason: processing the full frequency range gives the ML model maximum context

### Step 3: Process (`midi_processor.py`)
All steps are discrete sub-functions, each independently testable:

1. `filter_note_range(notes, low, high) -> list[NoteEvent]` — drop notes outside MIDI range
2. `filter_ghost_notes(notes, bpm) -> list[NoteEvent]` — if enabled, drop notes shorter than one 1/16th note duration before quantization
3. `quantize_onsets(notes, bpm) -> list[NoteEvent]` — snap note start times to nearest 1/16th grid
4. `set_durations(notes, bpm) -> list[NoteEvent]` — set all durations to exactly one 1/16th note
5. `apply_velocity(notes, dynamic) -> list[NoteEvent]` — if dynamic: map amplitude (0.0-1.0) to velocity (1-127); if fixed: set all to 100

### Step 4: Export (`midi_exporter.py`)
- Build a single-track `pretty_midi.PrettyMIDI` object (instrument: Acoustic Grand Piano, program 0)
- PPQN: 960
- `save(path: Path)` — write .mid file
- `get_temp_path() -> Path` — write to `~/.midigen/tmp/output.mid` for drag-drop

### Step 5: Preview (`gui/midi_preview.py`)
- Draw a simple piano roll: time on X axis, pitch on Y axis
- Notes as filled rectangles, color-coded by velocity if dynamic mode is on
- Read-only, no editing
- Renders from `list[NoteEvent]` directly

---

## GUI Layout

### Main Window

```
┌─────────────────────────────────────────────────────────────────┐
│ Menu: File | View (theme) | Help                                │
├─────────────────────────────────────────────────────────────────┤
│ [Load Audio File]  filename.mp3        [BPM: 128] [Auto-Detect] │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   SPECTROGRAM HEATMAP (pyqtgraph ImageItem)                     │
│   [████████[████ selection ████]████████████████████████████]   │
│    0:00    ↑ start              ↑ end                   3:45   │
│                                                                 │
│   Start: [  0:32  ]     End: [  1:04  ]   (MM:SS editable)     │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  NOTE RANGE                                                     │
│  [C2 ──●────────────────────────●── C8]                        │
│         Low: [C2 ▼]   High: [C6 ▼]   (dropdown, all semitones) │
│                                                                 │
│  CONFIDENCE THRESHOLD                                           │
│  [Low ───────────●──────────── High]   0.50                    │
│                                                                 │
│  [✓] Filter ghost notes      [ ] Dynamic velocity              │
│                                                                 │
│                  [ Generate MIDI ]                              │
├─────────────────────────────────────────────────────────────────┤
│  MIDI PREVIEW (piano roll, read-only)                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  C5 ▓▓▓▓      ▓▓▓▓  ▓▓▓▓                               │  │
│  │  A4      ▓▓▓▓        ▓▓▓▓  ▓▓▓▓                        │  │
│  │  F4 ▓▓▓▓      ▓▓▓▓                                      │  │
│  └───────────────────────────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  [Save .mid File]   [Open with System Default]   [Drag ↓ here] │
└─────────────────────────────────────────────────────────────────┘
```

### Theme
- Default: dark (background ~#1e1e1e, text #f0f0f0, accent #4a9eff)
- Toggle via View menu → "Light Mode / Dark Mode"
- Apply via QApplication.setStyleSheet with a full QSS string
- Persist theme choice in config.json

---

## Controls Detail

### BPM
- `QSpinBox`, range 40–300, default 120
- "Auto-Detect" button: runs `librosa.beat_track()` on loaded audio, fills spin box
- Manual entry always overrides; auto-detect only fires on button click

### Note Range
- Dual-handle slider (`QRangeSlider` from `superqt` library) displaying semitone positions
- Labels show note names (C2, C#2, D2 ... C8) not raw MIDI numbers
- Synced bidirectionally with two `QComboBox` dropdowns listing all note names in range
- Internally stored as MIDI note numbers

### Confidence Threshold
- `QSlider`, range 0.1–1.0 (store as int 10–100 internally, divide by 100)
- Label updates live showing current float value
- Default: 0.50

### Toggles
- "Filter ghost notes" — `QCheckBox`, default: checked
- "Dynamic velocity" — `QCheckBox`, default: unchecked

### Selection Region (Waveform View)
- Click and drag on spectrogram to set region
- Region displayed as semi-transparent overlay with draggable left/right handles
- Start/End `QLineEdit` fields in MM:SS format, bidirectionally synced with visual region
- Waveform loads and renders asynchronously after file is loaded (do not block UI)

### Export Row
- "Save .mid File" — `QFileDialog.getSaveFileName`, saves to chosen path
- "Open with System Default" — saves to temp path, then:
  - Windows: `os.startfile(path)`
  - Linux: `subprocess.run(['xdg-open', str(path)])`
- Drag-drop area — a `QLabel` styled as a drop zone; implements `QDrag` with `text/uri-list` MIME pointing to temp .mid file

---

## Settings Persistence

Config file: `Path.home() / '.midigen' / 'config.json'`

Fields to persist:
```json
{
  "bpm": 120,
  "confidence_threshold": 0.5,
  "note_low": 36,
  "note_high": 84,
  "filter_ghost_notes": true,
  "dynamic_velocity": false,
  "theme": "dark",
  "last_directory": "/home/user/music"
}
```

Load on startup, save on every change or on app close.

---

## Testing

### Fixture Format

```
tests/fixtures/
  {test_name}/
    audio.mp3          (or .wav / .flac)
    expected.mid
    metadata.json      {"bpm": 128, "start_sec": 0, "end_sec": 10,
                        "confidence_threshold": 0.5, "note_low": 36,
                        "note_high": 84, "filter_ghost_notes": true,
                        "dynamic_velocity": false}
```

### `tests/add_fixture.py`
CLI script that:
1. Accepts `--audio path` and `--midi path` arguments
2. Prompts interactively for metadata values
3. Creates the fixture subdirectory and copies files
4. Writes `metadata.json`

### Test Files

**`test_audio_loader.py`**
- Test loading each supported format (generate short synthetic wav with numpy/soundfile for wav, include a tiny real mp3 for format test)
- Test that trimming to start/end produces correct duration
- Test that mono conversion works on stereo input

**`test_midi_processor.py`** (most critical — pure logic, no audio)
- Test `filter_note_range`: notes outside range dropped, boundary notes kept
- Test `filter_ghost_notes`: notes shorter than 1/16th at given BPM dropped when enabled, kept when disabled
- Test `quantize_onsets`: notes snap to nearest 1/16th grid position correctly at various BPMs
- Test `set_durations`: all output notes have exactly 1/16th duration
- Test `apply_velocity`: dynamic mode maps amplitude correctly; fixed mode sets all to 100
- Use synthetic `NoteEvent` lists — no audio needed

**`test_transcriber.py`**
- Mock `basic_pitch.inference.predict` to return a known note array
- Verify returned `list[NoteEvent]` has correct schema and confidence filtering applied

**`test_midi_exporter.py`**
- Given a synthetic `list[NoteEvent]`, verify the written .mid file contains correct pitches and timings (load back with pretty_midi and compare)

**`test_pipeline.py`**
- For each fixture in `tests/fixtures/`: load audio, run full pipeline with fixture metadata, compare output MIDI to `expected.mid`
- Comparison: match on pitch exactly; allow onset timing tolerance of ±5ms (quantization can shift slightly)
- Skip fixtures that have no `expected.mid` yet (mark as xfail)

---

## Cross-Platform Notes

- Use `pathlib.Path` everywhere — never string concatenation for paths
- Config dir: `Path.home() / '.midigen'` — create on first launch if absent
- Temp dir: `Path.home() / '.midigen' / 'tmp'`
- Open file with system: `os.startfile(path)` on Windows, `subprocess.run(['xdg-open', str(path)])` on Linux. Detect with `sys.platform`.
- Audio dependencies: librosa uses soundfile for WAV/FLAC and audioread for MP3 — both must be in requirements.txt
- On Linux, audioread may need `ffmpeg` or `gstreamer` installed system-wide; note this in README

---

## requirements.txt (provide pinned major versions)

```
PyQt6>=6.6
pyqtgraph>=0.13
superqt>=0.6          # QRangeSlider for dual-handle note range slider
basic-pitch>=0.3
librosa>=0.10
pretty_midi>=0.2
numpy>=1.24
soundfile>=0.12
audioread>=3.0
pytest>=8.0
```

---

## Code Style Requirements

- Type hints on all function signatures
- Docstrings on all public functions and classes
- No global mutable state — pass `ProcessingConfig` explicitly through the pipeline
- Each `core/` module is fully importable and runnable without any GUI import
- GUI modules may import core modules but core modules must never import GUI modules
- Use dataclasses for all data transfer objects

---

## What NOT to Build (out of scope)

- No VST/plugin output
- No real-time audio monitoring
- No multi-track MIDI output
- No MIDI editing in the preview (read-only)
- No audio playback/monitoring
- No cloud/API calls

---

## Deliverable

Generate the complete file tree with all source files filled in and ready to run. The app must launch successfully after:

```bash
pip install -r requirements.txt
python main.py
```

on both Windows (Python 3.10+) and Linux (Python 3.10+).

---

## Continuation Notes (Session 1 — 2026-03-07)

### Current State
- **73/73 tests passing (67 unit + 6 pipeline)** — ALL GREEN
- All core modules implemented and working: audio_loader, transcriber, midi_processor, midi_exporter
- GUI fully functional with dark/light themes, spectrogram, piano roll, all controls
- Three DSP enhancement techniques implemented: harmonic filtering, multi-pass ensemble, HPSS
- Deep DSP research documented in `DSP_RESEARCH.md`

### piano4_chord — SOLVED
Was 50.9% F1, now **89.5% F1** (P=89.4%, R=89.7%).

**Root cause was twofold:**
1. Harmonic filter removed real chord voicings (octaves/fifths ARE real chord notes)
2. Onset threshold too low for rapid repeated chords — caused fragmented detections

**Fix:** `filter_harmonics: false`, `onset_threshold: 0.55`, `confidence_threshold: 0.23`

**Key learning:** Different material types need radically different parameter profiles.
The harmonic filter helps melodies but hurts chords. High onset threshold helps rapid
chords but would miss soft onsets in pads. See DSP_RESEARCH.md for the full systematic
threshold search and 8 ranked improvement strategies.

### What to Work On Next

**High priority (do first):**
1. **Adaptive harmonic filter** — Instead of binary on/off, detect chord density per onset
   group and skip filtering for groups with >3 simultaneous notes. This eliminates the need
   for per-fixture `filter_harmonics` overrides. (Low-medium effort, high impact)

2. **Material type presets in GUI** — Add a dropdown: Melody, Chords, Pads, Bass, Full Mix.
   Each sets sensible defaults for all ML/DSP parameters. User can fine-tune from there.
   (Low effort, high UX impact)

3. **More test fixtures** — Target 10+ before real track testing. Need: synth lead, isolated
   bass, two-hand piano, simple EDM loop, trance supersaw chords.

**Medium priority:**
4. **Complementary multi-pass** — Instead of majority-vote ensemble, run two passes with
   complementary configs and take the union. One pass catches clean onsets, the other catches
   notes the first missed.

5. **Temporal consistency filling** — If a note appears in 14/16 repeated chord strikes but
   is missing in 2, fill it in. Very effective for EDM's repetitive patterns.

6. **Chroma cross-reference** — Use librosa chroma features to validate/reject basic-pitch
   detections.

**Lower priority (exploratory):**
7. Frequency band splitting (uncertain if basic-pitch handles isolated bands well)
8. Chord completion via music theory (risky, could introduce false positives)
9. Spectrogram-based onset anchoring (high effort, significant rework)

### Files Not Yet Created
- `tests/add_fixture.py` — CLI helper for adding fixture pairs
- `build.py` — PyInstaller build script

### Technical Debt
- basic-pitch installation requires `--no-deps` workaround for Python 3.13
- Deprecation warnings from resampy (pkg_resources) and audioread (aifc/sunau)
- GUI drag-drop to FL Studio not manually verified
- Linux not yet tested
- Ensemble mode exists but hasn't proven effective — majority voting too strict for chords,
  needs rework to union-merge mode (see DSP_RESEARCH.md #3)
