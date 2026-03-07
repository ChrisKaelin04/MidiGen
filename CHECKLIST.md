# MidiGen — Implementation Checklist

Track progress here as Opus generates code and you verify/test each piece.

---

## Phase 1: Project Scaffold

- [ ] `main.py` — entry point, launches QApplication
- [ ] `requirements.txt` — all dependencies with pinned major versions
- [ ] `run.bat` — Windows launcher
- [ ] `run.sh` — Linux launcher
- [ ] `build.py` — optional PyInstaller build script
- [ ] `core/__init__.py` — dataclass definitions (AudioData, NoteEvent, ProcessingConfig)
- [ ] `core/audio_loader.py`
- [ ] `core/transcriber.py`
- [ ] `core/midi_processor.py`
- [ ] `core/midi_exporter.py`
- [ ] `gui/__init__.py`
- [ ] `gui/main_window.py`
- [ ] `gui/waveform_view.py`
- [ ] `gui/controls_panel.py`
- [ ] `gui/midi_preview.py`
- [ ] `tests/__init__.py`
- [ ] `tests/fixtures/README.md`
- [ ] `tests/add_fixture.py`
- [ ] `tests/test_audio_loader.py`
- [ ] `tests/test_midi_processor.py`
- [ ] `tests/test_transcriber.py`
- [ ] `tests/test_midi_exporter.py`
- [ ] `tests/test_pipeline.py`

---

## Phase 2: Core Logic Verification

### audio_loader.py
- [ ] Loads MP3 correctly
- [ ] Loads WAV correctly
- [ ] Loads FLAC correctly
- [ ] Trimming to start/end time works
- [ ] Mono conversion works on stereo files
- [ ] Returns correct `AudioData` schema

### transcriber.py
- [ ] Accepts `AudioData`, returns `list[NoteEvent]`
- [ ] Confidence threshold correctly filters low-confidence notes
- [ ] Does NOT filter by note range (that is midi_processor's job)

### midi_processor.py
- [ ] `filter_note_range` — drops notes outside MIDI range, keeps boundary notes
- [ ] `filter_ghost_notes` — drops sub-1/16th notes when enabled, keeps all when disabled
- [ ] `quantize_onsets` — snaps to nearest 1/16th grid at given BPM
- [ ] `set_durations` — all output notes exactly 1/16th duration
- [ ] `apply_velocity` — dynamic maps amplitude→velocity; fixed sets all to 100

### midi_exporter.py
- [ ] Builds valid single-track PrettyMIDI object
- [ ] Saves .mid file that opens in FL Studio
- [ ] Temp file written to `~/.midigen/tmp/output.mid`
- [ ] Drag-drop from GUI exposes correct MIME type

---

## Phase 3: GUI Verification

### main_window.py
- [ ] App launches without error
- [ ] Dark theme applied by default
- [ ] Light/dark toggle in View menu works and persists

### waveform_view.py
- [ ] Spectrogram heatmap renders after file load
- [ ] Click-drag creates selection region
- [ ] Selection handles are draggable
- [ ] Start/End MM:SS fields sync with visual region bidirectionally
- [ ] Waveform loads asynchronously (UI does not freeze)

### controls_panel.py
- [ ] BPM spin box accepts 40–300, defaults to 120
- [ ] Auto-Detect BPM button fires librosa, fills spin box
- [ ] Note range dual-handle slider shows note names (not raw MIDI numbers)
- [ ] Note range dropdowns stay in sync with slider
- [ ] Confidence slider updates label live
- [ ] Ghost note filter checkbox defaults ON
- [ ] Dynamic velocity checkbox defaults OFF

### midi_preview.py
- [ ] Piano roll renders after Generate MIDI
- [ ] Notes correctly placed by pitch and time
- [ ] Read-only (no interaction needed)

### Export row
- [ ] "Save .mid File" opens file dialog and saves correctly
- [ ] "Open with System Default" opens file in system MIDI handler
- [ ] Drag-drop area allows dragging .mid into FL Studio

---

## Phase 4: Settings Persistence
- [ ] Config file created at `~/.midigen/config.json` on first launch
- [ ] All settings restored on app restart (BPM, threshold, note range, toggles, theme, last directory)

---

## Phase 5: Testing

### Unit Tests (no audio files needed)
- [ ] `test_midi_processor.py` — all sub-functions pass with synthetic NoteEvent input
- [ ] `test_midi_exporter.py` — output .mid matches expected pitches/timings
- [ ] `test_transcriber.py` — mocked basic-pitch returns correct schema

### Pipeline Tests (requires fixture files)
- [ ] Add at least one simple fixture (single instrument, short clip)
- [ ] `test_pipeline.py` passes for that fixture
- [ ] Add more fixtures progressively (chords, bassline, full EDM clip)

---

## Phase 6: Cross-Platform

- [ ] Tested and working on Windows 10/11 with Python 3.10+
- [ ] Tested and working on Linux with Python 3.10+
- [ ] `run.bat` and `run.sh` both work correctly
- [ ] System-open works on both platforms

---

## Known Issues / To Investigate

_Use this section to track problems found during testing_

-

---

## Future Enhancements (out of scope now)

- Multi-track MIDI output (separate tracks for detected instruments)
- Audio playback within the app
- MIDI note editing in the preview
- Standalone .exe distribution via PyInstaller
- Additional export: drag directly to Ableton Live
