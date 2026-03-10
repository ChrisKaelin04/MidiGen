# MidiGen — Implementation Checklist

Track progress here as Opus generates code and you verify/test each piece.

---

## Phase 1: Project Scaffold

- [x] `main.py` — entry point, launches QApplication
- [x] `requirements.txt` — all dependencies (note: basic-pitch installed separately with --no-deps due to TF/Python 3.13 incompatibility)
- [x] `run.bat` — Windows launcher
- [x] `run.sh` — Linux launcher
- [ ] `build.py` — optional PyInstaller build script
- [x] `core/__init__.py` — dataclass definitions (AudioData, NoteEvent, ProcessingConfig)
- [x] `core/audio_loader.py` — includes HPSS support
- [x] `core/transcriber.py` — includes multi-pass ensemble mode
- [x] `core/midi_processor.py` — includes harmonic filter, snap_durations, preserve_durations
- [x] `core/midi_exporter.py`
- [x] `gui/__init__.py`
- [x] `gui/main_window.py` — dark/light themes, config persistence, pipeline worker
- [x] `gui/waveform_view.py` — spectrogram heatmap with async worker
- [x] `gui/controls_panel.py` — all controls including ML tuning and DSP sections
- [x] `gui/midi_preview.py` — piano roll with velocity coloring
- [x] `tests/__init__.py`
- [x] `tests/fixtures/README.md`
- [ ] `tests/add_fixture.py` — CLI helper (not yet created)
- [x] `tests/test_audio_loader.py`
- [x] `tests/test_midi_processor.py`
- [x] `tests/test_transcriber.py`
- [x] `tests/test_midi_exporter.py`
- [x] `tests/test_pipeline.py` — flat fixture system with precision/recall/F1 scoring

---

## Phase 2: Core Logic Verification

### audio_loader.py
- [x] Loads MP3 correctly
- [x] Loads WAV correctly
- [ ] Loads FLAC correctly (not tested with fixture, but librosa supports it)
- [x] Trimming to start/end time works
- [x] Mono conversion works on stereo files
- [x] Returns correct `AudioData` schema
- [x] HPSS (Harmonic/Percussive Source Separation) — `apply_hpss()` strips percussive content

### transcriber.py
- [x] Accepts `AudioData`, returns `list[NoteEvent]`
- [x] Confidence threshold correctly filters low-confidence notes
- [x] Does NOT filter by note range (that is midi_processor's job)
- [x] Multi-pass ensemble mode with 5 presets and majority voting
- [x] Writes temp WAV for basic-pitch 0.4.0 API compatibility

### midi_processor.py
- [x] `filter_note_range` — drops notes outside MIDI range, keeps boundary notes
- [x] `filter_ghost_notes` — drops sub-1/16th notes when enabled, keeps all when disabled
- [x] `quantize_onsets` — snaps to nearest 1/16th grid at given BPM
- [x] `set_durations` — all output notes exactly 1/16th duration
- [x] `snap_durations` — snaps end times to grid while preserving detected note lengths
- [x] `filter_harmonics` — removes likely overtone notes (1.5x amplitude ratio threshold)
- [x] `apply_velocity` — dynamic maps amplitude to velocity; fixed sets all to 100
- [x] `process()` — full pipeline with preserve_durations and filter_harmonics toggles

### midi_exporter.py
- [x] Builds valid single-track PrettyMIDI object (PPQN=960)
- [ ] Saves .mid file that opens in FL Studio (not manually verified yet)
- [x] Temp file written to `~/.midigen/tmp/output.mid`
- [ ] Drag-drop from GUI exposes correct MIME type (not verified)

---

## Phase 3: GUI Verification

### main_window.py
- [x] App launches without error
- [x] Dark theme applied by default
- [x] Light/dark toggle in View menu works and persists
- [ ] Full interactive testing not yet done

### waveform_view.py
- [x] Spectrogram heatmap renders after file load (async worker)
- [x] Click-drag creates selection region (LinearRegionItem)
- [x] Selection handles are draggable
- [x] Start/End MM:SS fields sync with visual region bidirectionally
- [x] Waveform loads asynchronously (UI does not freeze)

### controls_panel.py
- [x] BPM spin box accepts 40-300, defaults to 120
- [x] Auto-Detect BPM button fires librosa, fills spin box
- [x] Note range dual-handle slider shows note names (superqt QRangeSlider)
- [x] Note range dropdowns stay in sync with slider
- [x] Confidence slider updates label live
- [x] Ghost note filter checkbox defaults ON
- [x] Dynamic velocity checkbox defaults OFF
- [x] ML Model Tuning section (onset/frame thresholds, min note length, ensemble passes)
- [x] DSP & Filtering section (HPSS toggle, harmonic filter toggle)
- [x] Preserve durations toggle

### midi_preview.py
- [x] Piano roll renders after Generate MIDI
- [x] Notes correctly placed by pitch and time
- [x] Read-only (no interaction needed)

### Export row
- [x] "Save .mid File" opens file dialog and saves correctly
- [x] "Open with System Default" opens file in system MIDI handler
- [ ] Drag-drop area allows dragging .mid into FL Studio (not verified)

---

## Phase 4: Settings Persistence
- [x] Config file created at `~/.midigen/config.json` on first launch
- [x] All settings restored on app restart (BPM, threshold, note range, toggles, theme, last directory)

---

## Phase 5: Testing

### Unit Tests (no audio files needed) — ALL 161 PASSING
- [x] `test_midi_processor.py` — 79 tests (harmonics, fragments, patterns, timing, process)
- [x] `test_spectral_validator.py` — 56 tests (CQT, validation, recovery, overlaps, integration)
- [x] `test_midi_exporter.py` — 9 tests
- [x] `test_transcriber.py` — 6 tests (mocks basic_pitch.inference.predict)
- [x] `test_audio_loader.py` — 11 tests

### Pipeline Tests (requires fixture files) — 6/6 PASSING
- [x] piano1 — mono melody, 100% F1
- [x] piano2 — mono melody, 97.3% F1 (1 FP)
- [x] piano3_arp — mono arp, 96.4% F1 (ghost filter kills 8 borderline notes)
- [x] piano4_chord — 7th/9th chords, 89.7% F1 (blind spots B4/F#4)
- [x] piano5_leadandbass — melody+bass, 85.1% F1 (missed notes + overtone FPs)
- [x] piano6_pads — 7th/9th pads, 71.4% F1 (B4 undetectable, many below conf)

---

## Phase 6: Cross-Platform

- [x] Tested and working on Windows 11 with Python 3.13
- [ ] Tested and working on Linux with Python 3.10+
- [x] `run.bat` works correctly
- [ ] `run.sh` not yet tested on Linux
- [ ] System-open works on both platforms

---

## Known Issues / Bugs

- **piano4_chord FIXED** (was 41% recall / 50.9% F1, now 89.7% recall / 89.5% F1):
  Root cause was twofold: (1) harmonic filter removed real chord voicings (octaves/fifths
  are real notes in chords), (2) onset_threshold was too low for rapid repeated chords
  causing fragmented detections. Fix: disabled harmonic filter, raised onset to 0.55,
  lowered confidence to 0.23. See DSP_RESEARCH.md for full analysis.

- **Harmonic filter is material-dependent**: Works well for melodies, harmful for chords.
  Currently handled via per-fixture config override. Future improvement: adaptive harmonic
  filter that detects chord density and skips filtering for dense note groups.

- **basic-pitch + Python 3.13**: Must install basic-pitch with `pip install --no-deps basic-pitch` because it hard-depends on tensorflow in metadata even when using ONNX backend. Direct deps (onnxruntime, resampy, etc.) installed separately.

- **Deprecation warnings**: `pkg_resources` deprecation in resampy, `aifc`/`sunau` removed in Python 3.13 (audioread imports them). Non-blocking but may break in future Python versions.

- **Model blind spots**: basic-pitch cannot detect B4 (71) in piano4_chord at any
  threshold (amp 0.15-0.19, all below conf 0.23). F#4 (66) also very weak. In piano6_pads,
  B4 has ZERO raw detections. Only spectral recovery can fix these.

- **Ghost filter too aggressive for arps**: piano3_arp has 8 notes at 116ms (9ms below
  the 125ms 1/16th threshold at 120 BPM). Spectral-aware ghost filter needed.

- **Spectral validation research complete**: See `.claude/SPECTRAL_VALIDATION_RESEARCH.md`
  for deep audit of every note loss, what was tried, and design proposals for next session.

---

## Next Session Plan (see SPECTRAL_VALIDATION_RESEARCH.md for full design)

### Target: 99%+ F1 on ALL piano fixtures

Current: piano1=100%, piano2=97.3%, piano3=96.4%, piano4=89.7%, piano5=85.1%, piano6=71.4%

### Implementation order (on branch `feature/spectral-validation`):
1. **Spectral-aware ghost filter** — check CQT energy before killing short notes (fixes piano3 → ~99%)
2. **Confidence-gated spectral validation** — lower conf to 0.10, use CQT energy to validate/reject (fixes piano4/5/6)
3. **Chunk-based adaptive thresholds** — per-window energy percentiles (required for #2 to work across dynamic range)
4. **Onset-aligned chord completion** — use chord context to add missing notes at shared onsets
5. **Harmonic fingerprinting** — check if candidate has own harmonic series (vs just being an overtone)
6. **Polyphony hint UI** — user specifies mono/low-poly/chords/pads for parameter profiles
7. **Ensemble + spectral synergy** — ensemble consensus as second vote alongside spectral energy

---

## Future Enhancements (out of scope now)

- Multi-track MIDI output (separate tracks for detected instruments)
- Audio playback within the app
- MIDI note editing in the preview
- Standalone .exe distribution via PyInstaller
- Additional export: drag directly to Ableton Live
- More test fixtures (target ~10 before real track testing)
- Real-world track testing (EDM/trance with percussion, SFX)
