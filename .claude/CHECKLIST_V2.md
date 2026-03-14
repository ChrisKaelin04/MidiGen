# MidiGen v2 — Implementation Checklist

Branch: `feature/demucs-pipeline`
Target: 99%+ F1 on all fixtures, real-track support via Demucs

---

## Phase 1: Demucs Integration (Preprocessing)

- [ ] **Install/verify Demucs** — `pip install demucs`, check Python 3.13 compat (PyTorch dep)
- [ ] **Create `core/stem_separator.py`** — wrapper module
  - [ ] `separate_stems(audio_path, model='htdemucs') -> dict[str, AudioData]`
  - [ ] Return dict with keys: vocals, drums, bass, other (4-stem) or +guitar, +piano (6-stem)
  - [ ] Cache separated stems to `~/.midigen/tmp/stems/` to avoid re-processing
  - [ ] Progress callback for GUI status updates
- [ ] **Wire into GUI**
  - [ ] Add "Separate Stems" button (runs Demucs, shows progress)
  - [ ] Stem selector dropdown/checkboxes (which stems to transcribe)
  - [ ] Option to transcribe multiple stems and merge results
- [ ] **Wire into pipeline** — Demucs runs first, before all other preprocessing
- [ ] **Add test fixtures** — real multi-instrument tracks with expected MIDI per stem
- [ ] **Performance consideration** — 2-5 min on CPU, warn user, consider background processing

## Phase 2: Frequency Band Splitting (Spectral Validation Signal)

- [ ] **Create band splitting utility** in `core/audio_loader.py` or `core/spectral_validator.py`
  - [ ] `split_bands(audio, bands=[(0,300),(300,2000),(2000,nyquist)]) -> list[AudioData]`
  - [ ] Scipy bandpass filters, configurable boundaries
- [ ] **Use as validation signal, NOT as transcription input**
  - [ ] After basic-pitch transcribes full-spectrum audio...
  - [ ] Check: does each note have energy in the CORRECT band?
  - [ ] A bass note (MIDI 36-48) should have energy in the low band
  - [ ] A melody note (MIDI 60-84) should have energy in the mid band
  - [ ] Notes with energy ONLY in a wrong band = likely overtones/artifacts
- [ ] **Wire into spectral validator** — band energy as additional validation criterion
- [ ] **A/B test on all fixtures** — does band validation reduce false positives?

## Phase 3: Spectral Validation Fixes & Improvements

### Current Problems
- Spectral validation is built (656 lines, 56 tests) but defaults OFF
- Validation (rejecting FPs) works but is overly conservative — barely changes results
- Recovery (finding missed notes) floods with false positives because CQT can't
  distinguish real notes from harmonic overtones
- The `spectral_validate()` function signature in test_pipeline.py uses
  `recovery_min_energy_db` and `recovery_min_duration_sec` kwargs but the
  actual function uses `recovery_threshold_offset_db` and `min_recovery_duration_sec`
  — these may be mismatched and need verification

### Improvements to Implement

- [ ] **Fix kwarg mismatch** — verify test_pipeline.py calls match spectral_validate() signature
- [ ] **Spectral-aware ghost filter** (fixes piano3_arp)
  - [ ] Before killing a "too short" note, check CQT energy at that pitch
  - [ ] If energy sustains beyond detected end time -> extend the note
  - [ ] If energy drops off -> the note really is short, safe to kill
  - [ ] Should fix 8 notes killed at 116ms vs 125ms threshold (9ms gap)
- [ ] **Confidence-gated spectral validation** (biggest impact — fixes piano4/5/6)
  - [ ] Lower basic-pitch confidence to ~0.10 (pulls in 3x more candidates)
  - [ ] Use CQT energy to validate each candidate (keep if real energy, reject if noise)
  - [ ] Turns basic-pitch from "only trust high confidence" to "propose everything, CQT confirms"
  - [ ] Critical: thresholds must be RELATIVE to piece's own energy profile
- [ ] **Chunk-based adaptive thresholds** (required for confidence gating to work)
  - [ ] Divide audio into windows (1-2 bars based on BPM)
  - [ ] Compute local energy percentiles per window
  - [ ] Each note's validation threshold adapts to local dynamics
  - [ ] Fixes: quiet intro notes aren't killed by loud chorus's median
- [ ] **Harmonic fingerprinting** (distinguishes real notes from overtones)
  - [ ] A real C4 produces its OWN harmonics: C5, G5, C6...
  - [ ] An overtone of C3 at C4 does NOT produce independent harmonics above C4
  - [ ] Check if candidate note's harmonic series exists in CQT
  - [ ] If yes -> independently played. If no -> overtone of a lower note.
  - [ ] This is the key to making recovery work without flooding FPs
- [ ] **Onset-aligned chord completion** (further improves chords)
  - [ ] When multiple notes share an onset time, check what other pitches have energy
  - [ ] If 3 of 5 chord tones are detected + the other 2 have energy -> add them
  - [ ] Context-aware: guided by existing confident detections
- [ ] **Integrate band splitting as validation signal** (see Phase 2)

## Phase 4: Post-Processing Improvements

- [ ] **Spectral-aware ghost filter** (listed in Phase 3 but lives in midi_processor)
  - [ ] Pass CQT data to ghost filter decision
  - [ ] Check energy before killing short notes
- [ ] **Polyphony hint from user** (UX feature)
  - [ ] User specifies: "mono" / "low poly" / "chords" / "pads"
  - [ ] Each hint applies different parameter profiles:
    - Mono: aggressive ghost filter, strict confidence, no chord completion
    - Low poly: moderate settings
    - Chords: low confidence + spectral validation, chord completion enabled
    - Pads: very low confidence, fragment merging aggressive
  - [ ] Add as dropdown in GUI, wire into ProcessingConfig
- [ ] **Ensemble mode rework** — change from majority vote to union merge
  - [ ] Conservative pass + aggressive pass, keep union of both
  - [ ] Could recover blind-spot pitches that appear at low thresholds
- [ ] **Chroma cross-reference** — use librosa chroma as independent pitch validator
  - [ ] Compare each detected note's pitch class against chroma energy
  - [ ] Remove notes whose pitch class has no chroma support

## Phase 5: Testing & Fixtures

- [ ] **Synth test fixtures** — current fixtures are all piano
  - [ ] EDM lead synth (mono, bright harmonics)
  - [ ] Supersaw chord (high poly, dense harmonics)
  - [ ] Sub bass (low frequency, simple waveform)
  - [ ] Pad with reverb (sustained, washy)
- [ ] **Multi-instrument test fixtures** (for Demucs testing)
  - [ ] Simple EDM loop: kick + bass + lead
  - [ ] Full mix: drums + bass + chords + melody
- [ ] **Regression test harness** — track F1 over time per fixture
  - [ ] Store historical F1 scores
  - [ ] Alert if any fixture regresses by more than 1%

## Phase 6: UX Improvements

- [ ] **Wizard/baseline setter** (future — after accuracy is excellent)
  - [ ] Ask user: what type of sound? (synth lead, chord pad, bass, etc.)
  - [ ] Is it polyphonic? How many voices?
  - [ ] What's the expected tempo range?
  - [ ] Auto-configure all preprocessing + processing settings
- [ ] **Material presets dropdown** — quick preset selector
  - [ ] Melody, Chords, Pads, Bass, Full Mix
  - [ ] Each sets optimal config (confidence, onset, frame, filters, etc.)
- [ ] **Detected key display** — show estimated key in GUI
- [ ] **Per-stem transcription results** — when Demucs is used, show each stem's MIDI separately

---

## Priority Order (what to work on next)

1. **Fix spectral validator kwarg mismatch** — quick, prevents bugs
2. **Spectral-aware ghost filter** — trivial win, fixes piano3_arp to ~99%
3. **Confidence-gated spectral validation** — biggest architectural change, highest impact
4. **Chunk-based adaptive thresholds** — required for #3 to work well
5. **Harmonic fingerprinting** — required for recovery to not flood FPs
6. **Demucs integration** — enables real-track testing
7. **Band splitting as validation signal** — improves spectral validator accuracy
8. **Synth test fixtures** — validates that pipeline works beyond piano
9. **Polyphony hint + presets** — UX, makes app usable without manual tuning
10. **Wizard** — far future, after accuracy is excellent

---

## Current Scores (baseline to beat)

| Fixture | F1 | Primary Issue |
|---|---|---|
| piano1 | 100% | Perfect |
| piano2 | 97.3% | 1 FP overtone |
| piano3_arp | 96.4% | Ghost filter kills 8 real notes |
| piano4_chord | 89.7% | B4/F#4 blind spots, A3 FPs |
| piano5_lead | 85.1% | Missed D5/B4, overtone FPs |
| piano6_pads | 71.4% | B4 zero detections, many below conf |

## Test Counts: 186 passing
- 79 midi_processor
- 56 spectral_validator
- 27 audio_loader (19 new preprocessing tests)
- 9 midi_exporter
- 6 transcriber
- 6 pipeline
- 3 HPSS (in audio_loader, existing)
