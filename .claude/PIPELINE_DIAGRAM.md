# MidiGen Pipeline — Full Architecture Diagram

## Pipeline Flow (top to bottom = execution order)

```
 AUDIO FILE (MP3/WAV/FLAC)
       |
       v
 +--------------------------+
 | 1. LOAD AUDIO            |  core/audio_loader.py :: load_audio()
 |    - librosa @ 22050 Hz  |  Converts to mono float32, trims to user selection
 |    - Trim to selection   |
 +--------------------------+
       |
       v
 ==============================
  PREPROCESSING (before model)    All optional, all OFF by default
 ==============================
       |
       v
 +--------------------------+
 | 2. DEMUCS SEPARATION     |  [NOT YET BUILT]
 |    - Split into stems    |  Vocals / Drums / Bass / Other (synths)
 |    - User picks stem(s)  |  Only useful for full mixes, noop for isolated sounds
 +--------------------------+
       |
       v
 +--------------------------+
 | 3. HPSS                  |  core/audio_loader.py :: apply_hpss()
 |    - Strip percussive    |  Legacy — Demucs preferred when available
 |    - Keep harmonic       |  Helps: real tracks. Hurts: synth leads with transients
 +--------------------------+
       |
       v
 +--------------------------+
 | 4. NORMALIZE LOUDNESS    |  core/audio_loader.py :: normalize_loudness()
 |    - RMS-based LUFS      |  Makes basic-pitch see consistent input levels
 |    - Target: -14 LUFS    |  Helps: inconsistent volumes. Hurts: quiet pads
 +--------------------------+
       |
       v
 +--------------------------+
 | 5. NOISE GATE            |  core/audio_loader.py :: apply_noise_gate()
 |    - Threshold: -40 dB   |  Kills reverb tails and low-level bleed
 |    - Attack/release      |  Helps: reverby tracks. Hurts: sustained quiet notes
 +--------------------------+
       |
       v
 +--------------------------+
 | 6. PRE-EMPHASIS EQ       |  core/audio_loader.py :: apply_pre_emphasis()
 |    - Bandpass boost      |  User-settable dB (default 1.5 dB)
 |    - Default: 440-520 Hz |  Targets B4/F#4 weak zone in basic-pitch
 +--------------------------+
       |
       v
 +--------------------------+
 | 7. [BAND SPLITTING]      |  [NOT YET BUILT — experimental]
 |    - Split into low/     |  Feed bands to spectral validator as validation
 |      mid/high bands      |  signal, NOT to basic-pitch (confuses the model)
 +--------------------------+
       |
       v
 ==============================
  TRANSCRIPTION (the ML model)
 ==============================
       |
       v
 +--------------------------+
 | 8. BASIC-PITCH           |  core/transcriber.py :: transcribe()
 |    - ONNX inference      |  Confidence, onset, frame thresholds all user-settable
 |    - Confidence filter   |  Optional: ensemble multi-pass (majority vote)
 |    - Optional ensemble   |  Optional: Melodia pitch contour cleanup
 |    -> list[NoteEvent]    |  Output: pitch, start_sec, end_sec, amplitude
 +--------------------------+
       |
       v
 ==============================
  POST-PROCESSING (after model)
 ==============================
       |
       v
 +--------------------------+
 | 9. SPECTRAL VALIDATION   |  core/spectral_validator.py :: spectral_validate()
 |    - Compute CQT         |  [BUILT but defaults OFF — needs more work]
 |    - Validate: does note |  Relative thresholds (median of piece's own energy)
 |      have real energy?   |  Overtone disambiguation: is it a harmonic or real note?
 |    - Recover: find notes |  Recovery: scan for unoccupied spectral energy
 |      basic-pitch missed  |  Overlap resolution: fix same-pitch collisions
 |    - Resolve overlaps    |
 +--------------------------+
       |
       v
 +--------------------------+
 | 10. TIMING OFFSET        |  core/midi_processor.py :: process()
 |    - Shift all notes by  |  User-settable grid steps (compensate for latency)
 |      N grid units        |
 +--------------------------+
       |
       v
 +--------------------------+
 | 11. FRAGMENT MERGING     |  core/midi_processor.py :: merge_fragments()
 |    - Stitch adjacent     |  Same pitch, gap < tolerance = one note
 |      same-pitch notes    |  Re-attack detection prevents merging real repeats
 +--------------------------+
       |
       v
 +--------------------------+
 | 12. HARMONIC FILTER      |  core/midi_processor.py :: filter_harmonics()
 |    - Mode: off/on/       |  Adaptive: filter melody lines, preserve chords
 |      adaptive            |  Checks amplitude ratio between suspected overtone
 |    - Remove overtones    |  and its fundamental (1.5x threshold)
 +--------------------------+
       |
       v
 +--------------------------+
 | 13. NOTE RANGE FILTER    |  core/midi_processor.py :: filter_note_range()
 |    - Drop notes outside  |  User sets low/high via GUI sliders
 |      MIDI range          |
 +--------------------------+
       |
       v
 +--------------------------+
 | 14. GHOST NOTE FILTER    |  core/midi_processor.py :: filter_ghost_notes()
 |    - Drop notes shorter  |  Threshold = one grid unit at current BPM
 |      than 1 grid unit    |  Known issue: kills real short notes in fast arps
 +--------------------------+
       |
       v
 +--------------------------+
 | 15. QUANTIZE ONSETS      |  core/midi_processor.py :: quantize_onsets()
 |    - Snap start times    |  Grid: 1/8, 1/16, 1/32, 1/8T, 1/16T
 |      to nearest grid     |
 +--------------------------+
       |
       v
 +--------------------------+
 | 16. SNAP/SET DURATIONS   |  core/midi_processor.py :: snap_durations() or set_durations()
 |    - Preserve: snap end  |  User toggle: preserve detected lengths vs force grid
 |      to grid             |
 |    - Fixed: all = 1 grid |
 +--------------------------+
       |
       v
 +--------------------------+
 | 17. PATTERN FILLING      |  core/midi_processor.py :: fill_repeated_patterns()
 |    - Detect repeating    |  If pitch appears in 75%+ of repetitions, fill gaps
 |      chord/note patterns |  Good for EDM where 4/8/16 bar loops are the norm
 |    - Fill missing notes  |
 +--------------------------+
       |
       v
 +--------------------------+
 | 18. VELOCITY MAPPING     |  core/midi_processor.py :: apply_velocity()
 |    - Dynamic: amplitude  |  Velocity curve exponent: <1 boost quiet, >1 compress
 |      -> velocity (1-127) |
 |    - Fixed: all = 100    |
 +--------------------------+
       |
       v
 ==============================
  OUTPUT
 ==============================
       |
       v
 +--------------------------+
 | 19. MIDI EXPORT          |  core/midi_exporter.py :: build_midi() + save()
 |    - PrettyMIDI object   |  Single track, Acoustic Grand Piano (program 0)
 |    - PPQN: 960           |  Save to file or temp path for drag-drop
 +--------------------------+
       |
       v
 +--------------------------+
 | 20. PIANO ROLL PREVIEW   |  gui/midi_preview.py
 |    - Visual display      |  Read-only, color-coded by velocity
 +--------------------------+
```

## GUI Layout Order (matches pipeline)

```
+------------------------------------------+
| BPM | Confidence | Note Range            |  <- Global settings
+------------------------------------------+
| Preprocessing (before transcription)     |  <- HPSS, normalize, gate, EQ
+------------------------------------------+
| ML Model Tuning    | Post-processing     |  <- Onset/frame/ensemble | filters
+------------------------------------------+
|          [ Generate MIDI ]               |
+------------------------------------------+
| Piano Roll Preview                       |
+------------------------------------------+
| Save | Open | Drag                       |
+------------------------------------------+
```

## Key: What's Built vs What's Planned

| Step | Status | Module |
|---|---|---|
| Load audio | Done | audio_loader.py |
| Demucs | NOT BUILT | — |
| HPSS | Done | audio_loader.py |
| Normalize loudness | Done | audio_loader.py |
| Noise gate | Done | audio_loader.py |
| Pre-emphasis EQ | Done | audio_loader.py |
| Band splitting | NOT BUILT | — |
| basic-pitch | Done | transcriber.py |
| Spectral validation | Built, OFF, needs work | spectral_validator.py |
| All post-processing | Done | midi_processor.py |
| MIDI export | Done | midi_exporter.py |
| GUI | Done (all controls wired) | gui/ |
