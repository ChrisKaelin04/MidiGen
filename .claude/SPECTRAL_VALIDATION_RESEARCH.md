# Spectral Validation Research — 2026-03-10

## Branch: `feature/spectral-validation`

## What Was Built

New module `core/spectral_validator.py` with functions:
- `compute_cqt()` — CQT spectrogram, 84 bins (MIDI 24-108), 1 bin/semitone
- `cqt_energy_at()` — mean energy at a pitch/time region
- `expected_overtone_energy()` — harmonic decay model (10/16/20/24/28 dB per harmonic)
- `_compute_reference_energy()` — median energy of confident notes (relative baseline)
- `validate_notes()` — energy floor check + overtone disambiguation → (kept, rejected)
- `_detect_onsets_at_bin()` — frame-level onset detection (energy rise)
- `_find_sustained_regions()` — contiguous energy above threshold (catches swells)
- `recover_notes()` — find unoccupied spectral energy, propose new notes
- `resolve_overlaps()` — fix same-pitch overlapping notes (trim or dedup)
- `spectral_validate()` — top-level orchestrator

Tests: `tests/test_spectral_validator.py` — 56 tests, all passing.

Wired into: `test_pipeline.py`, `gui/main_window.py`, `ProcessingConfig` (8 new fields).
Default: `spectral_validate=False` in config (opt-in per fixture).

## What Was Learned — Deep Audit Results

### Every note loss categorized:

| Fixture | F1 | Issue | Root Cause | Fix Needed |
|---|---|---|---|---|
| piano1 | 100% | Perfect | — | — |
| piano2 | 97.3% | 1 FP at MIDI 38 | Overtone | Spectral validation |
| piano3_arp | 96.4% | 9 missed at pitch 68 | Ghost filter kills notes 9ms too short (116ms vs 125ms threshold) | Spectral-aware ghost filter |
| piano4_chord | 89.7% | 16 missed B4, 16 missed F#4, 16 FP A3, 2 FP | B4 amp 0.15-0.19 (ALL below conf 0.23), F#4 amp 0.17-0.23 (1/32 above conf) | Spectral recovery / confidence gating |
| piano5_lead | 85.1% | 7 missed, 20 FP | 3 D5 below conf, B4 missed; extras are low-register overtones | Spectral validation for FP removal + recovery |
| piano6_pads | 71.4% | 7 missed, 1 FP | B4: 0 raw detections. F#4/E3/B3/A4/D5 below confidence. | Spectral recovery essential |

### Key Data Points

**piano3_arp pitch 68 (Ab4):**
- 31 raw detections, 32 expected. Ghost filter kills 8 at 116ms (threshold 125ms at 120 BPM).
- These are real notes 9ms too short for the grid.

**piano4_chord blind spots:**
- Pitch 71 (B4): 15 raw detections, amplitude 0.156-0.196, ALL below confidence 0.23
- Pitch 66 (F#4): 32 raw detections, amplitude 0.173-0.235, only 1 above 0.23
- These pitches are inherently weak in basic-pitch's model

**piano6_pads:**
- B4 (71): ZERO raw detections at any confidence level
- F#4 (66): 2 raw at 0.20-0.21
- Many notes detected but below confidence 0.30: E3(52)=0.27, B3(59)=0.16-0.31, D5(74)=0.17-0.29

### What Spectral Validation Tried

1. **Absolute dB thresholds** — failed. -50 dB floor too high for some notes, too low for others. Quiet pieces have all notes below -35 dB, loud pieces have noise above it.

2. **Relative thresholds (median-based)** — better. Validation became harmless (no regression on 5/6 fixtures). But doesn't help because it can only REMOVE notes.

3. **Onset-based recovery** — detected energy rises of 10+ dB. Still too noisy, hundreds of false positives even at blind-spot-only pitches, because overtone energy also has onsets.

4. **Sustained region recovery** — find contiguous energy above threshold. Better than onset detection for swelling notes, but still too many false positives.

5. **Blind-spot-only recovery** — restrict to pitches 66 and 71 only. Best result: piano6_pads 71.4%→75.0% (+3.6%), piano4_chord 89.7%→89.9% (+0.2%). But adds false B4/F#4 everywhere else.

### Why Recovery Is Hard

The CQT shows energy at EVERY pitch where there's ANY acoustic content — harmonics of played notes, resonances, sympathetic vibration, etc. A C3 note produces real energy at C4 (octave), G4 (5th), C5, E5, G5... all of these show up as "sustained energy regions." Without a way to distinguish "independently played note" from "harmonic resonance of another note," recovery floods with false positives.

## Fixture Characteristics (user-provided)

| Fixture | Polyphony | Type |
|---|---|---|
| piano1 | Mono | Simple melody |
| piano2 | Mono | Simple melody |
| piano3_arp | Mono | Arpeggiated pattern |
| piano5_lead | Low poly (≤2 simultaneous) | Melody + bass |
| piano4_chord | High poly | 7th/9th chords |
| piano6_pads | High poly | 7th/9th chord pads |

User wants ability to specify polyphony hint: "this is a chord" vs "this is mono".

## Design Proposals (Not Yet Implemented)

### 1. Confidence-Gated Spectral Validation ("Lower the Bar, Then Verify")
**The single most impactful change possible.**

Current flow: basic-pitch → confidence filter (0.23-0.50) → pipeline
Proposed flow: basic-pitch → LOW confidence (0.10) → spectral validation → pipeline

- Lower confidence to pull in ~3x more notes including the blind spot pitches
- Use CQT energy to validate each — keep notes with real spectral energy, reject noise
- This turns basic-pitch from "only trust what it's very sure about" to "propose everything, spectrogram confirms"
- Ensemble consensus becomes a SECOND vote alongside spectral energy

**Critical**: thresholds must be RELATIVE to the piece's own energy profile, not absolute dB values.

### 2. Chunk-Based Adaptive Thresholds
Energy varies across a piece (quiet intro → loud chorus → breakdown). A single median is wrong.

- Divide into windows (1-2 bars based on BPM, e.g., 2 seconds at 120 BPM)
- Compute local energy percentiles per window
- Each note's validation threshold = local_median - offset
- Recovery threshold also adapts per-window

### 3. Spectral-Aware Ghost Filter
Before killing a "too short" note, check CQT energy at that pitch:
- If energy sustains beyond the detected end time → extend the note
- If energy drops off → the note really is short, kill it
- Directly fixes piano3_arp (8 notes at 116ms vs 125ms threshold)

### 4. Harmonic Fingerprinting
To distinguish a real note from an overtone:
- A real C4 produces its OWN harmonic series: C5, G5, C6...
- An overtone of C3 at C4 does NOT produce independent energy above C4
- Check if a candidate note's harmonics are present in the CQT
- If yes → independently played. If no → overtone.

### 5. Onset-Aligned Chord Completion
When multiple notes share an onset time:
- Check what other pitches have significant energy at that moment
- If 3 of 5 chord tones are detected and the other 2 have energy → add them
- Context-aware: guided by existing confident detections, not blind scanning

### 6. Polyphony Hint from User
User tells the system: "mono", "low poly (≤2)", "chords", "pads"
- Mono: aggressive ghost filter, strict confidence, no chord completion
- Low poly: moderate settings
- Chords: low confidence + spectral validation, chord completion enabled
- Pads: very low confidence + spectral validation, fragment merging aggressive

### 7. A/B Testing Against Confident Notes
Instead of absolute energy thresholds, compare candidate notes against confident ones:
- Confident notes at this onset have energies [-5, -8, -12] dB
- Candidate at [-20] dB — is that plausible for a chord tone?
- Compare against the energy RANGE of confident simultaneous notes
- Accept if within ~15 dB of weakest confident note at same onset

## Recommended Implementation Order

1. **Spectral-aware ghost filter** — trivial win, fixes piano3_arp to ~99%
2. **Confidence-gated spectral validation** — biggest architectural change, fixes piano4/5/6
3. **Chunk-based adaptive thresholds** — makes #2 work across dynamic range
4. **Onset-aligned chord completion** — further improves chords
5. **Harmonic fingerprinting** — cleans up false positive overtones
6. **Polyphony hint** — UX feature, applies different parameter profiles

## Current Test Status (2026-03-10)
- 135 unit tests passing (79 midi_processor + 56 spectral_validator)
- 6 pipeline tests passing (no regression from spectral work)
- F1 scores unchanged from baseline (spectral validation defaults to off)

## Files Modified/Created This Session
- **Created**: `core/spectral_validator.py`, `tests/test_spectral_validator.py`
- **Modified**: `core/__init__.py` (8 new ProcessingConfig fields), `tests/test_pipeline.py` (spectral wiring), `gui/main_window.py` (spectral wiring)
- **Branch**: `feature/spectral-validation`, no commits yet (all uncommitted changes)
