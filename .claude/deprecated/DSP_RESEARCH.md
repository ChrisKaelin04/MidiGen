# MidiGen — DSP Research & Accuracy Improvement Strategies

Deep dive into techniques for improving audio-to-MIDI transcription accuracy,
with findings from the piano4_chord breakthrough and a roadmap to 90%+ F1 on all piano fixtures.

---

## Current Baseline (All 73 tests passing)

| Fixture | Type | Expected | Detected | TP | P | R | F1 | Gap to 90% |
|---|---|---|---|---|---|---|---|---|
| piano1 | Simple melody | 8 | 8 | 8 | 100% | 100% | **100%** | -- |
| piano2 | Simple melody | 18 | 19 | 18 | 94.7% | 100% | **97.3%** | -- |
| piano3_arp | Arpeggios | 128 | 119 | 119 | 100% | 93.0% | **96.4%** | -- |
| piano4_chord | Rapid 5-note chords | 320 | 321 | 287 | 89.4% | 89.7% | **89.5%** | 0.5% |
| piano5_leadandbass | Melody + bass | 84 | 97 | 77 | 79.4% | 91.7% | **85.1%** | 4.9% |
| piano6_pads | Sustained pad chords | 17 | 16 | 10 | 62.5% | 58.8% | **60.6%** | **29.4%** |

### Fixtures needing improvement for 90% target:
1. **piano6_pads** (60.6%) — Biggest gap. Sustained 4-second chords.
2. **piano5_leadandbass** (85.1%) — Moderate gap. Extra false positive notes.
3. **piano4_chord** (89.5%) — Nearly there. Half a percent shy.

---

## The piano4_chord Breakthrough (Session 1)

### Problem
- 64 chord strikes, 5 notes each = 320 expected notes
- 4 distinct chord voicings (F#maj, Esus2, Bsus4, Dsus2), each repeated 16x at 8th-note intervals
- Old config (0.30/0.35/0.20, harmonic filter ON): **50.9% F1** (68% P, 41% R)

### Root Cause Analysis
Two compounding issues:

1. **The harmonic filter was killing chord notes.** Chords inherently contain intervals
   that match the harmonic series (octaves +12, fifths +19). The filter was designed to
   remove overtones from single-note lines but was stripping real chord voicings.
   Disabling it alone jumped F1 from 50.9% to 68.4%.

2. **Onset threshold was too low for rapid repeated chords.** At 0.35, the model was
   generating fragmented detections — splitting single chord strikes into multiple
   partial events. Raising onset_threshold to 0.55 forced cleaner onset detection,
   producing one coherent chord group per strike instead of scattered fragments.

### Winning Config
```json
{
  "confidence_threshold": 0.23,
  "onset_threshold": 0.55,
  "frame_threshold": 0.15,
  "minimum_note_length_ms": 40,
  "ensemble_passes": 1,
  "filter_harmonics": false
}
```
**Result: P=89.4%, R=89.7%, F1=89.5%** (from 50.9%)

### Key Insight: Parameter Interactions
The three basic-pitch parameters interact in non-obvious ways:

| Parameter | What it controls | Effect when raised |
|---|---|---|
| `confidence_threshold` | Post-filter: minimum note confidence to keep | Fewer notes, higher precision |
| `onset_threshold` | Model sensitivity to note beginnings | Fewer onsets detected, cleaner separation |
| `frame_threshold` | Model sensitivity to sustained note presence | Fewer frame activations, shorter notes |

For **repeated chords**, high onset_threshold + low confidence is optimal: the high onset
threshold prevents fragmented re-triggers while the low confidence captures all chord voices.

For **sustained pads**, low onset_threshold + low frame_threshold is needed: the model
must be sensitive to the single onset and track the sustained harmonic content.

---

## Systematic Threshold Search Results

### piano4_chord (rapid 5-note chords, 8th notes at 120 BPM)

| Config (conf/onset/frame) | Harm | Det | P | R | F1 |
|---|---|---|---|---|---|
| 0.30/0.35/0.20 | ON | 191 | 68.1% | 40.6% | 50.9% |
| 0.30/0.35/0.20 | OFF | 259 | 76.4% | 61.9% | 68.4% |
| 0.25/0.40/0.20 | OFF | 334 | 76.0% | 79.4% | 77.7% |
| 0.25/0.50/0.20 | OFF | 320 | 86.9% | 86.9% | 86.9% |
| 0.25/0.55/0.15 | OFF | 314 | 89.5% | 87.8% | 88.6% |
| **0.23/0.55/0.15** | **OFF** | **321** | **89.4%** | **89.7%** | **89.5%** |
| 0.30/0.55/0.20 | OFF | 257 | 93.4% | 75.0% | 83.2% |
| 0.20/0.55/0.20 | OFF | 343 | 84.0% | 90.0% | 86.9% |

### piano6_pads (sustained 4-second chords)

| Config (conf/onset/frame) | Det | P | R | F1 |
|---|---|---|---|---|
| 0.30/0.40/0.15 (current) | 16 | 62.5% | 58.8% | 60.6% |
| 0.20/0.30/0.10 | 24 | 54.2% | 76.5% | 63.4% |
| 0.15/0.50/0.10 | 23 | 60.9% | 82.4% | **70.0%** |
| 0.20/0.50/0.15 | 28 | 53.6% | 88.2% | 66.7% |
| 0.20/0.40/0.15 | 32 | 43.8% | 82.4% | 57.1% |

Best achievable with threshold tuning alone: **~70% F1**. Threshold tuning alone
cannot reach 90% on pads. We need additional techniques.

### Model Blind Spots (persistent across ALL threshold settings)
- **B4 (71)**: Never detected at any threshold above 0.10. Appears in piano4_chord AND piano6_pads.
- **E5 (76)**: In pads, expected to sustain 16 seconds — never detected as a sustained note.
- **F#4 (66)**: Very weakly detected, requires conf < 0.20.

---

## Diagnosed Problems Per Fixture

### piano6_pads — Why it's hard (60.6% F1)
The expected MIDI has 17 notes across 4 chords, each sustained for 4 seconds. The problems:

1. **E5 (76) sustains for 16 SECONDS** and is never detected. The model can't track
   a single note across that duration.
2. **B4 (71)** is a persistent model blind spot — never detected.
3. **F#4 (66)** detected at wrong onset time (0.0s instead of 8.0s).
4. **Fragmented detections** — model splits sustained notes into multiple short events.
   D4 appears as 3 fragments (12.0-13.1s, 13.1-14.7s, 14.7-15.9s) instead of one 4s note.
5. **False positives** — A5 (81), extra D3, extra C#5 detected as overtones/ghosts.

**Key insight**: The onset-matching algorithm penalizes fragmented notes. If D4 is split
into 3 pieces, only the first matches. The other 2 are false positives AND they prevent
matching subsequent expected notes.

### piano5_leadandbass — Why it's at 85.1%
- 97 detected vs 84 expected = 13 extra notes (79.4% precision)
- 91.7% recall means only 7 notes missed
- Primary issue is **false positives**, likely overtones from the bass register
- The harmonic filter should help HERE but it's currently ON (defaults.json)
- May benefit from stricter confidence threshold for the bass register

---

## Techniques to Push All Fixtures to 90%+

### Tier 1: Highest Impact, Implement First

#### 1. Adaptive Harmonic Filter (addresses piano5_leadandbass precision)
Instead of a binary on/off, make the harmonic filter context-aware:

- **Detect chord density**: Count simultaneous notes per onset group
- If > 3 notes detected simultaneously, reduce or disable harmonic filtering for that group
- If 1-2 notes, apply full harmonic filtering

This lets the filter work on melodies/bass (where overtones are the main false-positive
source) while leaving dense chords intact. Eliminates the need for per-fixture config overrides.

```python
def filter_harmonics_adaptive(notes: list[NoteEvent], onset_tol: float = 0.05,
                               chord_threshold: int = 3) -> list[NoteEvent]:
    groups = _group_by_onset(notes, onset_tol)
    result = []
    for group in groups:
        if len(group) > chord_threshold:
            result.extend(group)  # chord — keep all
        else:
            result.extend(filter_harmonics(group, onset_tol))  # melody — filter
    return result
```

**Expected impact**: Improve piano5_leadandbass precision (remove bass overtones),
maintain piano4_chord recall (don't filter chord notes). Could push piano5 from 85% to 90%+.

**Effort**: Low. 20-30 lines of code.

#### 2. Note Merging / Fragment Stitching (addresses piano6_pads fragmentation)
When the model splits a sustained note into multiple adjacent fragments, stitch them back
together. This directly fixes the piano6_pads problem where D4 appears as 3 fragments.

```python
def merge_fragments(notes: list[NoteEvent], gap_tol: float = 0.05) -> list[NoteEvent]:
    """Merge notes with same pitch where end of one ≈ start of next."""
    sorted_by_pitch_start = sorted(notes, key=lambda n: (n.pitch, n.start_sec))
    result = []
    i = 0
    while i < len(sorted_by_pitch_start):
        current = sorted_by_pitch_start[i]
        # Absorb subsequent fragments of same pitch
        while (i + 1 < len(sorted_by_pitch_start) and
               sorted_by_pitch_start[i + 1].pitch == current.pitch and
               sorted_by_pitch_start[i + 1].start_sec - current.end_sec <= gap_tol):
            i += 1
            next_note = sorted_by_pitch_start[i]
            current = NoteEvent(
                pitch=current.pitch,
                start_sec=current.start_sec,
                end_sec=next_note.end_sec,
                amplitude=max(current.amplitude, next_note.amplitude),
            )
        result.append(current)
        i += 1
    return result
```

**Expected impact**: Critical for piano6_pads. Reduces false positives (fragments no longer
count as separate incorrect notes) and could fix 3-5 notes per chord. Could push from 60% to 75%+.

**Effort**: Low. ~25 lines, pure function, easy to test.

#### 3. Temporal Pattern Repetition (addresses piano4_chord last 0.5% and general EDM)
If a chord pattern repeats consistently (same pitches at regular intervals) but a note
is missing from occasional instances, fill it in.

```python
def fill_repeated_patterns(notes: list[NoteEvent], bpm: float,
                            onset_tol: float = 0.05,
                            min_pattern_occurrences: int = 4,
                            fill_threshold: float = 0.75) -> list[NoteEvent]:
    """
    1. Group notes by onset time
    2. Identify repeating chord patterns at regular beat intervals
    3. For chords that repeat N times, if a pitch appears in >= fill_threshold
       of instances but is missing in others, add it to the missing positions
    """
```

**Expected impact**: Could push piano4_chord from 89.5% to 93%+. Very powerful for
EDM where 4/8/16 bar patterns are the norm.

**Effort**: Medium. ~60-80 lines. Needs careful logic to avoid hallucinating notes.

### Tier 2: Medium Impact, Implement After Tier 1

#### 4. Multi-Pass Union Merge (complement, don't vote)
The current ensemble uses majority voting which is too strict for chords. Instead:

- **Pass A**: Conservative config (high onset, catches clean boundaries)
- **Pass B**: Aggressive config (low onset, catches notes the first pass missed)
- **Merge**: Union of both, deduplicate by onset+pitch proximity

Any note found by EITHER pass is kept. This is the opposite of the current approach.

```python
def union_merge(pass_a: list[NoteEvent], pass_b: list[NoteEvent],
                onset_tol: float = 0.05) -> list[NoteEvent]:
    """Keep all notes from both passes, deduplicating matches."""
    result = list(pass_a)
    for note in pass_b:
        if not any(_notes_match(note, existing, onset_tol) for existing in result):
            result.append(note)
    return result
```

**Expected impact**: Could recover blind-spot pitches (B4, F#4) that appear at aggressive
thresholds but not conservative ones, without flooding with noise.

**Effort**: Low-medium. Modify `transcribe()` to support union mode alongside majority mode.

#### 5. Chroma Cross-Reference Validation
Use librosa's chroma features as an independent pitch detector to validate basic-pitch:

```python
import librosa
import numpy as np

def chroma_validate(notes: list[NoteEvent], audio_samples: np.ndarray,
                     sr: int, threshold: float = 0.3) -> list[NoteEvent]:
    """Remove notes whose pitch class is not confirmed by chroma analysis."""
    chroma = librosa.feature.chroma_cqt(y=audio_samples, sr=sr, hop_length=512)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=512)

    validated = []
    for note in notes:
        # Find chroma frames during note's onset
        mask = (times >= note.start_sec - 0.025) & (times <= note.start_sec + 0.025)
        if mask.any():
            pitch_class = note.pitch % 12
            chroma_energy = chroma[pitch_class, mask].mean()
            if chroma_energy >= threshold:
                validated.append(note)
            # else: chroma doesn't confirm this pitch class — likely false positive
        else:
            validated.append(note)  # edge case: keep if no chroma data
    return validated
```

**Expected impact**: Could remove false positives (A3/57 overtone in piano4_chord,
A5/81 in piano6_pads) without affecting real notes. Improves precision.

**Effort**: Medium. ~40 lines + needs tuning of chroma threshold.

#### 6. Pitch-Aware Onset Grouping
Currently we match detected vs expected notes individually. A smarter approach:
group all notes within a small onset window, compare the detected pitch SET against
the expected pitch SET, and score at the chord level.

This wouldn't change detection, but would give a fairer accuracy metric for chords
where the model gets 4/5 notes correct but the individual note-level matching is harsh.

**Expected impact**: More accurate scoring (might reveal we're better than we think).

**Effort**: Medium. Rewrite `_match_notes` in test_pipeline.py.

### Tier 3: Advanced / Music Theory Approaches

#### 7. Chord Recognition and Completion
When the model detects 3-4 notes of a chord, use music theory to identify the chord
quality and fill in missing voices.

**Approach:**
```
Detected: [F#2, C#4, A4]  →  Recognize as F# major (F#-A#-C#)
                              But A4 is not in F# major... it's F#m7? Or A/F#?

Detected: [E2, E3, B3]     →  E5 (power chord? sus2? context needed)
```

**The hard part**: Chord identification is ambiguous. [C, E, G] could be C major, but
it could also be Am7 without the A, or Em with a passing C. Context matters enormously.

**Implementation strategy:**
1. Build a chord template library (all common triads, 7ths, 9ths, sus chords)
2. For each onset group, find the best-matching template
3. If a template matches with 60%+ of its notes present, add the missing ones
4. Weight by: how common the chord is, whether it fits the detected key, voice leading

**Risk**: High false-positive potential. A misidentified chord adds completely wrong notes.
Must be optional ("Chord Assist" toggle) and conservative.

**Effort**: High. 200+ lines. Needs a chord theory module.

#### 8. Key Detection and Scale Filtering
Determine the key of the piece and use it to reject notes outside the scale.

```python
import librosa

def detect_key(samples: np.ndarray, sr: int) -> tuple[str, str]:
    """Detect musical key using chroma analysis."""
    chroma = librosa.feature.chroma_cqt(y=samples, sr=sr)
    chroma_mean = chroma.mean(axis=1)

    # Krumhansl-Schmuckler key profiles
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

    # Correlate with each possible key
    best_key = None
    best_corr = -1
    for shift in range(12):
        rotated = np.roll(chroma_mean, -shift)
        for mode, profile in [('major', major_profile), ('minor', minor_profile)]:
            corr = np.corrcoef(rotated, profile)[0, 1]
            if corr > best_corr:
                best_corr = corr
                note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                best_key = (note_names[shift], mode)
    return best_key
```

Once we know the key, we can:
- Flag detected notes outside the scale (likely false positives)
- Weight missing-note candidates by scale membership
- Provide the user with a "Detected Key" display in the GUI

**Expected impact**: Moderate. Most false positives in our fixtures happen to be
scale-adjacent or overtone-related, not out-of-key. But useful for real tracks where
random noise can produce out-of-key detections.

**Effort**: Medium. Key detection is well-studied and libraries exist.

#### 9. Voice Leading / Melodic Continuity
Track individual voices across time and use melodic motion constraints to:
- Reject notes that create implausible melodic leaps (>octave jump in a melody line)
- Fill in notes that would create smooth voice leading where gaps exist

**Example**: If we detect [C4, E4, G4] then [C4, E4, ?] then [C4, E4, G4], the missing
note in the middle is almost certainly G4 (voice continuity).

This is essentially a Hidden Markov Model or dynamic programming problem:
- States: possible pitches per time step
- Transitions: penalize large intervals, reward stepwise motion
- Emissions: basic-pitch detection confidence

**Expected impact**: High for real music. Voice leading is a strong musical prior.

**Effort**: Very high. 300+ lines, needs careful modeling.

#### 10. Spectral Reassignment / CQT Analysis
Use librosa's Constant-Q Transform for better frequency resolution in low octaves,
then cross-reference with basic-pitch output.

```python
cqt = librosa.cqt(y=samples, sr=sr, hop_length=512,
                   fmin=librosa.note_to_hz('C2'), n_bins=60)
```

CQT has logarithmic frequency resolution (unlike FFT's linear resolution), making it
much better at resolving bass notes. basic-pitch uses a similar representation internally
but we can use an independent CQT as cross-reference.

**Expected impact**: Could help with bass note accuracy in piano5_leadandbass.

**Effort**: Medium. Need to convert CQT bins back to MIDI note numbers and compare.

#### 11. Frequency Band Splitting
Split audio into bands (e.g., <200Hz, 200-2000Hz, >2000Hz), transcribe each
independently, merge results.

**Rationale:** In dense chords, lower-frequency notes mask higher ones due to auditory
masking effects. Isolating frequency bands could help the model detect notes that get
buried in the full spectrum.

```python
from scipy.signal import butter, filtfilt

def bandpass(samples, sr, low_hz, high_hz, order=4):
    nyq = sr / 2
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype='band')
    return filtfilt(b, a, samples).astype(np.float32)
```

**Risk**: basic-pitch was trained on full-spectrum audio. Isolated bands may confuse
the model — a bass note without its harmonics sounds different than what the model learned.

**Effort**: Medium. Need proper bandpass filtering + merge logic.

#### 12. Spectrogram-Based Independent Onset Detection
Use librosa's onset detection independently of basic-pitch to establish a timing grid,
then use basic-pitch only for pitch identification.

**Approach:**
1. `librosa.onset.onset_detect()` to find all onset times
2. For each onset, extract a short window of audio (~200ms)
3. Run basic-pitch on each window (or use CQT/chroma to identify pitches)
4. This separates "when" from "what pitch"

**Expected impact**: Could dramatically improve onset accuracy for staccato material.
librosa's onset detection is simpler but more robust than basic-pitch's neural onset detection.

**Effort**: High. Significant pipeline rework.

---

## Proposed Git Branches for Tomorrow

### Branch 1: `feature/fragment-merging` (Tier 1, START HERE)
**Goal**: Merge fragmented note detections for sustained notes.

- Add `merge_fragments()` to `midi_processor.py`
- Add it to the `process()` pipeline (after quantize_onsets, before snap_durations)
- Add unit tests for fragment merging
- Re-tune piano6_pads config with fragment merging enabled
- **Target**: Push piano6_pads from 60% to 75%+

### Branch 2: `feature/adaptive-harmonic-filter` (Tier 1)
**Goal**: Auto-detect chords vs melody and apply harmonic filter selectively.

- Add `filter_harmonics_adaptive()` to `midi_processor.py`
- Replace binary `do_filter_harmonics` with `harmonic_filter_mode: 'off' | 'on' | 'adaptive'`
- Update ProcessingConfig, GUI controls, test fixtures
- Remove `filter_harmonics: false` overrides from piano4_chord.json and piano6_pads.json
- **Target**: Single config works well for BOTH melodies and chords

### Branch 3: `feature/temporal-filling` (Tier 1)
**Goal**: Fill missing notes in repeating patterns.

- Add `fill_repeated_patterns()` to `midi_processor.py`
- Detect regular beat intervals using BPM
- If a pitch appears in 75%+ of chord instances at that interval, fill gaps
- Add GUI toggle: "Pattern Fill" checkbox
- **Target**: Push piano4_chord from 89.5% to 93%+

### Branch 4: `feature/union-merge-ensemble` (Tier 2)
**Goal**: Rework ensemble mode from majority-vote to union-merge.

- Add `union_merge()` alongside existing `_ensemble_merge()` in `transcriber.py`
- Add ensemble_mode parameter: 'majority' | 'union'
- Run conservative + aggressive passes, keep union of results
- **Target**: Recover model blind spots (B4, F#4) while maintaining precision

### Branch 5: `feature/chroma-validation` (Tier 2)
**Goal**: Use chroma features to validate/reject detected notes.

- Add `chroma_validate()` to a new `core/dsp_helpers.py` module
- Cross-reference basic-pitch output with librosa chroma
- Remove notes whose pitch class has no chroma support
- **Target**: Remove overtone false positives, improve precision across all fixtures

### Branch 6: `feature/music-theory` (Tier 3, ADVANCED)
**Goal**: Key detection, chord recognition, scale-aware filtering.

- New module: `core/music_theory.py`
- Implement key detection using Krumhansl-Schmuckler algorithm
- Implement chord template matching
- Add optional chord completion with conservative fill
- Add "Detected Key" display in GUI
- **Target**: Long-term accuracy improvement, especially for real tracks

### Branch 7: `feature/material-presets` (UX)
**Goal**: Add material type presets to the GUI.

- Add preset dropdown to controls_panel.py: Melody, Chords, Pads, Bass, Full Mix
- Each preset sets confidence, onset, frame, min_note_length, harmonic_filter, HPSS
- User can still fine-tune after selecting preset
- **Target**: Make the app usable without manual parameter tuning

---

## Recommended Session 2 Execution Order

```
1. feature/fragment-merging     ← biggest bang for buck (piano6_pads 60%→75%+)
2. feature/adaptive-harmonic    ← eliminates per-fixture config hacks
3. feature/temporal-filling     ← pushes repeating patterns to 93%+
4. Re-run all fixtures, tune configs
5. feature/material-presets     ← UX win, makes app usable
6. If time: feature/union-merge or feature/chroma-validation
```

After these, we should be at 85%+ on ALL fixtures. The remaining gap to 90% on
piano6_pads will likely require either music-theory approaches or accepting that
basic-pitch has fundamental blind spots for certain pitches (B4) in certain contexts.

---

## HPSS Findings
- HPSS **hurts** chord detection (removes harmonic energy the model needs)
- HPSS helps for pads and single-note lines by cleaning transients
- HPSS will likely be essential for real tracks with percussion
- For piano-only fixtures, HPSS provides no benefit

---

## Appendix: basic-pitch Parameter Reference

### `onset_threshold` (basic-pitch internal)
Controls the onset detection sensitivity in the neural network. Higher values
require stronger evidence of a new note beginning.
- Range: 0.0-1.0, Default: 0.5
- For repeated chords: 0.5-0.6 (prevents re-triggers)
- For sustained notes: 0.3-0.4 (catches soft onsets)

### `frame_threshold` (basic-pitch internal)
Controls frame-level activation threshold. Each frame (~11.6ms) is classified
as note-active or not. Lower values detect more sustained content.
- Range: 0.0-1.0, Default: 0.3
- For short staccato: 0.2-0.3
- For sustained: 0.1-0.2

### `minimum_note_length_ms` (basic-pitch internal)
Minimum note duration in milliseconds. Notes shorter than this are discarded.
- Default: 58ms
- For rapid passages: 30-50ms
- For pads: 80-127ms

### `confidence_threshold` (our post-filter)
Applied after basic-pitch returns results. Removes notes below this amplitude.
- Range: 0.0-1.0, Default: 0.5
- For clean piano: 0.3-0.5
- For dense chords: 0.2-0.3
- For noisy mixes: 0.4-0.6

### Optimal per-material configs (discovered)

| Material | Conf | Onset | Frame | MinMs | Harm | Notes |
|---|---|---|---|---|---|---|
| Simple melody | 0.50 | 0.50 | 0.30 | 58 | ON | Default works great |
| Arpeggios | 0.50 | 0.50 | 0.30 | 58 | ON | Default works great |
| Rapid chords | 0.23 | 0.55 | 0.15 | 40 | OFF | High onset prevents re-triggers |
| Sustained pads | 0.30 | 0.40 | 0.15 | 80 | OFF | Low frame tracks sustain |
| Lead + bass | 0.50 | 0.50 | 0.30 | 58 | ON | Harmonic filter helps with bass overtones |
