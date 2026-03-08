# Test Fixtures

Drop audio and MIDI files into this folder. They are matched by filename stem
(case-insensitive):

    piano_simple.wav   <-->   Piano_Simple.mid
    bass_line.mp3      <-->   BASS_LINE.mid
    synth_chords.flac  <-->   synth_chords.mid

Supported audio formats: `.mp3`, `.wav`, `.flac`

## Configuration

`defaults.json` contains the shared processing config used for all fixtures.
To override settings for a specific file, create `{stem}.json` (case-insensitive
match), e.g. `bass_line.json`:

```json
{
  "bpm": 140,
  "note_low": 24
}
```

Only the fields you include will override the defaults.

## What to render

- Keep clips short (5-15 seconds)
- Simple is better: single instrument, clear notes
- Vary the instrument type: piano, synth lead, bass, pluck, etc.
- Export both the audio render AND the source MIDI from your DAW
- Name them identically (minus extension)
