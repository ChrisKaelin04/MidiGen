"""CLI helper to add audio + MIDI fixture pairs for testing.

Copies files into tests/fixtures/ with matching names.
Optionally creates a per-file config override JSON.

Usage:
    python tests/add_fixture.py --audio clip.wav --midi clip.mid
    python tests/add_fixture.py --audio clip.wav --midi clip.mid --config
    python tests/add_fixture.py --dir path/to/folder   (bulk add all pairs)
"""

import argparse
import json
import shutil
from pathlib import Path


FIXTURES_DIR = Path(__file__).parent / 'fixtures'
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac'}


def add_single(audio: Path, midi: Path, prompt_config: bool) -> None:
    """Add a single audio + MIDI pair."""
    if not audio.exists():
        print(f"Error: audio file not found: {audio}")
        return
    if not midi.exists():
        print(f"Error: MIDI file not found: {midi}")
        return

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Use the audio file's stem as the canonical name
    stem = audio.stem
    audio_dest = FIXTURES_DIR / f"{stem}{audio.suffix.lower()}"
    midi_dest = FIXTURES_DIR / f"{stem}.mid"

    if audio_dest.exists():
        print(f"Warning: overwriting existing {audio_dest.name}")
    if midi_dest.exists():
        print(f"Warning: overwriting existing {midi_dest.name}")

    shutil.copy2(audio, audio_dest)
    shutil.copy2(midi, midi_dest)
    print(f"  Added: {audio_dest.name} + {midi_dest.name}")

    if prompt_config:
        _create_override(stem)


def add_from_dir(source_dir: Path, prompt_config: bool) -> None:
    """Bulk add all matched audio+MIDI pairs from a directory."""
    if not source_dir.is_dir():
        print(f"Error: not a directory: {source_dir}")
        return

    # Index files by lowercase stem
    audio_files: dict[str, Path] = {}
    midi_files: dict[str, Path] = {}

    for f in source_dir.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() in AUDIO_EXTENSIONS:
            audio_files[f.stem.lower()] = f
        elif f.suffix.lower() == '.mid':
            midi_files[f.stem.lower()] = f

    matched = sorted(set(audio_files.keys()) & set(midi_files.keys()))
    if not matched:
        print(f"No matching audio+MIDI pairs found in {source_dir}")
        return

    print(f"Found {len(matched)} pairs:")
    for stem in matched:
        add_single(audio_files[stem], midi_files[stem], prompt_config=False)

    if prompt_config:
        resp = input("\nCreate a shared config override for all? [y/N]: ").strip().lower()
        if resp in ('y', 'yes'):
            for stem in matched:
                _create_override(stem)

    unmatched_audio = set(audio_files.keys()) - set(midi_files.keys())
    unmatched_midi = set(midi_files.keys()) - set(audio_files.keys())
    if unmatched_audio:
        print(f"\n  Audio without MIDI (skipped): {', '.join(sorted(unmatched_audio))}")
    if unmatched_midi:
        print(f"\n  MIDI without audio (skipped): {', '.join(sorted(unmatched_midi))}")


def _create_override(stem: str) -> None:
    """Interactively create a per-file config override."""
    print(f"\n  Config override for '{stem}' (Enter to skip each field):")
    overrides: dict = {}

    bpm = input("    BPM [skip]: ").strip()
    if bpm:
        overrides["bpm"] = float(bpm)

    conf = input("    Confidence threshold 0.0-1.0 [skip]: ").strip()
    if conf:
        overrides["confidence_threshold"] = float(conf)

    low = input("    Note low MIDI number [skip]: ").strip()
    if low:
        overrides["note_low"] = int(low)

    high = input("    Note high MIDI number [skip]: ").strip()
    if high:
        overrides["note_high"] = int(high)

    if overrides:
        override_path = FIXTURES_DIR / f"{stem}.json"
        with open(override_path, 'w') as f:
            json.dump(overrides, f, indent=2)
        print(f"    Saved: {override_path.name}")
    else:
        print("    No overrides — will use defaults.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add test fixtures to MidiGen")
    parser.add_argument('--audio', type=Path, help="Path to a single audio file")
    parser.add_argument('--midi', type=Path, help="Path to matching MIDI file")
    parser.add_argument('--dir', type=Path, help="Directory to bulk-import matched pairs from")
    parser.add_argument('--config', action='store_true', help="Prompt for per-file config overrides")
    args = parser.parse_args()

    if args.dir:
        add_from_dir(args.dir, args.config)
    elif args.audio and args.midi:
        add_single(args.audio, args.midi, args.config)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python tests/add_fixture.py --audio clip.wav --midi clip.mid")
        print("  python tests/add_fixture.py --dir path/to/renders/")
        print("  python tests/add_fixture.py --dir path/to/renders/ --config")


if __name__ == "__main__":
    main()
