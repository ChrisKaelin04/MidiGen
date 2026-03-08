#!/bin/bash
echo "Starting MidiGen..."
cd "$(dirname "$0")/.."
python3 -m run.main
