#!/bin/bash

INPUT_FILE=$1
OUTPUT_FILE="synthetic/$(basename "$INPUT_FILE")"

rm -f "$OUTPUT_FILE"
mkdir -p synthetic
./preview.py "$INPUT_FILE"
metasyn create-meta "$INPUT_FILE" -o /tmp/gmf.json
metasyn synthesize /tmp/gmf.json -o "$OUTPUT_FILE"
./preview.py "$OUTPUT_FILE"

