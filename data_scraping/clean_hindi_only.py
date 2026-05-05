"""
Create a strict Hindi-only version of a JSONL dataset.

Input JSONL format:
  {"text": "...", "source": "...", "lang": "hi"}

Output:
    - Keeps only Devanagari script text and basic Hindi punctuation.
    - Removes all Latin/English and other foreign characters.
    - Removes empty parenthesis artifacts like () and ( - ).
    - Writes only the text field in final JSONL records.
    - Drops records that become too short after cleaning.

Run:
  python clean_hindi_only.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

INPUT_PATH = Path("data/processed/nonconv_hindi_medical_40M_clean.jsonl")
OUTPUT_PATH = Path("data/processed/nonconv_hindi_medical_40M_hindi_only.jsonl")

# Allowed:
# - Devanagari block: U+0900-U+097F
# - Space/newline/tab
# - Hindi/neutral punctuation that is useful for text continuity
ALLOWED_CHARS_RE = re.compile(r"[^\u0900-\u097F \t\n।॥,;:!?()\-\"'“”‘’0-9०-९]")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
MULTI_NL_RE = re.compile(r"\n{3,}")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,;:!?।॥)])")
ENGLISH_RE = re.compile(r"[A-Za-z]")
EMPTY_PARENS_RE = re.compile(r"\(\s*[-,:;]*\s*\)")

MIN_WORDS_AFTER_CLEAN = 12


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")

    # Remove URLs first.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Keep only allowed Hindi characters/punctuation.
    text = ALLOWED_CHARS_RE.sub(" ", text)

    # Remove empty parenthesis fragments created by stripping.
    text = EMPTY_PARENS_RE.sub(" ", text)

    # Normalize spacing.
    text = MULTI_NL_RE.sub("\n\n", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)

    return text.strip()


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    dropped_short = 0
    dropped_empty = 0
    dropped_non_hindi = 0

    with INPUT_PATH.open("r", encoding="utf-8") as fin, OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = obj.get("text", "")
            if not text:
                dropped_empty += 1
                continue

            cleaned = clean_text(text)
            if not cleaned:
                dropped_empty += 1
                continue

            # Hard guard: no English letters at all.
            if ENGLISH_RE.search(cleaned):
                dropped_non_hindi += 1
                continue

            if len(cleaned.split()) < MIN_WORDS_AFTER_CLEAN:
                dropped_short += 1
                continue

            fout.write(json.dumps({"text": cleaned}, ensure_ascii=False) + "\n")
            kept += 1

    print("=" * 56)
    print("Strict Hindi-only cleaning finished")
    print("=" * 56)
    print(f"Input file     : {INPUT_PATH}")
    print(f"Output file    : {OUTPUT_PATH}")
    print(f"Total records  : {total:,}")
    print(f"Kept records   : {kept:,}")
    print(f"Dropped empty  : {dropped_empty:,}")
    print(f"Dropped short  : {dropped_short:,}")
    print(f"Dropped non-hi : {dropped_non_hindi:,}")


if __name__ == "__main__":
    main()
