"""
audit_and_clean.py
==================
Step 1 – AUDIT:  Scan nonconv_hindi_medical_40M_clean.jsonl and report issues.
Step 2 – CLEAN:  Fix / drop each issue and write the final training-ready file.
Step 3 – FORMAT: Show how the training data should look for two common recipes:
            (a) Continued pre-training  (raw text, no template)
            (b) Instruction fine-tuning (chat template)

Run:
    python audit_and_clean.py
"""

import json, re
from pathlib import Path
from tqdm import tqdm
import tiktoken

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT  = Path("data/processed/nonconv_hindi_medical_40M_clean.jsonl")
OUTPUT = Path("data/processed/nonconv_hindi_medical_final.jsonl")

# ── Regex patterns ────────────────────────────────────────────────────────────
HTML_RE      = re.compile(r"<[^>]{1,200}>")
URL_RE       = re.compile(r"https?://\S+|www\.\S+")
LATIN_RE     = re.compile(r"[A-Za-z]+")
MIXED_TOKEN_RE = re.compile(r"\b(?=\w*[A-Za-z])\w+\b")
MULTI_NL_RE  = re.compile(r"\n{3,}")
MULTI_SP_RE  = re.compile(r"[ \t]{2,}")
SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?।])")
BOILER_RE    = re.compile(
    r"(कॉपीराइट|सर्वाधिकार\s*सुरक्षित|विज्ञापन\s*:|cookie\s*policy"
    r"|privacy\s*policy|terms\s*of\s*use|सभी अधिकार सुरक्षित)",
    re.IGNORECASE,
)
# Lines that are just a heading number / bullet / whitespace noise
JUNK_LINE_RE = re.compile(r"^\s*[\d\.\-\*]{1,4}\s*$")


def clean_text(text: str) -> str | None:
    """
    Returns cleaned text, or None if the doc should be dropped entirely.
    """
    # 1. Remove HTML tags
    text = HTML_RE.sub(" ", text)

    # 2. Remove URLs
    text = URL_RE.sub("", text)

    # 2.1 Remove all English/mixed-script tokens completely
    text = MIXED_TOKEN_RE.sub(" ", text)
    text = LATIN_RE.sub(" ", text)

    # 3. Remove boilerplate sentences (split on danda / newline, filter, rejoin)
    sentences = re.split(r"(?<=[।\n])", text)
    sentences = [s for s in sentences if not BOILER_RE.search(s)]
    text = "".join(sentences)

    # 4. Drop junk-only lines
    lines = text.split("\n")
    lines = [l for l in lines if not JUNK_LINE_RE.match(l)]
    text = "\n".join(lines)

    # 5. Normalise whitespace
    text = MULTI_NL_RE.sub("\n\n", text)
    text = MULTI_SP_RE.sub(" ", text)
    text = SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = text.strip()

    # 6. Drop if too short after cleaning
    if len(text.split()) < 60:
        return None

    # 7. Drop if no longer Hindi-dominant
    dev = sum(1 for c in text if "\u0900" <= c <= "\u097F")
    if dev / max(len(text), 1) < 0.25:
        return None

    # 8. Final hard check: reject any remaining Latin characters
    if LATIN_RE.search(text):
        return None

    return text


# ──────────────────────────────────────────────────────────────────────────────
# STEP 1 – AUDIT
# ──────────────────────────────────────────────────────────────────────────────
def audit():
    with open(INPUT, encoding="utf-8") as f:
        lines = f.readlines()

    counts = dict(html=0, urls=0, english=0, excess_ws=0, boilerplate=0,
                  very_short=0, non_hindi=0, total=len(lines))

    for line in lines:
        obj   = json.loads(line)
        text  = obj["text"]
        words = text.split()

        if HTML_RE.search(text):           counts["html"] += 1
        if URL_RE.search(text):            counts["urls"] += 1
        if LATIN_RE.search(text):          counts["english"] += 1
        if re.search(r"[ \t]{2,}", text):  counts["excess_ws"] += 1
        if BOILER_RE.search(text):         counts["boilerplate"] += 1
        if len(words) < 60:                counts["very_short"] += 1
        dev = sum(1 for c in text if "\u0900" <= c <= "\u097F")
        if dev / max(len(text), 1) < 0.25: counts["non_hindi"] += 1

    print("=" * 55)
    print("  AUDIT RESULTS")
    print("=" * 55)
    n = counts["total"]
    for k, v in counts.items():
        if k == "total":
            print(f"  {'Total documents':<28}: {v:>6,}")
        else:
            print(f"  {k:<28}: {v:>6,}  ({v/n*100:.1f}%)")
    print("=" * 55)
    return counts


# ──────────────────────────────────────────────────────────────────────────────
# STEP 2 – CLEAN
# ──────────────────────────────────────────────────────────────────────────────
def clean_and_save():
    enc = tiktoken.get_encoding("cl100k_base")

    with open(INPUT, encoding="utf-8") as f:
        lines = f.readlines()

    kept = []
    dropped = 0
    total_tokens = 0

    for line in tqdm(lines, desc="Cleaning"):
        obj  = json.loads(line)
        text = clean_text(obj["text"])
        if text is None:
            dropped += 1
            continue
        obj["text"] = text
        n = len(enc.encode(text))
        total_tokens += n
        kept.append(json.dumps(obj, ensure_ascii=False))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(kept) + "\n")

    print(f"\n  Docs kept    : {len(kept):,}")
    print(f"  Docs dropped : {dropped:,}")
    print(f"  Total tokens : {total_tokens:,}")
    print(f"  Output       : {OUTPUT}")
    return len(kept), total_tokens


# ──────────────────────────────────────────────────────────────────────────────
# STEP 3 – SHOW TRAINING FORMAT
# ──────────────────────────────────────────────────────────────────────────────
def show_training_format():
    with open(OUTPUT, encoding="utf-8") as f:
        sample = json.loads(f.readline())["text"][:600]

    print("""
╔══════════════════════════════════════════════════════════════╗
║         HOW TRAINING DATA SHOULD LOOK FOR YOUR USE CASE      ║
╚══════════════════════════════════════════════════════════════╝

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 OPTION A – Continued Pre-training / Domain Adaptation
 (Best for your non-conversational knowledge dataset)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Goal  : Teach the model Hindi medical vocabulary & facts.
 Format: Each sample is just the raw "text" field.
         No template, no instruction wrapper needed.

 JSONL line (what you already have):
   {"text": "मधुमेह एक चयापचय रोग है जिसमें..."}

 Training code just reads:
   sample["text"]   → feed directly as input_ids

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 OPTION B – Instruction Fine-tuning (chat template)
 (Use this AFTER Option A, for Q&A / chatbot behaviour)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Goal  : Teach the model to answer medical questions in Hindi.
 Format: Each sample has a system prompt, user question,
         and assistant answer.

 JSONL line:
   {
     "messages": [
       {"role": "system",    "content": "आप एक हिंदी चिकित्सा सहायक हैं।"},
       {"role": "user",      "content": "मधुमेह के लक्षण क्या हैं?"},
       {"role": "assistant", "content": "मधुमेह के मुख्य लक्षण हैं..."}
     ]
   }

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 RECOMMENDATION FOR YOUR CASE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Since your dataset is non-conversational reference text:

   Step 1 → Use OPTION A (continued pre-training) with this
             dataset. This gives the model Hindi medical knowledge.

   Step 2 → Separately create ~500-2000 Q&A pairs in Hindi
             (OPTION B format) to teach it HOW to answer.

 Your current dataset is PERFECT for Step 1 as-is.
""")
    print(f" Sample from your cleaned dataset:\n")
    print(f" {sample[:400]}...\n")


if __name__ == "__main__":
    print("\n── STEP 1: Audit ────────────────────────────────────────")
    audit()

    print("\n── STEP 2: Clean & Save ─────────────────────────────────")
    clean_and_save()

    print("\n── STEP 3: Training Format Guide ────────────────────────")
    show_training_format()
