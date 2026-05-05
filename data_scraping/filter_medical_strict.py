"""
Strict medical relevance filter + 40 M-token sampler
=====================================================
Takes data/processed/nonconv_hindi_medical.jsonl (full dataset),
keeps only documents with sufficient medical keyword density,
then randomly samples down to TARGET_TOKENS.

Output: data/processed/nonconv_hindi_medical_40M_clean.jsonl
"""

import json, random, re
from pathlib import Path
from tqdm import tqdm
import tiktoken

# ── Config ────────────────────────────────────────────────────────────────────
INPUT   = Path("data/processed/nonconv_hindi_medical.jsonl")
OUTPUT  = Path("data/processed/nonconv_hindi_medical_40M_clean.jsonl")
TARGET_TOKENS   = 40_000_000
MIN_DOC_TOKENS  = 100        # drop very short stubs
# Minimum medical keyword hits per 100 words (pure keyword density, no padding)
MED_DENSITY_THRESHOLD = 0.5   # at least 0.5 medical-keyword hits per 100 words
MED_MIN_HITS = 2               # absolute minimum keyword matches in the whole doc
RANDOM_SEED = 42

# ── Extended Hindi medical keyword set ───────────────────────────────────────
MEDICAL_KEYWORDS = {
    # General medical
    "रोग", "बीमारी", "रोगी", "मरीज", "रोगाणु", "रोगजनक",
    "दवा", "दवाई", "औषधि", "औषध", "दवाएं",
    "उपचार", "इलाज", "चिकित्सा", "चिकित्सक", "उपाय",
    "स्वास्थ्य", "आरोग्य", "स्वस्थ",
    "लक्षण", "निदान", "परीक्षण", "जांच", "परीक्षा",
    "नुस्खा", "खुराक", "मात्रा",
    # Body systems / organs
    "रक्त", "रक्तचाप", "रक्तस्राव",
    "हृदय", "दिल", "हृदयरोग",
    "मस्तिष्क", "दिमाग", "तंत्रिका",
    "यकृत", "जिगर", "लिवर",
    "गुर्दा", "किडनी", "वृक्क",
    "फेफड़ा", "फेफड़े", "श्वसन",
    "हड्डी", "अस्थि", "जोड़",
    "आंत", "पाचन", "पेट", "आमाशय",
    "त्वचा", "चर्म",
    "आंख", "नेत्र",
    "कान", "श्रवण",
    "नाक", "श्वासनली",
    "मांसपेशी", "पेशी",
    "अग्न्याशय", "थायरॉइड", "थायराइड",
    # Diseases
    "मधुमेह", "शुगर", "डायबिटीज",
    "कैंसर", "अर्बुद", "ट्यूमर",
    "संक्रमण", "इन्फेक्शन",
    "वायरस", "बैक्टीरिया", "जीवाणु", "विषाणु",
    "बुखार", "ज्वर",
    "दर्द", "पीड़ा",
    "मलेरिया", "तपेदिक", "क्षय", "टीबी",
    "डेंगू", "चिकनगुनिया",
    "कोविड", "कोरोना",
    "एड्स", "एचआईवी",
    "एलर्जी",
    "अस्थमा", "दमा",
    "पोलियो",
    "हैजा",
    "टाइफाइड",
    "हेपेटाइटिस", "पीलिया",
    "मेनिनजाइटिस",
    "अल्जाइमर",
    "पार्किंसन",
    "गठिया", "आर्थराइटिस",
    "एनीमिया", "रक्ताल्पता",
    "उच्च रक्तचाप", "निम्न रक्तचाप",
    "मोटापा",
    "कुपोषण",
    # Treatments / procedures
    "शल्य", "शल्यचिकित्सा", "ऑपरेशन", "सर्जरी",
    "टीका", "टीकाकरण", "वैक्सीन",
    "प्रतिरक्षा", "इम्युनिटी",
    "एंटीबायोटिक", "एंटीवायरल",
    "अस्पताल", "हॉस्पिटल",
    "डॉक्टर", "चिकित्सक", "विशेषज्ञ",
    "नर्स", "परिचारिका",
    "एक्स-रे", "एमआरआई", "सीटी स्कैन", "अल्ट्रासाउंड",
    "प्रयोगशाला", "लैब",
    "रोकथाम", "बचाव", "निवारण",
    "पुनर्वास", "रिकवरी",
    "पोषण", "आहार",
    "व्यायाम", "योग",
}

def medical_quality_score(text: str) -> tuple[float, int]:
    """
    Returns (keyword_density_per_100_words, absolute_hit_count).
    Purely keyword-based – word-length padding removed.
    """
    words = text.split()
    if not words:
        return 0.0, 0
    med_hits = sum(1 for w in words if w.strip("।,.!?") in MEDICAL_KEYWORDS)
    density = med_hits / max(len(words), 1) * 100   # hits per 100 words
    return density, med_hits


def main():
    enc = tiktoken.get_encoding("cl100k_base")

    print("── Pass 1: strict medical relevance filter ──────────────────────────")
    passed = []
    stats = {"total": 0, "too_short": 0, "low_quality": 0, "passed": 0}

    with open(INPUT, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Filtering"):
        obj = json.loads(line)
        text = obj["text"]
        stats["total"] += 1

        # Token count gate
        n_tokens = len(enc.encode(text))
        if n_tokens < MIN_DOC_TOKENS:
            stats["too_short"] += 1
            continue

        # Medical relevance gate
        density, hits = medical_quality_score(text)
        if hits < MED_MIN_HITS or density < MED_DENSITY_THRESHOLD:
            stats["low_quality"] += 1
            continue

        passed.append((line, n_tokens, density))
        stats["passed"] += 1

    print(f"\n  Total docs     : {stats['total']:>10,}")
    print(f"  Dropped – short: {stats['too_short']:>10,}")
    print(f"  Dropped – off-topic: {stats['low_quality']:>10,}")
    print(f"  Passed filter  : {stats['passed']:>10,}")

    available_tokens = sum(n for _, n, _ in passed)
    print(f"  Tokens available after filter: {available_tokens:,}")

    if available_tokens < TARGET_TOKENS:
        print(f"\n⚠  Only {available_tokens:,} tokens pass the filter "
              f"(< {TARGET_TOKENS:,} target).")
        print("  Consider lowering MED_DENSITY_THRESHOLD, adding more sources,")
        print("  or re-running collect_hindi_medical_data.py with --max-per-source 40000.")
        target = available_tokens
    else:
        target = TARGET_TOKENS

    print(f"\n── Pass 2: random sample down to {target:,} tokens ─────────────────")
    # Sort by score descending, but shuffle within buckets to avoid bias
    random.seed(RANDOM_SEED)
    random.shuffle(passed)           # shuffle first for randomness
    passed.sort(key=lambda x: x[2], reverse=True)  # then prefer higher-scored docs

    total_tokens = 0
    kept = 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w", encoding="utf-8") as out:
        for line, n_tokens, score in tqdm(passed, desc="Sampling"):
            if total_tokens + n_tokens > target:
                continue
            out.write(line)
            total_tokens += n_tokens
            kept += 1
            if total_tokens >= target:
                break

    print(f"\n── Results ──────────────────────────────────────────────────────────")
    print(f"  Documents kept  : {kept:,}")
    print(f"  Total tokens    : {total_tokens:,}")
    print(f"  Output file     : {OUTPUT}")


if __name__ == "__main__":
    main()
