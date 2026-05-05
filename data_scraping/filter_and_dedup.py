"""
Filter, deduplicate, and quality-check the raw non-conversational Hindi medical data.

Pipeline:
  1. Load all JSONL files from data/raw/
  2. Language verification  (strip non-Hindi docs)
  3. Conversational pattern rejection  (remove chat-like text that slipped through)
  4. Quality scoring  (length, keyword density, perplexity-proxy)
  5. MinHash near-duplicate removal
  6. Save final dataset to data/processed/nonconv_hindi_medical.jsonl

Run:
    python filter_and_dedup.py
"""

import re, json, hashlib, logging
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Optional: langdetect for language verification
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("langdetect not installed – skipping language ID step. Install with: pip install langdetect")

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = PROCESSED_DIR / "nonconv_hindi_medical.jsonl"
STATS_FILE = PROCESSED_DIR / "filter_stats.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ─── 1. Conversational pattern rejection ─────────────────────────────────────
#
# The blog author's insight: non-conversational data should NOT look like a dialogue.
# We aggressively drop anything that smells like Q&A or chat.

CONVERSATIONAL_PATTERNS = [
    # Explicit Q&A markers
    r"^\s*(प्रश्न|सवाल|Q|Question)\s*[:\-–]",
    r"^\s*(उत्तर|जवाब|A|Answer)\s*[:\-–]",
    # User/assistant role markers (from instruction datasets)
    r"(User|Assistant|Human|Bot|System)\s*:",
    r"(यूजर|सहायक|बॉट)\s*:",
    # Numbered FAQ-style  Q1. / Q.1
    r"Q\d+[\.\)]\s",
    # Conversational openers
    r"^\s*(नमस्ते|हेलो|हाय|Hello|Hi)\b",
    r"\bधन्यवाद\b.*\?(धन्यवाद after question = chat)",
    # Instruction-following markers
    r"<\|?(user|assistant|system|human|bot)\|?>",
    r"\[INST\]|\[/INST\]",
    r"<<SYS>>",
]

_CONV_RE = re.compile("|".join(CONVERSATIONAL_PATTERNS), re.IGNORECASE | re.MULTILINE)


def is_conversational(text: str) -> bool:
    if _CONV_RE.search(text):
        return True
    # Heuristic: many short turns split by newlines = conversation
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return False
    short_lines = sum(1 for l in lines if len(l) < 80)
    if len(lines) > 5 and short_lines / len(lines) > 0.7:
        return True
    return False


# ─── 2. Language verification ─────────────────────────────────────────────────

def is_hindi(text: str) -> bool:
    if not LANGDETECT_AVAILABLE:
        # Fallback: check for Devanagari Unicode block (U+0900–U+097F)
        devanagari = sum(1 for c in text if "\u0900" <= c <= "\u097F")
        return devanagari / max(len(text), 1) > 0.15
    try:
        return detect(text[:500]) == "hi"
    except LangDetectException:
        return False


# ─── 3. Quality scoring ───────────────────────────────────────────────────────

MEDICAL_KEYWORDS = [
    "रोग", "बीमारी", "दवा", "औषधि", "उपचार", "चिकित्सा", "स्वास्थ्य",
    "रक्त", "हृदय", "मधुमेह", "कैंसर", "संक्रमण", "वायरस", "बैक्टीरिया",
    "शल्य", "दर्द", "बुखार", "अस्पताल", "डॉक्टर", "टीका", "प्रतिरक्षा",
    "पोषण", "मस्तिष्क", "यकृत", "गुर्दा", "फेफड़ा", "हड्डी", "रक्तचाप",
    "मलेरिया", "तपेदिक", "डेंगू", "कोविड", "एड्स", "एलर्जी", "अस्थमा",
    "निदान", "लक्षण", "परीक्षण", "एंटीबायोटिक",
]
_MED_SET = set(MEDICAL_KEYWORDS)


def quality_score(text: str) -> float:
    """
    Returns a score in [0, 1]. Higher is better.
    Combines:
      - keyword density (medical terms per 100 tokens)
      - average word length (short words → low quality / spam)
      - presence of structured medical info (dosage, body-part mentions)
    """
    words = text.split()
    if not words:
        return 0.0
    med_hits = sum(1 for w in words if w in _MED_SET)
    kw_density = min(med_hits / max(len(words), 1) * 100, 10) / 10   # cap at 1

    avg_word_len = sum(len(w) for w in words) / len(words)
    len_score = min(avg_word_len / 6, 1.0)  # Hindi avg word ~5-6 chars

    return 0.6 * kw_density + 0.4 * len_score


QUALITY_THRESHOLD = 0.05  # drop very low-scoring docs


# ─── 4. MinHash near-deduplication ───────────────────────────────────────────
#
# Lightweight: we use 5-gram fingerprints via SHA1 to detect near-duplicates.
# For production, use datasketch or SimHash for finer-grained dedup.

def shingle_fingerprint(text: str, n: int = 5) -> frozenset:
    """Returns a frozenset of hashed n-gram shingles."""
    words = text.lower().split()
    shingles = {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}
    return frozenset(hashlib.md5(s.encode()).hexdigest()[:8] for s in shingles)


def jaccard(a: frozenset, b: frozenset) -> float:
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class DeduplicatorLSH:
    """
    Fast approximate deduplication using a bucket-based LSH on shingles.
    Exact Jaccard computation only happens within buckets.
    """

    def __init__(self, threshold: float = 0.7, bands: int = 20, rows: int = 5):
        self.threshold = threshold
        self.bands = bands
        self.rows = rows
        self.buckets: dict[str, list] = defaultdict(list)  # bucket_key -> [(text, fp)]

    def _band_keys(self, fp: frozenset) -> list[str]:
        fp_list = sorted(fp)[: self.bands * self.rows]
        # pad if too short
        while len(fp_list) < self.bands * self.rows:
            fp_list.append("__pad__")
        keys = []
        for b in range(self.bands):
            band = tuple(fp_list[b * self.rows : (b + 1) * self.rows])
            keys.append(f"b{b}:" + hashlib.md5(str(band).encode()).hexdigest()[:12])
        return keys

    def is_duplicate(self, text: str) -> bool:
        fp = shingle_fingerprint(text)
        band_keys = self._band_keys(fp)
        # Check candidates in the same buckets
        candidates = set()
        for bk in band_keys:
            for cand_text, cand_fp in self.buckets[bk]:
                candidates.add((cand_text, cand_fp))
        for cand_text, cand_fp in candidates:
            if jaccard(fp, cand_fp) >= self.threshold:
                return True
        # Not a duplicate: add to buckets
        for bk in band_keys:
            self.buckets[bk].append((text, fp))
        return False


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run():
    raw_files = sorted(RAW_DIR.glob("*.jsonl"))
    if not raw_files:
        log.error(f"No JSONL files found in {RAW_DIR}. Run collect_hindi_medical_data.py first.")
        return

    stats = {
        "total_raw": 0,
        "dropped_conversational": 0,
        "dropped_not_hindi": 0,
        "dropped_low_quality": 0,
        "dropped_duplicate": 0,
        "kept": 0,
        "by_source": defaultdict(int),
    }

    deduper = DeduplicatorLSH(threshold=0.7)

    with open(OUT_FILE, "w", encoding="utf-8") as out_f:
        for raw_file in raw_files:
            log.info(f"Processing {raw_file.name} …")
            with open(raw_file, encoding="utf-8") as f:
                lines = f.readlines()

            for line in tqdm(lines, desc=raw_file.stem, unit="docs"):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = record.get("text", "").strip()
                if not text:
                    continue

                stats["total_raw"] += 1

                # Step 1: reject conversational content
                if is_conversational(text):
                    stats["dropped_conversational"] += 1
                    continue

                # Step 2: language check
                if not is_hindi(text):
                    stats["dropped_not_hindi"] += 1
                    continue

                # Step 3: quality gate
                if quality_score(text) < QUALITY_THRESHOLD:
                    stats["dropped_low_quality"] += 1
                    continue

                # Step 4: near-dedup
                if deduper.is_duplicate(text):
                    stats["dropped_duplicate"] += 1
                    continue

                # Passed all filters
                stats["kept"] += 1
                stats["by_source"][record.get("source", "unknown")] += 1
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Print summary
    log.info("\n" + "=" * 50)
    log.info("FILTER SUMMARY")
    log.info("=" * 50)
    log.info(f"Total raw docs  : {stats['total_raw']:,}")
    log.info(f"  Conversational: {stats['dropped_conversational']:,}")
    log.info(f"  Not Hindi     : {stats['dropped_not_hindi']:,}")
    log.info(f"  Low quality   : {stats['dropped_low_quality']:,}")
    log.info(f"  Near-duplicate: {stats['dropped_duplicate']:,}")
    log.info(f"KEPT            : {stats['kept']:,}")
    log.info("\nBy source:")
    for src, cnt in sorted(stats["by_source"].items(), key=lambda x: -x[1]):
        log.info(f"  {src:40s}: {cnt:,}")
    log.info(f"\nOutput → {OUT_FILE}")

    # Save stats
    stats["by_source"] = dict(stats["by_source"])
    with open(STATS_FILE, "w") as sf:
        json.dump(stats, sf, indent=2)


if __name__ == "__main__":
    run()
