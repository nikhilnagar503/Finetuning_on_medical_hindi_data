"""
Non-Conversational Hindi Medical Data Collection Pipeline
==========================================================
Collects plain medical text (NOT Q&A / chat format) from multiple sources:
  1. Hindi Wikipedia      – medical/health article bodies
  2. AI4Bharat Sangraha   – curated Hindi web text, med-filtered
  3. Sangraha Unverified  – large web-crawled Hindi, med-filtered
  4. Varta Hindi News     – Hindi news corpus (replaces Vikaspedia)
  5. Sangraha Synthetic   – LLM-generated + web-sourced Hindi
  6. CulturaX Hindi       – large multilingual web corpus, med-filtered
  7. NHP (nhp.gov.in)     – National Health Portal of India (scraped)
  8. AYUSH (ayush.gov.in) – Ministry of AYUSH portal (scraped)

Target: ~40 M tokens total after filtering.

All output is saved as JSONL with fields:
  { "text": "...", "source": "...", "lang": "hi" }
"""

import os, re, json, time, logging
import requests
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
from datasets import load_dataset

# ─── Config ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = "collection.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# Minimum token-like length for a sample to be kept
MIN_CHARS = 200
MAX_CHARS = 8000         # very long docs hurt packing; split them later


# ─── Medical keyword filter (Hindi + transliterated + English) ──────────────

MEDICAL_KEYWORDS = [
    # Hindi
    "रोग", "बीमारी", "दवा", "औषधि", "उपचार", "चिकित्सा", "स्वास्थ्य",
    "रक्त", "हृदय", "श्वसन", "मधुमेह", "कैंसर", "संक्रमण", "वायरस",
    "बैक्टीरिया", "शल्य", "शल्यचिकित्सा", "दर्द", "बुखार", "अस्पताल",
    "डॉक्टर", "चिकित्सक", "नुस्खा", "पर्चा", "टीका", "प्रतिरक्षा",
    "पोषण", "अंग", "मस्तिष्क", "तंत्रिका", "यकृत", "गुर्दा", "फेफड़ा",
    "हड्डी", "मांसपेशी", "त्वचा", "नेत्र", "श्रवण", "पाचन", "रक्तचाप",
    "मलेरिया", "टी.बी.", "तपेदिक", "डेंगू", "कोविड", "एड्स", "एचआईवी",
    "एलर्जी", "अस्थमा", "मोटापा", "थायरॉयड", "पक्षाघात", "अपस्मार",
    "गर्भावस्था", "प्रसव", "नवजात", "बाल रोग", "वृद्धावस्था",
    "एंटीबायोटिक", "प्रतिजैविक", "एनेस्थीसिया", "विकिरण", "कीमोथेरेपी",
    "निदान", "लक्षण", "परीक्षण", "प्रयोगशाला", "एक्स-रे", "एमआरआई",
    # English terms often embedded even in Hindi medical text
    "hospital", "medicine", "disease", "treatment", "patient", "doctor",
    "surgery", "diagnosis", "symptom", "vaccine", "virus", "bacteria",
    "cancer", "diabetes", "blood pressure", "infection", "therapy",
    "pharmacy", "clinical", "medical",
]

_KW_PATTERN = re.compile("|".join(re.escape(k) for k in MEDICAL_KEYWORDS), re.IGNORECASE)


def is_medical(text: str) -> bool:
    return bool(_KW_PATTERN.search(text))


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    # Remove edit-section markers, ref tags, etc. that sometimes leak from Wikipedia
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"==.*?==", "", text)
    return text.strip()


def write_jsonl(records: list[dict], path: Path):
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"  → {len(records)} records appended to {path}")


# ─── Source 1: Hindi Wikipedia ───────────────────────────────────────────────

def collect_hindi_wikipedia(max_samples: int = 15_000):
    log.info("=== Hindi Wikipedia ===")
    out = OUTPUT_DIR / "hindi_wikipedia.jsonl"

    # The 'wikipedia' dataset for Hindi is small enough to stream fully
    ds = load_dataset("wikimedia/wikipedia", "20231101.hi", split="train", streaming=True)

    records, skipped = [], 0
    for article in tqdm(ds, desc="Wikipedia-hi", unit="articles"):
        text = clean_text(article.get("text", ""))
        if len(text) < MIN_CHARS:
            skipped += 1
            continue
        if not is_medical(text):
            skipped += 1
            continue

        # Split long articles into paragraphs that stay within MAX_CHARS
        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) >= MIN_CHARS]
        for para in paragraphs:
            if len(para) > MAX_CHARS:
                # chunk by sentence boundary
                for chunk in chunk_text(para, MAX_CHARS):
                    records.append({"text": chunk, "source": "wikipedia_hi", "lang": "hi"})
            else:
                records.append({"text": para, "source": "wikipedia_hi", "lang": "hi"})

        if len(records) >= max_samples:
            break

    write_jsonl(records, out)
    log.info(f"Wikipedia done: {len(records)} kept, {skipped} skipped\n")


# ─── Source 2: AI4Bharat Sangraha (Hindi) ────────────────────────────────────

def collect_sangraha(max_samples: int = 20_000):
    """
    Sangraha is a large, quality-filtered multilingual dataset.
    The Hindi subset contains web articles – mostly non-conversational by nature.
    HuggingFace repo: ai4bharat/sangraha
    Subset used: 'verified/hin'  (human-verified, highest quality)
    """
    log.info("=== AI4Bharat Sangraha (verified Hindi) ===")
    out = OUTPUT_DIR / "sangraha_hi.jsonl"

    ds = load_dataset(
        "ai4bharat/sangraha",
        "verified",
        split="hin",
        streaming=True,
    )

    records, skipped = [], 0
    for sample in tqdm(ds, desc="Sangraha-hi", unit="docs"):
        text = clean_text(sample.get("text", ""))
        if len(text) < MIN_CHARS:
            skipped += 1
            continue
        if not is_medical(text):
            skipped += 1
            continue
        for chunk in chunk_text(text, MAX_CHARS):
            records.append({"text": chunk, "source": "sangraha_verified_hi", "lang": "hi"})

        if len(records) >= max_samples:
            break

    write_jsonl(records, out)
    log.info(f"Sangraha done: {len(records)} kept, {skipped} skipped\n")


# ─── Source 3: mC4 Hindi ─────────────────────────────────────────────────────

def collect_mc4_hindi(max_samples: int = 15_000):
    """
    AI4Bharat Sangraha – unverified Hindi split.
    Much larger than the verified split; web-crawled Hindi text.
    Uses the same dataset as Sangraha but the 'unverified' config.
    """
    log.info("=== Sangraha Unverified Hindi (web-crawled) ===")
    out = OUTPUT_DIR / "mc4_hi.jsonl"

    ds = load_dataset("ai4bharat/sangraha", "unverified", split="hin", streaming=True)

    records, skipped = [], 0
    for sample in tqdm(ds, desc="mC4-hi", unit="docs"):
        text = clean_text(sample.get("text", ""))
        if len(text) < MIN_CHARS:
            skipped += 1
            continue
        if not is_medical(text):
            skipped += 1
            continue
        for chunk in chunk_text(text, MAX_CHARS):
            records.append({"text": chunk, "source": "mc4_hi", "lang": "hi"})

        if len(records) >= max_samples:
            break

    write_jsonl(records, out)
    log.info(f"Sangraha-unverified done: {len(records)} kept, {skipped} skipped\n")


# ─── Source 4: Vikaspedia (Indian Gov Health Portal in Hindi) ────────────────

VIKASPEDIA_SEED_URLS = [
    "https://hi.vikaspedia.in/health",
    "https://hi.vikaspedia.in/health/diseases-and-conditions",
    "https://hi.vikaspedia.in/health/nutrition",
    "https://hi.vikaspedia.in/health/women-health",
    "https://hi.vikaspedia.in/health/child-health",
    "https://hi.vikaspedia.in/health/first-aid",
    "https://hi.vikaspedia.in/health/ayush",
    "https://hi.vikaspedia.in/health/hygiene-sanitation",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; MedHindiBot/1.0; "
        "+https://github.com/your_project)"
    )
}


def scrape_vikaspedia(max_pages: int = 150, delay: float = 1.5):
    """
    Vikaspedia uses Next.js (JavaScript-rendered) so BeautifulSoup cannot scrape it.
    Replaced with Varta – a large Hindi news corpus from AI4Bharat.
    News articles are non-conversational prose, good for medical/health topics.
    HuggingFace repo: rahular/varta
    """
    log.info("=== Varta Hindi News Corpus (replaces Vikaspedia) ===")
    out = OUTPUT_DIR / "vikaspedia_hi.jsonl"

    ds = load_dataset("rahular/varta", split="train", streaming=True)

    record_list = []
    skipped = 0
    for sample in tqdm(ds, desc="Varta-hi", unit="docs"):
        # Varta has 'headline' and 'content' fields; use content
        text = clean_text(sample.get("content", sample.get("text", "")))
        if len(text) < MIN_CHARS:
            skipped += 1
            continue
        if not is_medical(text):
            skipped += 1
            continue
        for chunk in chunk_text(text, MAX_CHARS):
            record_list.append({"text": chunk, "source": "varta_hi", "lang": "hi"})

        if len(record_list) >= max_pages * 10:
            break

    write_jsonl(record_list, out)
    log.info(f"Varta done: {len(record_list)} chunks kept, {skipped} skipped\n")


# ─── Source 5: CC-100 Hindi ───────────────────────────────────────────────────

def collect_cc100_hindi(max_samples: int = 15_000):
    """
    AI4Bharat Sangraha – synthetic Hindi split.
    Replaces CC-100 which uses a deprecated loading script.
    Contains LLM-generated and web-sourced Hindi text.
    """
    log.info("=== Sangraha Synthetic Hindi ===")
    out = OUTPUT_DIR / "cc100_hi.jsonl"

    ds = load_dataset("ai4bharat/sangraha", "synthetic", split="hin_Deva", streaming=True)

    records, skipped = [], 0
    for sample in tqdm(ds, desc="CC100-hi", unit="docs"):
        text = clean_text(sample.get("text", ""))
        if len(text) < MIN_CHARS:
            skipped += 1
            continue
        if not is_medical(text):
            skipped += 1
            continue
        for chunk in chunk_text(text, MAX_CHARS):
            records.append({"text": chunk, "source": "cc100_hi", "lang": "hi"})

        if len(records) >= max_samples:
            break

    write_jsonl(records, out)
    log.info(f"Sangraha-synthetic done: {len(records)} kept, {skipped} skipped\n")


# ─── Source 6: CulturaX Hindi ────────────────────────────────────────────────

def collect_culturax_hindi(max_samples: int = 30_000):
    """
    CulturaX: A Large, Multilingual and Filtered Web Corpus.
    HuggingFace: uonlp/CulturaX  |  config: hi  |  split: train
    Fields: url, text
    78 GB+ Hindi corpus – medical content extracted via keyword filter.
    """
    log.info("=== CulturaX Hindi ===")
    out = OUTPUT_DIR / "culturax_hi.jsonl"

    ds = load_dataset("uonlp/CulturaX", "hi", split="train", streaming=True)

    records, skipped = [], 0
    for sample in tqdm(ds, desc="CulturaX-hi", unit="docs"):
        text = clean_text(sample.get("text", ""))
        if len(text) < MIN_CHARS:
            skipped += 1
            continue
        if not is_medical(text):
            skipped += 1
            continue
        for chunk in chunk_text(text, MAX_CHARS):
            records.append({"text": chunk, "source": "culturax_hi", "lang": "hi"})
        if len(records) >= max_samples:
            break

    write_jsonl(records, out)
    log.info(f"CulturaX done: {len(records)} kept, {skipped} skipped\n")


# ─── Source 7: National Health Portal (NHP) web scraping ─────────────────────

NHP_BASE = "https://www.nhp.gov.in"
NHP_SEED_URLS = [
    "https://www.nhp.gov.in/disease-a-z",
    "https://www.nhp.gov.in/healthlyliving/",
    "https://www.nhp.gov.in/nutritionlive/",
    "https://www.nhp.gov.in/ayush",
    "https://www.nhp.gov.in/maternal-health",
    "https://www.nhp.gov.in/child-health",
    "https://www.nhp.gov.in/senior-citizen-health",
]


def scrape_nhp(max_pages: int = 300, delay: float = 1.5):
    """
    National Health Portal of India – authoritative Hindi medical content.
    Crawls disease/health article pages, keeps Devanagari-dominant paragraphs.
    """
    log.info("=== National Health Portal (NHP) ===")
    out = OUTPUT_DIR / "nhp_hi.jsonl"

    visited: set[str] = set()
    to_visit = list(NHP_SEED_URLS)
    records: list[dict] = []
    pages_scraped = 0

    session = requests.Session()
    session.headers.update(HEADERS)

    while to_visit and pages_scraped < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "lxml")

            # Try common content containers
            content_div = (
                soup.find("div", class_=re.compile(r"content|article|main|body", re.I))
                or soup.find("article")
                or soup.find("main")
                or soup.find("div", id=re.compile(r"content|main", re.I))
            )
            container = content_div or soup.find("body")
            if container:
                # Remove nav/script/style noise
                for tag in container.find_all(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()

                text_parts = []
                for elem in container.find_all(["p", "li", "td"]):
                    t = elem.get_text(separator=" ", strip=True)
                    dev_count = sum(1 for c in t if "\u0900" <= c <= "\u097F")
                    if dev_count / max(len(t), 1) > 0.2 and len(t) >= 80:
                        text_parts.append(t)

                if text_parts:
                    full_text = clean_text(" ".join(text_parts))
                    if len(full_text) >= MIN_CHARS and is_medical(full_text):
                        for chunk in chunk_text(full_text, MAX_CHARS):
                            records.append({"text": chunk, "source": "nhp_hi", "lang": "hi"})
                        pages_scraped += 1

            # Collect more links on the same domain
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href.startswith("/"):
                    href = NHP_BASE + href
                if href.startswith(NHP_BASE) and href not in visited and len(to_visit) < 2000:
                    to_visit.append(href)

            time.sleep(delay)

        except Exception as exc:
            log.warning(f"NHP scrape error ({url}): {exc}")

    write_jsonl(records, out)
    log.info(f"NHP done: {len(records)} chunks from {pages_scraped} pages\n")


# ─── Source 8: AYUSH Portal web scraping ─────────────────────────────────────

AYUSH_BASE = "https://ayush.gov.in"
AYUSH_SEED_URLS = [
    "https://ayush.gov.in/ayurveda",
    "https://ayush.gov.in/yoga",
    "https://ayush.gov.in/naturopathy",
    "https://ayush.gov.in/unani",
    "https://ayush.gov.in/siddha",
    "https://ayush.gov.in/homeopathy",
    "https://ayush.gov.in/page/about-ayush",
]


def scrape_ayush(max_pages: int = 200, delay: float = 1.5):
    """
    Ministry of AYUSH (Ayurveda, Yoga, Naturopathy, Unani, Siddha, Homeopathy).
    Extracts Hindi paragraphs from ayush.gov.in.
    """
    log.info("=== AYUSH Portal ===")
    out = OUTPUT_DIR / "ayush_hi.jsonl"

    visited: set[str] = set()
    to_visit = list(AYUSH_SEED_URLS)
    records: list[dict] = []
    pages_scraped = 0

    session = requests.Session()
    session.headers.update(HEADERS)

    while to_visit and pages_scraped < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                continue
            soup = BeautifulSoup(resp.text, "lxml")

            body = soup.find("body")
            if body:
                for tag in body.find_all(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()

                text_parts = []
                for elem in body.find_all(["p", "article", "section"]):
                    t = elem.get_text(separator=" ", strip=True)
                    dev_count = sum(1 for c in t if "\u0900" <= c <= "\u097F")
                    if dev_count / max(len(t), 1) > 0.2 and len(t) >= 80:
                        text_parts.append(t)

                if text_parts:
                    full_text = clean_text(" ".join(text_parts))
                    if len(full_text) >= MIN_CHARS and is_medical(full_text):
                        for chunk in chunk_text(full_text, MAX_CHARS):
                            records.append({"text": chunk, "source": "ayush_hi", "lang": "hi"})
                        pages_scraped += 1

            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href.startswith("/"):
                    href = AYUSH_BASE + href
                if href.startswith(AYUSH_BASE) and href not in visited and len(to_visit) < 1000:
                    to_visit.append(href)

            time.sleep(delay)

        except Exception as exc:
            log.warning(f"AYUSH scrape error ({url}): {exc}")

    write_jsonl(records, out)
    log.info(f"AYUSH done: {len(records)} chunks from {pages_scraped} pages\n")


# ─── Utility: chunk long text ─────────────────────────────────────────────────

def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text at sentence boundaries to stay within max_chars."""
    if len(text) <= max_chars:
        return [text]
    # Split on Hindi/English sentence-ending punctuation
    sentences = re.split(r"(?<=[।.!?])\s+", text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    return chunks if chunks else [text[:max_chars]]


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collect non-conversational Hindi medical data (target: 40 M tokens)"
    )
    parser.add_argument(
        "--sources", nargs="+",
        choices=[
            "wikipedia", "sangraha", "mc4", "vikaspedia", "cc100",
            "culturax", "nhp", "ayush", "all",
        ],
        default=["all"],
        help="Which sources to collect from (default: all)",
    )
    parser.add_argument(
        "--max-per-source", type=int, default=25_000,
        help="Max samples per HuggingFace source (default 25k → ~20M raw tokens per source)",
    )
    args = parser.parse_args()

    run_all = "all" in args.sources

    if run_all or "wikipedia" in args.sources:
        collect_hindi_wikipedia(max_samples=args.max_per_source)

    if run_all or "sangraha" in args.sources:
        collect_sangraha(max_samples=args.max_per_source)

    if run_all or "mc4" in args.sources:
        collect_mc4_hindi(max_samples=args.max_per_source)

    if run_all or "vikaspedia" in args.sources:
        scrape_vikaspedia(max_pages=500)

    if run_all or "cc100" in args.sources:
        collect_cc100_hindi(max_samples=args.max_per_source)

    if run_all or "culturax" in args.sources:
        collect_culturax_hindi(max_samples=args.max_per_source)

    if run_all or "nhp" in args.sources:
        scrape_nhp(max_pages=300)

    if run_all or "ayush" in args.sources:
        scrape_ayush(max_pages=200)

    log.info("All sources collected. Run filter_and_dedup.py next.")
