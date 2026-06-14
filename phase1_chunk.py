"""
phase1_chunk.py
---------------
Reads scraped raw paper JSONs -> cleans text -> chunks into
~400-word segments -> saves as JSONL ready for Phase 2.

Output:
    data/chunks/all_chunks.jsonl     one JSON per line, each a chunk
    data/chunks/chunk_stats.json     counts per domain tag
    logs/chunk.log                   run log
"""

import os, json, re
from datetime import datetime
from pathlib import Path

RAW_DIR = Path("data/raw_papers")
CHUNK_DIR = Path("data/chunks")
RUN_LOG = Path("logs/chunk.log")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG.parent.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 400  # target words per chunk
CHUNK_OVERLAP = 50  # word overlap between consecutive chunks

# Same 11 domain categories as v1
DOMAIN_TAGS = {
    "atmosphere": [
        "atmospheric pressure",
        "co2",
        "greenhouse",
        "climate model",
        "global circulation",
        "warming",
        "temperature increase",
    ],
    "pioneer_biology": [
        "cyanobacteria",
        "pioneer organism",
        "lichen",
        "biofilm",
        "chlorella",
        "anabaena",
        "deinococcus",
        "halophile",
        "xerophile",
    ],
    "regolith": [
        "regolith",
        "soil formation",
        "mineral weathering",
        "basalt",
        "perchlorate",
        "iron oxide",
        "jarosite",
        "clay mineral",
    ],
    "radiation": [
        "uv radiation",
        "uv flux",
        "ionizing radiation",
        "cosmic ray",
        "radiation shielding",
        "dna damage",
        "ld50",
    ],
    "water": [
        "water activity",
        "brine",
        "ice",
        "subsurface water",
        "permafrost",
        "water activity",
        "hygroscopic",
    ],
    "nitrogen": [
        "nitrogen fixation",
        "nitrogenase",
        "ammonia",
        "n2",
        "nitrogen cycle",
        "nitrate",
    ],
    "oxygen": [
        "oxygen production",
        "photosynthesis",
        "electrolysis",
        "o2 partial pressure",
        "oxygenic",
    ],
    "isru": [
        "isru",
        "in-situ resource",
        "resource utilization",
        "propellant production",
        "sabatier",
        "electrolyzer",
    ],
    "timeline": [
        "terraforming timeline",
        "centuries",
        "millennia",
        "stage",
        "phase transition",
        "habitability threshold",
    ],
    "survival": [
        "microbial survival",
        "desiccation",
        "freeze-thaw",
        "sporulation",
        "endospore",
        "metabolic reactivation",
    ],
}


def log_msg(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def auto_tag(text):
    text_lower = text.lower()
    tags = []
    for domain, keywords in DOMAIN_TAGS.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(domain)
    return tags if tags else ["general"]


def clean_text(text):
    text = re.sub(r"\[[\d,\s]+\]", "", text)
    text = re.sub(r"\([\w\s]+,\s*\d{4}\)", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def main():
    raw_files = list(RAW_DIR.glob("*.json"))
    log_msg("=" * 60)
    log_msg(f"TERRA-MARS Chunker v1.5")
    log_msg(f"Input:  {RAW_DIR} ({len(raw_files)} files)")
    log_msg(f"Output: {CHUNK_DIR}")
    log_msg("=" * 60)

    all_chunks = []
    domain_counts = {}
    skipped = 0
    skipped_pmc = 0
    skipped_short = 0

    for fpath in raw_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                paper = json.load(f)
        except Exception as e:
            log_msg(f"  Skip {fpath.name}: {e}")
            skipped += 1
            continue

        title = paper.get("title", "").strip()
        abstract = paper.get("abstract", "").strip()
        source = paper.get("source", "unknown")

        # Need at least a title + meaningful abstract
        if len(abstract) < 100:
            skipped += 1
            if source == "pmc":
                skipped_pmc += 1
            else:
                skipped_short += 1
            continue

        full_text = f"{title}. {abstract}"
        full_text = clean_text(full_text)
        tags = auto_tag(full_text)
        chunks = chunk_text(full_text)

        for i, chunk in enumerate(chunks):
            record = {
                "chunk_id": f"{fpath.stem}_chunk{i:03d}",
                "title": title,
                "text": chunk,
                "source": source,
                "url": paper.get("url", ""),
                "published": paper.get("published", ""),
                "license": paper.get("license", "open_access"),
                "query": paper.get("query", ""),
                "domains": tags,
                "chunk_idx": i,
                "total_chunks": len(chunks),
            }
            all_chunks.append(record)
            for tag in tags:
                domain_counts[tag] = domain_counts.get(tag, 0) + 1

    # Save chunks
    out_path = CHUNK_DIR / "all_chunks.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Save stats
    stats = {
        "input_files": len(raw_files),
        "papers_skipped": skipped,
        "skipped_pmc_no_abs": skipped_pmc,
        "skipped_short_abstract": skipped_short,
        "papers_chunked": len(raw_files) - skipped,
        "total_chunks": len(all_chunks),
        "domain_counts": domain_counts,
        "avg_chunks_per_paper": round(
            len(all_chunks) / max(1, len(raw_files) - skipped), 1
        ),
    }
    with open(CHUNK_DIR / "chunk_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    log_msg("")
    log_msg("=" * 60)
    log_msg(f"DONE")
    log_msg(f"Input papers:         {len(raw_files)}")
    log_msg(f"Skipped (PMC empty):  {skipped_pmc}")
    log_msg(f"Skipped (other):      {skipped_short}")
    log_msg(f"Papers chunked:       {len(raw_files) - skipped}")
    log_msg(f"Total chunks:         {len(all_chunks)}")
    log_msg(f"Output: {out_path}")
    log_msg("")
    log_msg("Domain breakdown:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        log_msg(f"  {domain:20s} {count:5d}")
    log_msg("=" * 60)


if __name__ == "__main__":
    main()
