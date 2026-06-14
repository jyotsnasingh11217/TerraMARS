"""
phase1_scrape.py
----------------
Scrapes Mars terraforming papers from arXiv, Semantic Scholar, PMC.
18 queries. Output goes to data/raw_papers/.

Output:
    data/raw_papers/       individual JSON files
    data/scrape_log.csv    CSV log of what was found
    logs/scrape.log        run log
"""
 
import os, json, time, csv, re, requests
from datetime import datetime
from pathlib import Path
 
OUTPUT_DIR = Path("data/raw_papers")
LOG_FILE   = Path("data/scrape_log.csv")
RUN_LOG    = Path("logs/scrape.log")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
 
SLEEP_BETWEEN = 1.5
 
QUERIES = [
    "Mars terraforming",
    "Mars atmospheric pressure increase",
    "Mars pioneer organisms astrobiology",
    "Mars regolith biotransformation",
    "cyanobacteria Mars survival",
    "Mars greenhouse gas warming",
    "Mars nitrogen fixation",
    "Mars ISRU in-situ resource utilization",
    "Mars climate model habitability",
    "extremophile Mars analog",
    "Mars perchlorate microorganism",
    "Mars water activity microbial",
    "Mars soil formation organic carbon",
    "Deinococcus Mars radiation",
    "Mars oxygen production photosynthesis",
    "Mars permafrost subsurface life",
    "Mars UV radiation microbial survival",
    "Mars sulfate brine habitability",
]
 
papers = {}
 
def log_msg(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
 
def scrape_arxiv(query, max_results=50):
    base = "https://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results,
              "sortBy": "relevance", "sortOrder": "descending"}
    results = []
    try:
        resp = requests.get(base, params=params, timeout=30)
        resp.raise_for_status()
        entries = re.findall(r"<entry>(.*?)</entry>", resp.text, re.DOTALL)
        for entry in entries:
            title = re.search(r"<title>(.*?)</title>", entry, re.DOTALL)
            abstract = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
            arxiv_id = re.search(r"<id>(.*?)</id>", entry)
            authors = re.findall(r"<name>(.*?)</name>", entry)
            published = re.search(r"<published>(.*?)</published>", entry)
            if not (title and abstract and arxiv_id):
                continue
            results.append({
                "title": title.group(1).strip().replace("\n", " "),
                "abstract": abstract.group(1).strip().replace("\n", " "),
                "url": arxiv_id.group(1).strip(),
                "authors": authors,
                "published": published.group(1)[:10] if published else "",
                "source": "arxiv", "license": "open_access", "query": query,
            })
    except Exception as e:
        log_msg(f"  arXiv error for '{query}': {e}")
    return results
 
def scrape_semantic_scholar(query, max_results=50):
    base = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": min(max_results, 100),
              "fields": "title,abstract,authors,year,openAccessPdf,externalIds,publicationDate",
              "openAccessPdf": ""}
    results = []
    try:
        resp = requests.get(base, params=params, timeout=30)
        if resp.status_code == 429:
            log_msg("  SS rate limit, sleeping 10s")
            time.sleep(10)
            return results
        resp.raise_for_status()
        for p in resp.json().get("data", []):
            pdf_info = p.get("openAccessPdf")
            if not pdf_info:
                continue
            paper = {
                "title": p.get("title", "").strip(),
                "abstract": p.get("abstract", "") or "",
                "url": pdf_info.get("url", ""),
                "authors": [a["name"] for a in p.get("authors", [])],
                "published": p.get("publicationDate") or str(p.get("year", "")),
                "source": "semantic_scholar", "license": "open_access", "query": query,
                "doi": p.get("externalIds", {}).get("DOI", ""),
            }
            if paper["title"] and paper["abstract"]:
                results.append(paper)
    except Exception as e:
        log_msg(f"  SS error for '{query}': {e}")
    return results
 
def scrape_pmc(query, max_results=30):
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    sp = {"db": "pmc", "term": f"{query} AND open access[filter]", "retmax": max_results,
          "retmode": "json", "tool": "terra_mars_scraper", "email": "research@example.com"}
    results = []
    try:
        resp = requests.get(search_url, params=sp, timeout=30)
        resp.raise_for_status()
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return results
        time.sleep(0.5)
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        fp = {"db": "pmc", "id": ",".join(ids), "retmode": "json",
              "tool": "terra_mars_scraper", "email": "research@example.com"}
        fresp = requests.get(fetch_url, params=fp, timeout=30)
        fresp.raise_for_status()
        fdata = fresp.json().get("result", {})
        for pmcid in ids:
            doc = fdata.get(pmcid, {})
            title = doc.get("title", "").strip()
            if not title:
                continue
            results.append({
                "title": title, "abstract": "",
                "url": f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/",
                "authors": [a.get("name", "") for a in doc.get("authors", [])],
                "published": doc.get("pubdate", ""),
                "source": "pmc", "license": "open_access", "query": query, "pmcid": pmcid,
            })
    except Exception as e:
        log_msg(f"  PMC error for '{query}': {e}")
    return results
 
def dedupe_key(paper):
    return paper["title"].lower().strip()[:80]
 
def main():
    log_rows = []
    log_msg("=" * 60)
    log_msg(f"TERRA-MARS Scraper v1.5 - {len(QUERIES)} queries")
    log_msg("=" * 60)
 
    for q_idx, query in enumerate(QUERIES):
        log_msg(f"\n[{q_idx+1}/{len(QUERIES)}] '{query}'")
        for source_name, scrape_fn, per_q in [
            ("arXiv", scrape_arxiv, 40),
            ("SemanticScholar", scrape_semantic_scholar, 40),
            ("PMC", scrape_pmc, 20),
        ]:
            found = scrape_fn(query, per_q)
            new_count = 0
            for paper in found:
                key = dedupe_key(paper)
                if key not in papers and paper["title"]:
                    papers[key] = paper
                    new_count += 1
                    safe = re.sub(r"[^a-z0-9]", "_", key)[:60]
                    out_path = OUTPUT_DIR / f"{source_name}_{safe}.json"
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(paper, f, indent=2, ensure_ascii=False)
                    log_rows.append({"query": query, "source": source_name,
                                     "title": paper["title"][:80],
                                     "url": paper.get("url", ""),
                                     "license": paper.get("license", "")})
            log_msg(f"   {source_name:20s} -> {len(found):3d} found, {new_count:3d} new (total: {len(papers)})")
            time.sleep(SLEEP_BETWEEN)
 
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "source", "title", "url", "license"])
        writer.writeheader()
        writer.writerows(log_rows)
 
    log_msg("\n" + "=" * 60)
    log_msg(f"DONE  Total unique papers: {len(papers)}")
    log_msg(f"Output: {OUTPUT_DIR}")
    log_msg("=" * 60)
 
if __name__ == "__main__":
    main()
 
