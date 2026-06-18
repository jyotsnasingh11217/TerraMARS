"""
phase2_generate_v16.py
----------------------
 Synthetic instruction tuning data generation.
 teacher (llama3.2:3b via Ollama),
  same input chunks, same Alpaca output format.

Output:
    data/training/v16_training_data.jsonl
    logs/v16_generation.log
"""

import json
import random
import time
import requests
from datetime import datetime
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
CHUNKS_FILE = Path("data/chunks/all_chunks.jsonl")
OUTPUT_FILE = Path("data/training/v16_training_data.jsonl")
LOG_FILE = Path("logs/v16_generation.log")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
TEACHER_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"
EXAMPLES_PER_CHUNK = 3
TEMPERATURE = 0.4
TOP_P = 0.9
MAX_RESPONSE_TOKENS = 800
TIMEOUT_SECONDS = 120
MIN_RESPONSE_CHARS = 20

# Required fields in extraction-template constraints.
EXTRACTION_REQUIRED_FIELDS = [
    "parameter",
    "value",
    "unit",
    "condition",
    "terraforming_stage",
]

random.seed(42)

# ── Six templates (same as v1.5, schema tightened on extraction) ─────────────
# IMPORTANT: NO scientific values are hard-coded in any template.
# All numerical content comes from the chunk text only.

EXTRACTION_INSTRUCTION = """Extract quantitative constraints from the TEXT below.

STRICT RULES:
1. Use ONLY numbers, units, and values that appear in the TEXT.
2. Do NOT use any external knowledge or memorized values.
3. Do NOT compute, convert, or derive values that are not in the TEXT.
4. Every constraint MUST include ALL 5 fields shown below.
5. If "terraforming_stage" cannot be determined from TEXT, use 0.
6. If "condition" cannot be determined from TEXT, use "not specified".

TEXT:
{chunk}

Respond with ONLY a JSON object in this EXACT format. Do not omit any field:
{{
  "constraints": [
    {{
      "parameter": "parameter_name",
      "value": 0.0,
      "unit": "unit_here",
      "condition": "condition_here_or_not_specified",
      "terraforming_stage": 0
    }}
  ]
}}

If TEXT contains no quantitative constraints, return: {{"constraints": []}}"""


QA_INSTRUCTION = """Answer a question about Mars terraforming based ONLY on the TEXT below.

STRICT RULES:
1. Use ONLY information explicitly present in TEXT.
2. Do NOT add facts, numbers, or values from your training memory.
3. If TEXT does not contain enough information, say so.

TEXT:
{chunk}

Generate one question that can be answered from TEXT, then answer it.

Format your response as:
Q: <your question>
A: <your answer, using only TEXT>"""


ORGANISM_INSTRUCTION = """Based ONLY on the TEXT below, describe what organism, microbial property, or
biological adaptation is being discussed, and what conditions affect it.

STRICT RULES:
1. Use ONLY content from TEXT.
2. Do NOT invent organism names or properties not in TEXT.
3. Do NOT add numerical values that are not in TEXT.

TEXT:
{chunk}

Write a focused 2-4 sentence summary using only what TEXT says."""


STAGE_REASONING_INSTRUCTION = """Based ONLY on the TEXT below, identify which terraforming stage or phase
this information applies to (e.g., early stage, microbial pioneer phase, plant
introduction, etc.), and explain why using TEXT.

STRICT RULES:
1. Use ONLY information present in TEXT.
2. Do NOT speculate beyond what TEXT supports.
3. Do NOT introduce numerical values that are not in TEXT.

TEXT:
{chunk}

Provide a brief reasoning chain (3-5 sentences)."""


INTERVENTION_INSTRUCTION = """Based ONLY on the TEXT below, describe one engineering or biological
intervention mentioned, and what it accomplishes.

STRICT RULES:
1. Use ONLY content from TEXT.
2. Do NOT invent interventions not in TEXT.
3. Do NOT add numerical values that are not in TEXT.

TEXT:
{chunk}

Describe the intervention in 2-4 sentences."""


COT_SURVIVAL_INSTRUCTION = """Based ONLY on the TEXT below, build a short chain-of-thought reasoning
about microbial or biological survival on Mars.

STRICT RULES:
1. Use ONLY facts present in TEXT.
2. Each reasoning step must reference content from TEXT.
3. Do NOT add numerical values that are not in TEXT.

TEXT:
{chunk}

Format:
Step 1: ...
Step 2: ...
Step 3: ...
Conclusion: ..."""


TEMPLATES = {
    "extraction": EXTRACTION_INSTRUCTION,
    "qa": QA_INSTRUCTION,
    "organism": ORGANISM_INSTRUCTION,
    "stage_reasoning": STAGE_REASONING_INSTRUCTION,
    "intervention": INTERVENTION_INSTRUCTION,
    "cot_survival": COT_SURVIVAL_INSTRUCTION,
}

TEMPLATE_NAMES = list(TEMPLATES.keys())


# ── Helpers ──────────────────────────────────────────────────────────────────
def log_msg(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_chunks(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_existing_chunk_ids(path):
    """For resume: which chunks have already been processed."""
    seen = set()
    if not path.exists():
        return seen
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    d = json.loads(line)
                    if d.get("source_chunk_id"):
                        seen.add(d["source_chunk_id"])
                except json.JSONDecodeError:
                    continue
    return seen


def call_teacher(prompt):
    """Call Llama 3.2 3B via Ollama. Return generated text or None on failure."""
    payload = {
        "model": TEACHER_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "num_predict": MAX_RESPONSE_TOKENS,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        log_msg(f"  Teacher call failed: {e}")
        return None


def extract_first_json_object(text):
    """Find first balanced {...} block in text. Return parsed dict or None."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def passes_sanity_checks(response, template_name):
    """Sanity checks. Returns (passed: bool, reason: str)."""
    if response is None:
        return False, "no_response"
    if len(response.strip()) < MIN_RESPONSE_CHARS:
        return False, "too_short"

    if template_name == "extraction":
        parsed = extract_first_json_object(response)
        if parsed is None:
            return False, "invalid_json"
        if "constraints" not in parsed:
            return False, "no_constraints_key"
        constraints = parsed["constraints"]
        if not isinstance(constraints, list):
            return False, "constraints_not_list"
        # Empty constraints list is allowed (means: nothing extractable)
        if len(constraints) == 0:
            return True, "ok_empty_constraints"
        # Every constraint must have all 5 required fields
        for c in constraints:
            if not isinstance(c, dict):
                return False, "constraint_not_dict"
            for field in EXTRACTION_REQUIRED_FIELDS:
                if field not in c:
                    return False, f"missing_field_{field}"
        return True, "ok"

    # Non-extraction templates: only length check applies.
    return True, "ok"


def build_alpaca_record(chunk, template_name, response):
    """Wrap teacher output in the Alpaca-style record used by v1.5."""
    template_text = TEMPLATES[template_name]
    instruction = template_text.format(chunk=chunk["text"])
    return {
        "instruction": instruction,
        "input": "",
        "output": response,
        "type": template_name,
        "source_chunk_id": chunk.get("chunk_id", ""),
        "source_paper": chunk.get("paper_id", ""),
        "source_text": chunk.get("text", "")[:500],
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    log_msg("=" * 60)
    log_msg("TERRA-MARS v1.6 Synthetic Data Generation (tightened)")
    log_msg(f"Teacher: {TEACHER_MODEL}")
    log_msg(f"Input:   {CHUNKS_FILE}")
    log_msg(f"Output:  {OUTPUT_FILE}")
    log_msg("=" * 60)

    if not CHUNKS_FILE.exists():
        log_msg(f"ERROR: {CHUNKS_FILE} not found.")
        return

    chunks = load_chunks(CHUNKS_FILE)
    log_msg(f"Loaded {len(chunks)} chunks")

    seen_ids = load_existing_chunk_ids(OUTPUT_FILE)
    if seen_ids:
        log_msg(f"Resume mode: {len(seen_ids)} chunks already processed")
    todo = [c for c in chunks if c.get("chunk_id", "") not in seen_ids]
    log_msg(f"Remaining to process: {len(todo)}")

    total_attempted = 0
    total_passed = 0
    total_failed = 0
    fail_reasons = {}

    start = time.time()
    out_fh = open(OUTPUT_FILE, "a", encoding="utf-8")

    try:
        for idx, chunk in enumerate(todo):
            picked = random.sample(TEMPLATE_NAMES, k=EXAMPLES_PER_CHUNK)

            for template_name in picked:
                total_attempted += 1
                prompt = TEMPLATES[template_name].format(chunk=chunk["text"])
                response = call_teacher(prompt)

                ok, reason = passes_sanity_checks(response, template_name)
                if not ok:
                    total_failed += 1
                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
                    continue

                record = build_alpaca_record(chunk, template_name, response)
                out_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_fh.flush()
                total_passed += 1

            if (idx + 1) % 25 == 0:
                elapsed = (time.time() - start) / 60
                rate = (idx + 1) / max(elapsed, 0.01)
                remain = (len(todo) - idx - 1) / max(rate, 0.01)
                log_msg(
                    f"  [{idx + 1}/{len(todo)}] "
                    f"passed={total_passed} failed={total_failed} "
                    f"elapsed={elapsed:.1f}min eta={remain:.1f}min"
                )

    finally:
        out_fh.close()

    elapsed_min = (time.time() - start) / 60
    log_msg("")
    log_msg("=" * 60)
    log_msg("DONE")
    log_msg(f"Chunks processed:     {len(todo)}")
    log_msg(f"Examples attempted:   {total_attempted}")
    log_msg(f"Examples passed:      {total_passed}")
    log_msg(f"Examples failed:      {total_failed}")
    if fail_reasons:
        log_msg(f"Failure reasons:")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            log_msg(f"    {reason}: {count}")
    log_msg(f"Total elapsed:        {elapsed_min:.1f} minutes")
    log_msg(f"Output:               {OUTPUT_FILE}")
    log_msg("=" * 60)


if __name__ == "__main__":
    main()
