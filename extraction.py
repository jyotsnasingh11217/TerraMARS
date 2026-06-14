"""
retry_extraction_v3.py
----------------------
Test the v1.6 fine-tuned model's extraction behavior across
10 different numeric-heavy chunks using greedy decoding
(temperature=0).

Goal: see whether ANY chunks produce complete, schema-valid
JSON output. This is honest characterization, not cherry-picking.

Fixes from v2:
  - Parser already strips markdown ```json``` fences
  - MAX_NEW_TOK = 1500 (avoids truncation)
  
New in v3:
  - N_CANDIDATES = 10 (broader sample)
  - Reports per-chunk: valid_json, schema_complete,
    n_constraints, output_truncated_mid_json

Run from /home/exouser/jyotsna/terramars/:
    export HF_TOKEN="hf_..."
    python code/retry_extraction_v3.py
"""

import os
import re
import json
import torch
from datetime import datetime
from pathlib import Path

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ── Paths ────────────────────────────────────────────────────────────────────
ADAPTER_DIR = Path("output/v16_final_adapter")
CHUNKS_FILE = Path("data/chunks/all_chunks.jsonl")
OUTPUT_FILE = Path("data/sample_extraction_retries_v3.json")

# ── Config ───────────────────────────────────────────────────────────────────
BASE_MODEL_ID = "google/gemma-3-1b-it"
HF_TOKEN      = os.environ.get("HF_TOKEN", "")
MAX_NEW_TOK   = 1500
N_CANDIDATES  = 10


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


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def count_numeric_tokens(text):
    pattern = r"\d+\.?\d*\s*(?:°C|K|mbar|kPa|Pa|nm|cm|mm|m|km|"
    pattern += r"%|wt%|Gy|mGy|W/m|kJ|MJ|day|year|yr|hour|hr)"
    return len(re.findall(pattern, text))


def pick_numeric_chunks(chunks, n=N_CANDIDATES):
    scored = []
    for i, c in enumerate(chunks):
        text  = c.get("text", "")
        score = count_numeric_tokens(text)
        scored.append((score, i, c))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [(c, score) for score, _, c in scored[:n]]


def strip_markdown_fences(text):
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    return cleaned.strip()


def extract_first_json_object(text):
    cleaned = strip_markdown_fences(text)
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def main():
    log("=" * 60)
    log(f"v1.6 Extraction Test V3 — {N_CANDIDATES} numeric-heavy chunks")
    log(f"Decoding: greedy (temperature=0)")
    log(f"MAX_NEW_TOK={MAX_NEW_TOK}")
    log("=" * 60)

    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    log(f"Loaded {len(chunks)} chunks")

    candidates = pick_numeric_chunks(chunks, n=N_CANDIDATES)
    log(f"Selected {N_CANDIDATES} numeric-heavy chunks:")
    for i, (c, score) in enumerate(candidates, 1):
        title = c.get("text", "")[:70].replace("\n", " ")
        log(f"  [{i}] score={score:2d} {title}...")

    log("\nLoading model + adapter...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, token=HF_TOKEN if HF_TOKEN else None
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb,
        device_map="auto",
        token=HF_TOKEN if HF_TOKEN else None,
    )
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()
    log("Model loaded")

    def generate(prompt):
        full_prompt = (
            f"<start_of_turn>user\n{prompt}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOK,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

    log("\nGenerating extraction outputs...")
    attempts = []
    for i, (chunk, score) in enumerate(candidates, 1):
        log(f"  [{i}/{N_CANDIDATES}] running...")
        prompt   = EXTRACTION_INSTRUCTION.format(chunk=chunk["text"])
        response = generate(prompt)
        parsed   = extract_first_json_object(response)
        valid    = (
            parsed is not None
            and isinstance(parsed, dict)
            and "constraints" in parsed
        )

        # Detect mid-JSON truncation
        cleaned = strip_markdown_fences(response)
        open_braces  = cleaned.count("{")
        close_braces = cleaned.count("}")
        truncated_mid_json = open_braces > close_braces

        # Schema completeness
        schema_complete = False
        if valid:
            schema_complete = all(
                isinstance(c, dict)
                and {"parameter", "value", "unit", "condition",
                     "terraforming_stage"}.issubset(set(c.keys()))
                for c in parsed["constraints"]
            )

        n_constraints = len(parsed["constraints"]) if valid else 0
        log(
            f"      length={len(response):4d} "
            f"valid={valid} "
            f"schema={schema_complete} "
            f"n_constraints={n_constraints} "
            f"truncated_mid_json={truncated_mid_json}"
        )

        attempts.append({
            "attempt":             i,
            "chunk_id":            chunk.get("chunk_id", ""),
            "paper_id":            chunk.get("paper_id", ""),
            "numeric_score":       score,
            "chunk_text":          chunk.get("text", ""),
            "model_output":        response,
            "output_length":       len(response),
            "valid_json":          valid,
            "schema_complete":     schema_complete,
            "n_constraints":       n_constraints,
            "truncated_mid_json":  truncated_mid_json,
            "parsed_json":         parsed,
        })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "model":         "TerraMARS v1.6",
            "generated_at":  datetime.now().isoformat(),
            "decoding":      "greedy (temperature=0)",
            "max_new_tok":   MAX_NEW_TOK,
            "n_attempts":    len(attempts),
            "attempts":      attempts,
        }, f, indent=2, ensure_ascii=False)

    valid_count     = sum(1 for a in attempts if a["valid_json"])
    schema_count    = sum(1 for a in attempts if a["schema_complete"])
    truncated_count = sum(1 for a in attempts if a["truncated_mid_json"])
    empty_count     = sum(1 for a in attempts if a["output_length"] == 0)

    log("")
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"Total attempts:           {N_CANDIDATES}")
    log(f"Empty outputs:            {empty_count}")
    log(f"Valid JSON:               {valid_count}")
    log(f"Schema-complete:          {schema_count}")
    log(f"Truncated mid-JSON:       {truncated_count}")
    log("")
    log(f"Output: {OUTPUT_FILE}")
    log("=" * 60)


if __name__ == "__main__":
    main()
