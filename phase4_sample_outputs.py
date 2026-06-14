"""
phase4_sample_outputs.py
------------------------
Generate sample outputs from the v1.6 fine-tuned model.

This is NOT evaluation. These outputs are illustrative
examples for the paper's "Case Studies" section,
demonstrating what the pipeline produces.

For each of 5 example prompts (one per template type,
since extraction is the most distinctive), the script
generates an output using:
  Base Gemma 3 1B IT + v16 LoRA adapter

Run from /home/exouser/jyotsna/terramars/:
    export HF_TOKEN="hf_..."
    python code/phase4_sample_outputs.py

Output:
    data/sample_outputs.json
    logs/sample_outputs.log
"""

import os
import json
import time
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
OUTPUT_FILE = Path("data/sample_outputs.json")
RUN_LOG     = Path("logs/sample_outputs.log")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
RUN_LOG.parent.mkdir(parents=True, exist_ok=True)

# ── Config ───────────────────────────────────────────────────────────────────
BASE_MODEL_ID = "google/gemma-3-1b-it"
HF_TOKEN      = os.environ.get("HF_TOKEN", "")
MAX_NEW_TOK   = 400


def log_msg(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ── Templates (same wording as phase2_generate_v16.py) ──────────────────────
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
    "extraction":      EXTRACTION_INSTRUCTION,
    "qa":              QA_INSTRUCTION,
    "organism":        ORGANISM_INSTRUCTION,
    "stage_reasoning": STAGE_REASONING_INSTRUCTION,
    "intervention":    INTERVENTION_INSTRUCTION,
    "cot_survival":    COT_SURVIVAL_INSTRUCTION,
}


# ── Pick example chunks (deterministic by index) ─────────────────────────────
def pick_example_chunks(chunks, n=5):
    """Pick chunks that look reasonably 'Mars-like' based on simple
    keyword density. This is deterministic for reproducibility."""
    keywords = ["mars", "martian", "regolith", "atmosphere",
                "radiation", "perchlorate", "co2", "habitability",
                "microbial", "water"]

    scored = []
    for i, c in enumerate(chunks):
        text_lc = c.get("text", "").lower()
        score = sum(text_lc.count(k) for k in keywords)
        scored.append((score, i, c))

    scored.sort(key=lambda x: (-x[0], x[1]))  # by score desc, then index
    return [c for _, _, c in scored[:n]]


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    log_msg("=" * 60)
    log_msg("TERRA-MARS v1.6 Sample Output Generation")
    log_msg("=" * 60)

    # Load chunks
    log_msg("Loading chunks...")
    chunks = []
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    log_msg(f"  Loaded {len(chunks)} chunks")

    example_chunks = pick_example_chunks(chunks, n=5)
    log_msg(f"  Selected {len(example_chunks)} top Mars-like chunks")

    # Load model with adapter
    log_msg("\nLoading base model + v16 adapter...")
    bnb_config = BitsAndBytesConfig(
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
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN if HF_TOKEN else None,
    )
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()
    log_msg(f"  Model loaded")

    # Generation function
    def generate(prompt, max_new_tokens=MAX_NEW_TOK):
        full_prompt = (
            f"<start_of_turn>user\n{prompt}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return response.strip()

    # Run one example per template per chunk
    # To keep it short: 5 chunks × 1 template each, rotating
    template_names = list(TEMPLATES.keys())
    samples = []

    log_msg("\nGenerating samples...")
    for i, chunk in enumerate(example_chunks):
        template_name = template_names[i % len(template_names)]
        instruction   = TEMPLATES[template_name].format(chunk=chunk["text"])

        log_msg(f"  [{i+1}/{len(example_chunks)}] template={template_name} "
                f"chunk={chunk.get('chunk_id', '')[:30]}")

        start = time.time()
        response = generate(instruction)
        elapsed  = time.time() - start

        samples.append({
            "example_id":     i + 1,
            "template":       template_name,
            "chunk_id":       chunk.get("chunk_id", ""),
            "paper_id":       chunk.get("paper_id", ""),
            "chunk_text":     chunk.get("text", ""),
            "instruction":    instruction,
            "model_output":   response,
            "generation_time_sec": round(elapsed, 2),
        })

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "model":         "TerraMARS v1.6 (Gemma 3 1B IT + LoRA r=16)",
            "adapter_path":  str(ADAPTER_DIR),
            "generated_at":  datetime.now().isoformat(),
            "n_samples":     len(samples),
            "samples":       samples,
        }, f, indent=2, ensure_ascii=False)

    log_msg("")
    log_msg("=" * 60)
    log_msg("DONE")
    log_msg(f"Samples generated:    {len(samples)}")
    log_msg(f"Output:               {OUTPUT_FILE}")
    log_msg("=" * 60)


if __name__ == "__main__":
    main()
