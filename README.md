# TerraMARS

A domain-adapted small-language-model pipeline for Mars terraforming literature. TerraMARS fine-tunes Gemma 3 1B IT on a corpus of 614 open-access papers to produce both free-form question answering and structured (JSON) quantitative extraction from Mars science abstracts.

This repository contains the complete pipeline: scraping, chunking, synthetic instruction-tuning data generation, QLoRA fine-tuning, and illustrative output generation.

## Pipeline Overview

```
Phase 1: Scraping              → 614 papers
   ↓
Phase 2: Synthetic data gen    → 1,179 training examples
   ↓
Phase 3: QLoRA fine-tuning     → 13M-param LoRA adapter
   ↓
Phase 4: Illustrative outputs  → QA + JSON samples
```


## Scripts

### `phase1_scrape.py`
Scrapes 614 Mars terraforming papers from arXiv, PubMed Central, and Semantic Scholar using 18 search queries that combine the term "Mars" with domain-specific terms.

### `phase2_generate.py`
Generates synthetic instruction-tuning data using Llama 3.2 3B (via Ollama) as the teacher. Six task templates: constraint extraction, QA, organism identification, stage reasoning, intervention, and chain-of-thought survival. Strict schema check enforces five required fields on extraction outputs.

Produces 1,179 valid examples from 1,182 attempts (99.7% pass rate) in approximately four hours.

### `phase3_finetune.py`
QLoRA fine-tuning of `google/gemma-3-1b-it` with 4-bit NF4 quantization. LoRA rank 16, alpha 32, applied to seven projection matrices.

### `phase4_sample_outputs.py`
Generates illustrative outputs from the fine-tuned model for the five top Mars-keyword-density chunks, one template per chunk. Used to produce the QA example in the paper.

### `extraction.py`
Tests the extraction template on the ten most numeric-rich chunks with greedy decoding (temperature 0). Used to produce the JSON example in the paper.

## Requirements

- Python 3.10+
- PyTorch with CUDA support
- transformers, peft, trl, bitsandbytes, datasets
- Ollama (for Phase 2 teacher model)
- NVIDIA GPU with at least 10 GB memory (for Phase 3 and Phase 4)
- HuggingFace token with access to `google/gemma-3-1b-it`

## Quick Start

```bash
# Phase 1: scrape
python code/phase1_scrape.py

# Phase 2: generate synthetic data (requires Ollama with llama3.2:3b)
ollama pull llama3.2:3b
python code/phase2_generate_v16.py

# Phase 3: fine-tune (requires HuggingFace token)
export HF_TOKEN="hf_..."
python code/phase3_finetune_v16.py

# Phase 4: generate illustrative outputs
python code/phase4_sample_outputs.py
python code/retry_extraction_v3.py
```

## Reproducibility

All experiments use random seed 42. Deterministic greedy decoding (temperature 0) is used for output generation. Hyperparameters and configuration are recorded inline in each script.

## Citation

If you use TerraMARS, please cite:

```bibtex
@article{singh2026terramars,
  title  = {TerraMARS: A Domain-Adapted Small-Language-Model Pipeline for Mars Terraforming Literature},
  author = {Singh, Jyotsna and Hu, Xiao and Black, Ash and Larsen, Jeff and Saleska, Scott},
  year   = {2026}
}
```


## Acknowledgements
University of Arizona, Tucson, Arizona, USA

This work used the Jetstream2 system at Indiana University through ACCESS allocation [ACCESS-XXXXX].

## License

Apache 2.0