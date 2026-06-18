"""
phase3_finetune.py
----------------------
QLoRA fine-tuning of Gemma 3 1B IT on clean training data.

Output:
    output/v16_final_adapter/         trained QLoRA adapter
    output/v16_checkpoints/           training checkpoints
    logs/finetune_v16.log             run log
"""

import os
import json
import time
import torch
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTConfig, SFTTrainer

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_FILE = Path("data/training/v16_training_data.jsonl")
OUTPUT_DIR = Path("output/v16_checkpoints")
ADAPTER_DIR = Path("output/v16_final_adapter")
RUN_LOG = Path("logs/finetune_v16.log")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
RUN_LOG.parent.mkdir(parents=True, exist_ok=True)

# ── Config (same as v1.5) ────────────────────────────────────────────────────
MODEL_ID = "google/gemma-3-1b-it"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

MAX_SEQ_LEN = 256
BATCH_SIZE = 2
GRAD_ACCUM = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2
WARMUP_RATIO = 0.05
SAVE_STEPS = 50


def log_msg(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(RUN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_alpaca(example):
    """Convert Alpaca format to Gemma chat template."""
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")
    if inp:
        user_msg = f"{instruction}\n\nInput: {inp}"
    else:
        user_msg = instruction
    text = (
        f"<start_of_turn>user\n{user_msg}<end_of_turn>\n"
        f"<start_of_turn>model\n{output}<end_of_turn>"
    )
    return {"text": text}


def main():
    log_msg("=" * 60)
    log_msg("TERRA-MARS v1.6 Fine-Tuning")
    log_msg(f"Base model:  {MODEL_ID}")
    log_msg(f"Data file:   {DATA_FILE}")
    log_msg(f"Output:      {ADAPTER_DIR}")
    log_msg("=" * 60)

    if not torch.cuda.is_available():
        log_msg("ERROR: No GPU detected. QLoRA needs CUDA.")
        return
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    log_msg(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    log_msg("\nLoading dataset...")
    if not DATA_FILE.exists():
        log_msg(f"ERROR: {DATA_FILE} not found. Run phase2_generate_v16.py first.")
        return
    raw_data = load_jsonl(DATA_FILE)
    log_msg(f"  Loaded {len(raw_data)} examples")

    formatted = [format_alpaca(ex) for ex in raw_data]
    hf_dataset = Dataset.from_list(formatted)

    splits = hf_dataset.train_test_split(test_size=0.05, seed=42)
    train_data = splits["train"]
    val_data = splits["test"]
    log_msg(f"  Train: {len(train_data)}  |  Val: {len(val_data)}")

    log_msg(f"\nLoading {MODEL_ID} with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN if HF_TOKEN else None,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_TOKEN if HF_TOKEN else None,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    log_msg(f"  Model loaded. Parameters: {model.num_parameters() / 1e9:.2f}B")

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log_msg(
        f"\nTrainable parameters: {trainable:,} ({100 * trainable / total:.2f}% of {total:,})"
    )

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        dataloader_num_workers=2,
        report_to="none",
        run_name="terra-mars-v16",
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
    )

    steps_per_epoch = max(len(train_data) // (BATCH_SIZE * GRAD_ACCUM), 1)
    total_steps = steps_per_epoch * NUM_EPOCHS
    log_msg(f"\nStarting training...")
    log_msg(f"  Steps per epoch: {steps_per_epoch}")
    log_msg(f"  Total steps:     {total_steps}")
    log_msg(f"  Effective batch: {BATCH_SIZE * GRAD_ACCUM}")

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time
    log_msg(f"\nTraining done in {elapsed / 60:.1f} minutes")

    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    log_msg(f"\nAdapter saved to: {ADAPTER_DIR}")

    log_msg("\n" + "=" * 60)
    log_msg("DONE")
    log_msg(f"Training time:       {elapsed / 60:.1f} minutes")
    log_msg(f"Trainable params:    {trainable:,}")
    log_msg(f"Training examples:   {len(train_data)}")
    log_msg(f"Validation examples: {len(val_data)}")
    log_msg(f"Final adapter:       {ADAPTER_DIR}")
    log_msg("=" * 60)


if __name__ == "__main__":
    main()
