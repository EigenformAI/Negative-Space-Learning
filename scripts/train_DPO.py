import sys
import os

# Set memory optimization environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

# For Tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not available. Experiment tracking will be disabled.")

logger = logging.getLogger(__name__)

PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

def clear_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def load_dpo_dataset(file_path: str, max_examples: int = None) -> Dataset:
    """Load DPO dataset from JSONL file with optional limit."""
    logger.info(f"Loading DPO dataset from {file_path}")
    
    data = {"prompt": [], "chosen": [], "rejected": []}
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if max_examples and count >= max_examples:
                break
                
            try:
                item = json.loads(line.strip())
                if 'instruction' in item and 'chosen' in item and 'rejected' in item:
                    # Apply the same prompt template used in SFT
                    prompt_formatted = PROMPT.format(instruction=item['instruction'])
                    
                    data["prompt"].append(prompt_formatted)
                    data["chosen"].append(item['chosen'])
                    data["rejected"].append(item['rejected'])
                    count += 1
                else:
                    logger.warning(f"Line {line_num}: Missing required fields, skipping")
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: JSON decode error - {e}")
                continue
    
    logger.info(f"Loaded {count} DPO examples from {file_path}")
    dataset = Dataset.from_dict(data)
    return dataset

def build_model(model_name, use_quantization=True, lora_rank=32):
    clear_memory()
    compute_dtype = torch.float16
    
    # Handle quantization configuration
    quantization_config = None
    if use_quantization:
        logger.info("Loading model with 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    
    # Map the model to the correct GPU based on local rank for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device_map = {'': local_rank}
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map=device_map,
    )
    
    # setattr(model, 'model_parallel', True)
    # setattr(model, 'is_parallelizable', True)
    
    # Prepare for k-bit training if quantized
    if use_quantization:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    # Add LoRA adapters
    logger.info(f'Init LoRA modules with rank {lora_rank}...')
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        inference_mode=False,
        r=lora_rank, 
        lora_alpha=lora_rank,
        lora_dropout=0.0,
        init_lora_weights=True,
    )
    model = get_peft_model(model, peft_config)

    # Ensure norm and gate layers are in FP32
    for name, module in model.named_modules():
        if 'norm' in name or 'gate' in name:
            module.to(torch.float32)
    
    clear_memory()
    return model

def main():
    # Configuration
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Base model
    DATA_PATH = "/Your-dataset-path/dpo_dataset.jsonl"  # Path to your DPO dataset
    OUTPUT_DIR = "/Output-path/qwen25-7b-lora-dpo"  # Where to save the LoRA adapter and training logs
    
    # DPO specific hyperparameters
    BETA = 0.1  # The beta factor in DPO loss. Higher means less deviation from reference model. Default: 0.1
    # NOTE: DPO learning rates should be MUCH SMALLER than SFT (e.g., 5e-6 vs 2e-4)
    LEARNING_RATE = 5e-6 
    # ----------------------------------------------------
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting simple Qwen 2.5 DPO training...")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Data: {DATA_PATH}")
    
    if WANDB_AVAILABLE:
        wandb.init(
            project="qwen-simple-dpo",
            name="qwen25-7b-lora-4bit-dpo",
            config={
                "model": MODEL_NAME,
                "lora_rank": 32,
                "learning_rate": LEARNING_RATE,
                "beta": BETA,
                "batch_size": 1,
                "gradient_accumulation_steps": 32,
            }
        )

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_NAME,
        model_max_length=1024,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded")
    
    # Load dataset
    train_dataset = load_dpo_dataset(DATA_PATH, max_examples=15000)
    train_dataset = train_dataset.shuffle(seed=42)
    logger.info(f"Dataset loaded: {len(train_dataset)} examples")
    
    # Build model
    model = build_model(MODEL_NAME, use_quantization=True, lora_rank=32)
    logger.info("Model loaded")

    # Create DPO training arguments
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        learning_rate=LEARNING_RATE,
        weight_decay=0.0,
        warmup_ratio=0.03,
        logging_steps=5,
        lr_scheduler_type="cosine",
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        dataloader_num_workers=1,
        remove_unused_columns=False,
        report_to="wandb" if WANDB_AVAILABLE else "none",
        # DPO specific settings
        beta=BETA,
        max_prompt_length=512,
        max_length=1024,
    )

    # Initialize DPO Trainer
    # When using PEFT (LoRA), we don't need to pass a separate reference_model.
    # DPOTrainer automatically treats the base model (without adapters) as the reference,
    # and trains the adapter weights to maximize the preference reward.
    trainer = DPOTrainer(
        model=model,
        ref_model=None, # None = use the base PEFT model without adapters as reference
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    clear_memory()
    
    # Start training
    logger.info("Starting DPO training...")
    trainer.train()
    trainer.save_state()
    
    # Save the LoRA adapter
    logger.info("Saving LoRA adapter...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    logger.info("Training completed!")
    logger.info(f"LoRA Adapter saved to: {OUTPUT_DIR}")
    logger.info("Next: Use your 'merge_LoRA.py' script to merge this adapter cleanly into a 16-bit base model.")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
