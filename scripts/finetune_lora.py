#!/usr/bin/env python3
"""
COMICS Fine-Tuning Script - LLaVA with LoRA
============================================
Fine-tunes LLaVA-OneVision on comic panel prediction task using LoRA.

Usage:
    python scripts/finetune_lora.py --project_dir /path/to/project --training_size medium

Author: Harsha Sekar
Date: December 2024
"""

import os
import sys
import json
import pickle
import random
import gc
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    LlavaOnevisionProcessor,
    LlavaOnevisionForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# =============================================================================
# CONFIGURATION
# =============================================================================

TRAINING_CONFIGS = {
    "quick": {
        "num_train_examples": 1000,
        "num_val_examples": 200,
        "num_epochs": 1,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "description": "Quick test run (~30 min)"
    },
    "medium": {
        "num_train_examples": 15000,
        "num_val_examples": 1000,
        "num_epochs": 1,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "description": "Medium run (~2-3 hours)"
    },
    "full": {
        "num_train_examples": -1,
        "num_val_examples": 5000,
        "num_epochs": 3,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "description": "Full training (8+ hours)"
    }
}

MODEL_ID = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
CONTEXT_PANELS = 5
MAX_IMAGE_SIZE = 672
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-5
RANDOM_SEED = 42


# =============================================================================
# DATASET CLASS
# =============================================================================

class ComicsDataset(Dataset):
    """Dataset for comic panel prediction task."""
    
    def __init__(self, sequences: List[Dict], processor, max_samples: int = -1):
        self.processor = processor
        
        self.sequences = []
        for seq in sequences:
            if self._is_valid(seq):
                self.sequences.append(seq)
                if max_samples > 0 and len(self.sequences) >= max_samples:
                    break
        
        print(f"Dataset: {len(self.sequences)} valid sequences")
    
    def _is_valid(self, seq: Dict) -> bool:
        """Check if sequence is valid for training."""
        if len(seq.get("context", [])) < CONTEXT_PANELS:
            return False
        if not seq.get("scene_prediction"):
            return False
        for panel in seq["context"][-CONTEXT_PANELS:]:
            if Path(panel["image_path"]).exists():
                return True
        return False
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx: int):
        try:
            seq = self.sequences[idx]
            
            # Load and resize images
            images = []
            context_panels = seq["context"][-CONTEXT_PANELS:]
            
            for panel in context_panels:
                img_path = Path(panel["image_path"])
                if img_path.exists():
                    img = Image.open(img_path).convert("RGB")
                    if max(img.size) > MAX_IMAGE_SIZE:
                        ratio = MAX_IMAGE_SIZE / max(img.size)
                        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                        img = img.resize(new_size, Image.LANCZOS)
                    images.append(img)
                else:
                    images.append(Image.new("RGB", (336, 336), color="gray"))
            
            # Get context texts
            context_texts = seq.get("context_texts", [])[-CONTEXT_PANELS:]
            while len(context_texts) < CONTEXT_PANELS:
                context_texts.insert(0, "")
            
            # Create prompt and target
            prompt = self._create_prompt(context_texts)
            target_text = seq.get("scene_prediction", "")
            
            # Build conversation format
            content = [{"type": "image"} for _ in images]
            content.append({"type": "text", "text": prompt})
            
            conversation = [
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": target_text}]}
            ]
            
            text = self.processor.apply_chat_template(conversation, tokenize=False)
            
            inputs = self.processor(
                images=images,
                text=text,
                return_tensors="pt",
                padding=False,
            )
            
            result = {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "labels": inputs["input_ids"].squeeze(0).clone(),
            }
            
            if "pixel_values" in inputs:
                result["pixel_values"] = inputs["pixel_values"]
            if "image_sizes" in inputs:
                result["image_sizes"] = inputs["image_sizes"]
            
            del images
            return result
            
        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            return None
    
    def _create_prompt(self, context_texts: List[str]) -> str:
        """Create the prediction prompt."""
        prompt = """You are looking at 5 consecutive panels from a comic book.

Here is the text from each panel:
"""
        for i, text in enumerate(context_texts, 1):
            if text and text.strip():
                prompt += f"Panel {i}: {text.strip()[:400]}\n"
            else:
                prompt += f"Panel {i}: [No text]\n"
        
        prompt += """\nBased on what you see in these 5 panels, describe what happens in the NEXT panel (Panel 6).

Include the scene, any dialogue, and sound effects."""
        return prompt


# =============================================================================
# DATA COLLATOR
# =============================================================================

class ComicsDataCollator:
    """Collate function for comics dataset."""
    
    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = processor.tokenizer.pad_token_id or 0
    
    def __call__(self, examples):
        examples = [ex for ex in examples if ex is not None]
        
        if len(examples) == 0:
            raise ValueError("All examples in batch were None")
        
        # Use first example (batch size 1 per GPU for memory)
        ex = examples[0]
        
        batch = {
            "input_ids": ex["input_ids"].unsqueeze(0),
            "attention_mask": ex["attention_mask"].unsqueeze(0),
            "labels": ex["labels"].unsqueeze(0),
        }
        
        if "pixel_values" in ex and ex["pixel_values"] is not None:
            batch["pixel_values"] = ex["pixel_values"]
        
        if "image_sizes" in ex and ex["image_sizes"] is not None:
            batch["image_sizes"] = ex["image_sizes"]
        
        return batch


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main(args):
    """Main training function."""
    
    print("=" * 70)
    print("COMICS FINE-TUNING WITH LoRA")
    print("=" * 70)
    
    # Configuration
    config = TRAINING_CONFIGS[args.training_size]
    project_dir = Path(args.project_dir)
    data_dir = project_dir / "data" / "processed"
    output_dir = project_dir / "checkpoints" / f"llava_comics_lora_{args.task_type}"
    
    print(f"\nConfiguration: {args.training_size}")
    print(f"  {config['description']}")
    print(f"  Training examples: {config['num_train_examples']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Gradient accumulation: {config['gradient_accumulation_steps']}")
    print(f"  Effective batch: {config['batch_size'] * config['gradient_accumulation_steps']}")
    
    # Check GPU
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU available!")
        return
    
    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Set seeds
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # Load processor
    print("\n[1/5] Loading processor...")
    processor = LlavaOnevisionProcessor.from_pretrained(
        MODEL_ID,
        cache_dir=str(project_dir / "model_cache")
    )
    
    # Load model
    print("[2/5] Loading model...")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=str(project_dir / "model_cache"),
        attn_implementation="eager"
    )
    
    # Configure LoRA
    print("[3/5] Configuring LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load data
    print("[4/5] Loading data...")
    
    train_file = data_dir / "train_sequences.pkl"
    val_file = data_dir / "val_sequences.pkl"
    
    with open(train_file, "rb") as f:
        train_sequences = pickle.load(f)
    with open(val_file, "rb") as f:
        val_sequences = pickle.load(f)
    
    print(f"  Loaded {len(train_sequences)} train, {len(val_sequences)} val sequences")
    
    # Create datasets
    train_dataset = ComicsDataset(
        train_sequences, 
        processor, 
        max_samples=config["num_train_examples"]
    )
    val_dataset = ComicsDataset(
        val_sequences,
        processor,
        max_samples=config["num_val_examples"]
    )
    
    # Training arguments
    print("[5/5] Setting up training...")
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
        evaluation_strategy="steps",
        save_total_limit=3,
        fp16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=ComicsDataCollator(processor),
    )
    
    # Train
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")
    
    start_time = datetime.now()
    trainer.train()
    
    # Save
    print("\nSaving model...")
    trainer.save_model()
    processor.save_pretrained(output_dir)
    
    # Summary
    elapsed = datetime.now() - start_time
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Time: {elapsed}")
    print(f"  Model saved to: {output_dir}")
    print("=" * 70)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLaVA on COMICS dataset")
    parser.add_argument("--project_dir", type=str, required=True, help="Project directory")
    parser.add_argument("--training_size", type=str, default="medium", 
                        choices=["quick", "medium", "full"], help="Training configuration")
    parser.add_argument("--task_type", type=str, default="prediction",
                        choices=["prediction", "description"], help="Task type")
    
    args = parser.parse_args()
    main(args)
