#!/usr/bin/env python3
"""
COMICS Evaluation Script
========================
Evaluate fine-tuned LLaVA model on comic panel prediction task.

Usage:
    python scripts/evaluate_finetuned.py \
        --project_dir /path/to/project \
        --checkpoint_path checkpoints/llava_comics_lora_pred \
        --num_examples 100

Author: Harsha Sekar
Date: December 2024
"""

import os
import sys
import json
import pickle
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import torch
from PIL import Image
from tqdm import tqdm

from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration
from peft import PeftModel

# Evaluation metrics
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_ID = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
CONTEXT_PANELS = 5
MAX_IMAGE_SIZE = 672
MAX_NEW_TOKENS = 200


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def load_model(checkpoint_path: str, cache_dir: str):
    """Load the fine-tuned model with LoRA adapter."""
    
    print(f"Loading processor from {MODEL_ID}...")
    processor = LlavaOnevisionProcessor.from_pretrained(MODEL_ID, cache_dir=cache_dir)
    
    print(f"Loading base model from {MODEL_ID}...")
    base_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_dir,
        attn_implementation="eager"
    )
    
    print(f"Loading LoRA adapter from {checkpoint_path}...")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    
    return model, processor


def create_prompt(context_texts: List[str]) -> str:
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


def generate_prediction(model, processor, images: List[Image.Image], context_texts: List[str]) -> str:
    """Generate prediction for a single example."""
    
    prompt = create_prompt(context_texts)
    
    # Build conversation
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": prompt})
    
    conversation = [{"role": "user", "content": content}]
    
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    inputs = processor(
        images=images,
        text=text,
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=processor.tokenizer.pad_token_id
        )
    
    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[1]
    generated = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    
    return generated.strip()


def calculate_metrics(predictions: List[str], references: List[str]) -> Dict:
    """Calculate BLEU, ROUGE, and BERTScore metrics."""
    
    results = {
        "bleu_1": [],
        "bleu_2": [],
        "bleu_3": [],
        "bleu_4": [],
        "rouge_1": [],
        "rouge_2": [],
        "rouge_l": [],
    }
    
    smoothie = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    print("\nCalculating metrics...")
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions)):
        # BLEU scores
        pred_tokens = pred.lower().split()
        ref_tokens = [ref.lower().split()]
        
        results["bleu_1"].append(sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie))
        results["bleu_2"].append(sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
        results["bleu_3"].append(sentence_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie))
        results["bleu_4"].append(sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))
        
        # ROUGE scores
        rouge_scores = scorer.score(ref, pred)
        results["rouge_1"].append(rouge_scores['rouge1'].fmeasure)
        results["rouge_2"].append(rouge_scores['rouge2'].fmeasure)
        results["rouge_l"].append(rouge_scores['rougeL'].fmeasure)
    
    # BERTScore
    print("Calculating BERTScore...")
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=True)
    
    # Aggregate
    metrics = {
        "bleu_1": sum(results["bleu_1"]) / len(results["bleu_1"]),
        "bleu_2": sum(results["bleu_2"]) / len(results["bleu_2"]),
        "bleu_3": sum(results["bleu_3"]) / len(results["bleu_3"]),
        "bleu_4": sum(results["bleu_4"]) / len(results["bleu_4"]),
        "rouge_1": sum(results["rouge_1"]) / len(results["rouge_1"]),
        "rouge_2": sum(results["rouge_2"]) / len(results["rouge_2"]),
        "rouge_l": sum(results["rouge_l"]) / len(results["rouge_l"]),
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F1.mean().item(),
    }
    
    return metrics


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main(args):
    """Main evaluation function."""
    
    print("=" * 70)
    print("COMICS MODEL EVALUATION")
    print("=" * 70)
    
    project_dir = Path(args.project_dir)
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = project_dir / checkpoint_path
    
    output_dir = Path(args.output_dir) if args.output_dir else project_dir / "outputs" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Output: {output_dir}")
    print(f"Examples: {args.num_examples}")
    
    # Load model
    model, processor = load_model(
        str(checkpoint_path),
        str(project_dir / "model_cache")
    )
    
    # Load test data
    print("\nLoading test data...")
    test_file = project_dir / "data" / "processed" / "test_sequences.pkl"
    with open(test_file, "rb") as f:
        test_sequences = pickle.load(f)
    
    print(f"Loaded {len(test_sequences)} test sequences")
    
    # Filter valid sequences and sample
    valid_sequences = []
    for seq in test_sequences:
        if len(seq.get("context", [])) >= CONTEXT_PANELS and seq.get("scene_prediction"):
            valid = all(Path(p["image_path"]).exists() for p in seq["context"][-CONTEXT_PANELS:])
            if valid:
                valid_sequences.append(seq)
                if len(valid_sequences) >= args.num_examples:
                    break
    
    print(f"Using {len(valid_sequences)} valid sequences for evaluation")
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = []
    references = []
    results = []
    
    for seq in tqdm(valid_sequences):
        # Load images
        images = []
        for panel in seq["context"][-CONTEXT_PANELS:]:
            img = Image.open(panel["image_path"]).convert("RGB")
            if max(img.size) > MAX_IMAGE_SIZE:
                ratio = MAX_IMAGE_SIZE / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.LANCZOS)
            images.append(img)
        
        context_texts = seq.get("context_texts", [])[-CONTEXT_PANELS:]
        reference = seq.get("scene_prediction", "")
        
        # Generate
        prediction = generate_prediction(model, processor, images, context_texts)
        
        predictions.append(prediction)
        references.append(reference)
        
        results.append({
            "prediction": prediction,
            "reference": reference,
            "context_texts": context_texts
        })
        
        # Clean up
        for img in images:
            img.close()
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, references)
    
    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nBLEU Scores:")
    print(f"  BLEU-1: {metrics['bleu_1']:.4f}")
    print(f"  BLEU-2: {metrics['bleu_2']:.4f}")
    print(f"  BLEU-3: {metrics['bleu_3']:.4f}")
    print(f"  BLEU-4: {metrics['bleu_4']:.4f}")
    print(f"\nROUGE Scores:")
    print(f"  ROUGE-1: {metrics['rouge_1']:.4f}")
    print(f"  ROUGE-2: {metrics['rouge_2']:.4f}")
    print(f"  ROUGE-L: {metrics['rouge_l']:.4f}")
    print(f"\nBERTScore:")
    print(f"  Precision: {metrics['bertscore_precision']:.4f}")
    print(f"  Recall: {metrics['bertscore_recall']:.4f}")
    print(f"  F1: {metrics['bertscore_f1']:.4f}")
    print("=" * 70)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    metrics_file = output_dir / f"metrics_{timestamp}.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")
    
    results_file = output_dir / f"predictions_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Predictions saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LLaVA model")
    parser.add_argument("--project_dir", type=str, required=True, help="Project directory")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of test examples")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    
    args = parser.parse_args()
    main(args)
