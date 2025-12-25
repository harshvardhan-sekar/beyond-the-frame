import argparse
import json
import math
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from packaging import version
from peft import LoraConfig, get_peft_model, PeftModel
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# Settings
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "/Users/harshasekar/Downloads/comic_data/"
DATA_DIR = "/Users/harshasekar/Downloads/comic_data/ground_truth_images"
JSON_FILE = "/Users/harshasekar/Downloads/comics_data/results_20251204_093353.json"
TRAIN_STEPS = 500
LEARNING_RATE = 1e-4
BATCH_SIZE = 1
RESOLUTION = 512

def parse_args():
    parser = argparse.ArgumentParser(description="Simple LoRA Fine-tuning script")
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    return parser.parse_args()

class ComicDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, images_dir, tokenizer, size=512):
        self.tokenizer = tokenizer
        self.size = size
        self.data = []

        # Load Metadata
        with open(json_path, "r") as f:
            full_data = json.load(f)
            results = full_data.get("results", [])

        # Match results to images
        for item in results:
            comic = item.get("comic_no")
            page = item.get("target_page_no")
            panel = item.get("target_panel_no")
            desc = item.get("scene_description")

            if comic and page and panel and desc:
                # Construct filename based on known format
                filename = f"{comic}_{page}_{panel}.jpg"
                filepath = os.path.join(images_dir, filename)
                
                if os.path.exists(filepath):
                    self.data.append({"image": filepath, "caption": desc})
        
        print(f"Dataset: Found {len(self.data)} training pairs.")

        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image"]).convert("RGB")
        pixel_values = self.image_transforms(image)

        # Tokenize text
        text_inputs = self.tokenizer(
            item["caption"], 
            padding="max_length", 
            truncation=True, 
            max_length=self.tokenizer.model_max_length, 
            return_tensors="pt"
        )
        input_ids = text_inputs.input_ids[0]

        return {"pixel_values": pixel_values, "input_ids": input_ids}

def main():
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=1
    )

    # 1. Load Models
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(MODEL_NAME, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(MODEL_NAME, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(MODEL_NAME, subfolder="unet")

    # Freeze frozen parameters
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # 2. Add LoRA
    unet_lora_config = LoraConfig(
        r=16, 
        lora_alpha=16, 
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]
    )
    unet = get_peft_model(unet, unet_lora_config)
    unet.print_trainable_parameters()

    # 3. Optimizer
    # Try 8-bit Adam if available AND we have CUDA
    optimizer_class = torch.optim.AdamW
    if torch.cuda.is_available():
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            print("Using 8-bit AdamW")
        except ImportError:
            print("Using standard AdamW (bnb not found)")
    else:
        print("Using standard AdamW (CPU detected)")

    optimizer = optimizer_class(
        unet.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-2
    )

    # 4. Dataset
    dataset = ComicDataset(JSON_FILE, DATA_DIR, tokenizer, size=RESOLUTION)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 5. Prepare
    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    
    # Move huge models to device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # 6. Training Loop
    print(f"Starting training for {TRAIN_STEPS} steps...")
    progress_bar = tqdm(range(TRAIN_STEPS), disable=not accelerator.is_local_main_process)
    unet.train()
    
    global_step = 0
    while global_step < TRAIN_STEPS:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                # Convert images to latents
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()

                # Add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict noise
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
            
            if global_step >= TRAIN_STEPS:
                break

    # 7. Save
    print("Saving LoRA weights...")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(OUTPUT_DIR)
        print(f"Saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
