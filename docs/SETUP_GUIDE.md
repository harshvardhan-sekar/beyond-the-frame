# 🔧 Setup Guide: COMICS Cloze VLM

This guide walks you through setting up the project on different environments.

---

## Table of Contents

1. [Local Setup (Mac/Linux)](#local-setup)
2. [Google Colab Setup](#google-colab-setup)
3. [NCSA Delta HPC Setup](#ncsa-delta-hpc-setup)
4. [Data Preparation](#data-preparation)
5. [Troubleshooting](#troubleshooting)

---

## Local Setup

### Prerequisites

- Python 3.10+
- Git
- ~50GB free disk space
- (Optional) CUDA-compatible GPU with 16GB+ VRAM

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/comics-cloze-vlm.git
cd comics-cloze-vlm
```

### Step 2: Create Environment

**Option A: Conda (Recommended)**

```bash
conda env create -f envs/environment.yml
conda activate comics-vlm
```

**Option B: venv + pip**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Verify Installation

```python
import torch
import transformers
from peft import LoraConfig

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Transformers: {transformers.__version__}")
print("✅ Installation successful!")
```

---

## Google Colab Setup

### Step 1: Open Notebook

1. Go to [Google Colab](https://colab.research.google.com)
2. File → Open notebook → GitHub tab
3. Enter repository URL
4. Select desired notebook

### Step 2: Configure Runtime

1. Runtime → Change runtime type
2. Select **T4 GPU** (free) or **A100** (Pro)
3. Click Save

### Step 3: Install Dependencies

Run this cell first in any notebook:

```python
!pip install -q transformers accelerate peft bitsandbytes
!pip install -q datasets evaluate bert-score rouge-score
```

### Step 4: Mount Google Drive (for data)

```python
from google.colab import drive
drive.mount('/content/drive')

# Your data should be at:
# /content/drive/MyDrive/comics_project/data/
```

---

## NCSA Delta HPC Setup

### Step 1: Login to Delta

```bash
# SSH login
ssh your_netid@login.delta.ncsa.illinois.edu

# Enter DUO 2FA when prompted
```

### Step 2: Initial Setup

```bash
# Create project directory
mkdir -p /scratch/bftl/$USER/comics_project
cd /scratch/bftl/$USER/comics_project

# Clone repository
git clone https://github.com/YOUR_USERNAME/comics-cloze-vlm.git .
```

### Step 3: Create Conda Environment

```bash
# Load anaconda module
module load anaconda3_gpu/24.1.0

# Create environment
conda create -n comics_env python=3.10 -y
conda activate comics_env

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt
```

### Step 4: Transfer Data

**Option A: SCP from local machine**

```bash
# From your local machine:
scp -r /path/to/data your_netid@login.delta.ncsa.illinois.edu:/scratch/bftl/your_netid/comics_project/data/
```

**Option B: Download from GCS**

```bash
# On Delta:
module load google-cloud-sdk
gsutil -m cp -r gs://your-bucket/comics_data/* ./data/
```

### Step 5: Submit Jobs

```bash
# Submit training job
sbatch scripts/run_finetune.sbatch

# Check job status
squeue -u $USER

# View logs
tail -f logs/finetune_*.out
```

### Step 6: Interactive Session (for debugging)

```bash
# Request interactive GPU session
srun --partition=gpuH200 --account=bftl-delta-gpu \
     --nodes=1 --gpus-per-node=1 --cpus-per-task=8 \
     --mem=64G --time=02:00:00 --pty bash

# Once allocated:
module load anaconda3_gpu/24.1.0
conda activate comics_env
python  # Now you can test interactively
```

---

## Data Preparation

### Expected Data Structure

```
data/
├── images/                    # Panel images
│   ├── comic_0001/
│   │   ├── 0_0.jpg           # page_panel.jpg format
│   │   ├── 0_1.jpg
│   │   └── ...
│   └── comic_0002/
│       └── ...
├── ocr/                       # OCR transcriptions
│   └── COMICS_OCR_WAVE1_sorted.csv
└── processed/                 # Generated during pipeline
    ├── train_sequences.pkl
    ├── val_sequences.pkl
    └── test_sequences.pkl
```

### OCR CSV Format

The OCR CSV should have these columns:

| Column | Description |
|--------|-------------|
| `comic_no` | Comic book number (0-1447) |
| `page_no` | Page number within comic |
| `panel_no` | Panel number within page |
| `x1, y1, x2, y2` | Bounding box coordinates |
| `text` | Extracted text |
| `confidence` | OCR confidence score |

### Running Data Pipeline

```bash
# 1. Data exploration
jupyter notebook notebooks/01_data_pipeline.ipynb

# 2. Create sequences
jupyter notebook notebooks/02_data_cleaning_pipeline.ipynb
```

This will create:
- `train_sequences.pkl` (~249K sequences)
- `val_sequences.pkl` (~53K sequences)
- `test_sequences.pkl` (~55K sequences)

---

## Troubleshooting

### CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size in config
2. Use gradient checkpointing
3. Reduce image resolution (max 384px)
4. Use 4-bit quantization

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use 4-bit quantization
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
```

### Module Not Found

**Symptoms**: `ModuleNotFoundError: No module named 'xxx'`

**Solution**:
```bash
# Make sure you're in the right environment
conda activate comics-vlm
# Or
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### SLURM Job Fails Immediately

**Symptoms**: Job exits with status 1 within seconds

**Solutions**:
1. Check error log: `cat logs/finetune_*.err`
2. Verify paths in batch script
3. Check module availability: `module avail anaconda`
4. Test interactively first

### HuggingFace Download Issues

**Symptoms**: Model download hangs or fails

**Solutions**:
```bash
# Set cache directory
export HF_HOME=/scratch/bftl/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/bftl/$USER/hf_cache

# Use offline mode after initial download
export TRANSFORMERS_OFFLINE=1
```

### Jupyter Kernel Dies

**Symptoms**: Kernel restarts during cell execution

**Solutions**:
1. Request more memory in Colab/SLURM
2. Process data in smaller batches
3. Clear GPU cache periodically:

```python
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
```

---

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **NCSA Help**: help@ncsa.illinois.edu
- **HuggingFace Forums**: For transformer-related questions

---

## Quick Reference

### Common Commands

```bash
# Activate environment
conda activate comics-vlm

# Submit job
sbatch scripts/run_finetune.sbatch

# Check job status
squeue -u $USER

# Cancel job
scancel JOB_ID

# View GPU usage
nvidia-smi

# Watch training progress
tail -f logs/finetune_*.out
```

### Important Paths (Delta)

```
/scratch/bftl/$USER/comics_project/     # Project root
├── data/                               # Data files
├── checkpoints/                        # Model checkpoints
├── outputs/                            # Results
├── logs/                               # Job logs
└── model_cache/                        # HuggingFace cache
```
