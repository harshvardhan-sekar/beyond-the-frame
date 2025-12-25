#!/bin/bash
# =============================================================================
# SLURM Batch Script: Fine-tune LLaVA with LoRA
# =============================================================================
# Submit: sbatch scripts/run_finetune.sbatch
# Monitor: squeue -u $USER && tail -f logs/finetune_*.out
# =============================================================================

#SBATCH --job-name=comics-finetune
#SBATCH --partition=gpuH200
#SBATCH --account=bftl-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@illinois.edu

# =============================================================================
# CONFIGURATION - EDIT THESE!
# =============================================================================

PROJECT_DIR="/scratch/bftl/hsekar/comics_project"
CONDA_ENV_PATH="/u/hsekar/comics_env"

# Training size: "quick" (~30 min), "medium" (~3 hrs), "full" (~8+ hrs)
TRAINING_SIZE="medium"

# Task type: "prediction" or "description"
TASK_TYPE="prediction"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

echo "============================================================"
echo "COMICS FINE-TUNING JOB"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Training size: $TRAINING_SIZE"
echo "Task type: $TASK_TYPE"
echo "============================================================"

# Load modules
module purge
module load anaconda3_gpu/24.1.0
module load cuda/12.2

# Activate environment
source activate $CONDA_ENV_PATH

# Set environment variables
export HF_HOME="${PROJECT_DIR}/model_cache"
export TRANSFORMERS_CACHE="${PROJECT_DIR}/model_cache"
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Navigate to project
cd $PROJECT_DIR

# =============================================================================
# RUN TRAINING
# =============================================================================

echo ""
echo "Starting training..."
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

python scripts/finetune_lora.py \
    --project_dir $PROJECT_DIR \
    --training_size $TRAINING_SIZE \
    --task_type $TASK_TYPE \
    2>&1 | tee logs/finetune_${SLURM_JOB_ID}_full.log

# =============================================================================
# COMPLETION
# =============================================================================

echo ""
echo "============================================================"
echo "Job completed at: $(date)"
echo "============================================================"
