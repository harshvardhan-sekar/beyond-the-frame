#!/bin/bash
# =============================================================================
# SLURM Batch Script: Evaluate Fine-tuned Model
# =============================================================================
# Submit: sbatch scripts/run_evaluation.sbatch
# =============================================================================

#SBATCH --job-name=comics-eval
#SBATCH --partition=gpuH200
#SBATCH --account=bftl-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_DIR="/scratch/bftl/hsekar/comics_project"
CONDA_ENV_PATH="/u/hsekar/comics_env"
CHECKPOINT_PATH="${PROJECT_DIR}/checkpoints/llava_comics_lora_pred"
NUM_TEST_EXAMPLES=100

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

echo "============================================================"
echo "COMICS EVALUATION JOB"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Test examples: $NUM_TEST_EXAMPLES"
echo "============================================================"

module purge
module load anaconda3_gpu/24.1.0
module load cuda/12.2

source activate $CONDA_ENV_PATH

export HF_HOME="${PROJECT_DIR}/model_cache"
export TRANSFORMERS_CACHE="${PROJECT_DIR}/model_cache"
export CUDA_VISIBLE_DEVICES=0

cd $PROJECT_DIR

# =============================================================================
# RUN EVALUATION
# =============================================================================

echo ""
echo "Starting evaluation..."
nvidia-smi --query-gpu=name,memory.free --format=csv
echo ""

python scripts/evaluate_finetuned.py \
    --project_dir $PROJECT_DIR \
    --checkpoint_path $CHECKPOINT_PATH \
    --num_examples $NUM_TEST_EXAMPLES \
    --output_dir "${PROJECT_DIR}/outputs/finetuned_pred" \
    2>&1 | tee logs/eval_${SLURM_JOB_ID}_full.log

echo ""
echo "============================================================"
echo "Evaluation completed at: $(date)"
echo "Results saved to: ${PROJECT_DIR}/outputs/finetuned_pred"
echo "============================================================"
