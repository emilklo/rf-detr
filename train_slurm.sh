#!/bin/bash
#SBATCH --job-name=rfdetr-train
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/rfdetr_%j.out
#SBATCH --error=logs/rfdetr_%j.err

echo "Starting RF-DETR training job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo ""

cd /cluster/work/emilkl/rf-detr

# Set up distributed training environment variables (required even for single GPU)
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1

# CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Show GPU info
echo "GPU Information:"
nvidia-smi
echo ""

# Check CUDA availability
echo "PyTorch CUDA check:"
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}'); print(f'PyTorch version: {torch.__version__}')"
echo ""

# Dataset verification
echo "Dataset verification:"
echo "Train annotations:"
ls -lh datasets/lars_rfdetr/train/_annotations.coco.json
echo "Valid annotations:"
ls -lh datasets/lars_rfdetr/valid/_annotations.coco.json
echo "Train images:"
ls datasets/lars_rfdetr/train/images | wc -l
echo ""

# Auto-detect GPU type and set profile
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [[ $GPU_NAME == *"A100"* ]]; then
    PROFILE="a100"
    echo "Detected A100 GPU - using A100 profile"
elif [[ $GPU_NAME == *"V100"* ]]; then
    PROFILE="v100"
    echo "Detected V100 GPU - using V100 profile"
else
    PROFILE="a100"
    echo "Unknown GPU type - using default A100 profile"
fi
echo ""

# Show current config
echo "Current training configuration:"
echo "------------------------------------------------------------"
cat config/train_config.yaml
echo "------------------------------------------------------------"
echo ""

# Run training with detected profile
echo "Starting training with profile: $PROFILE"
uv run python train.py --config config/train_config.yaml --profile $PROFILE

EXIT_CODE=$?

echo ""
echo "======================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code: $EXIT_CODE"
fi
echo "Date: $(date)"
echo "Check logs for output directory location"
echo "======================================================================"

exit $EXIT_CODE
