#!/bin/bash
#SBATCH --job-name=rfdetr-compare
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --time=03:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=6
#SBATCH --output=logs/rfdetr_compare_%j.out
#SBATCH --error=logs/rfdetr_compare_%j.err

echo "Starting RF-DETR model comparison visualization job (A100)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo ""

cd /cluster/work/emilkl/rf-detr
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "GPU Information:"
nvidia-smi
echo ""

echo "PyTorch CUDA check:"
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo ""

CHECKPOINT_A="output/checkpoint_best_total.pth"
CHECKPOINT_B="output_lr0.001_bs16_ep100/checkpoint_best_total.pth"
IMAGES_DIR="datasets/lars_rfdetr/valid/images"
ANNOTATIONS="datasets/lars_rfdetr/valid/_annotations.coco.json"
OUTPUT_DIR="visualizations_model_comparison_$(date +%Y%m%d_%H%M%S)"

echo "Checking inputs..."
[[ -f $CHECKPOINT_A ]] || { echo "Missing checkpoint A: $CHECKPOINT_A" && exit 1; }
[[ -f $CHECKPOINT_B ]] || { echo "Missing checkpoint B: $CHECKPOINT_B" && exit 1; }
[[ -d $IMAGES_DIR ]] || { echo "Missing images dir: $IMAGES_DIR" && exit 1; }
[[ -f $ANNOTATIONS ]] || { echo "Missing annotations: $ANNOTATIONS" && exit 1; }
echo "✓ Inputs found"
echo ""

mkdir -p "$OUTPUT_DIR"
echo "Output will be saved to: $OUTPUT_DIR"
echo ""

echo "Launching comparison visualization..."
uv run python visualize_model_comparison.py \
    --checkpoint-a "$CHECKPOINT_A" \
    --checkpoint-b "$CHECKPOINT_B" \
    --images-dir "$IMAGES_DIR" \
    --annotations "$ANNOTATIONS" \
    --output-dir "$OUTPUT_DIR" \
    --max-images 20 \
    --threshold 0.4 \
    --label-a "output" \
    --label-b "output_lr0.001_bs16_ep100"

echo ""
echo "Job completed: $(date)"
echo "Generated files in: $OUTPUT_DIR"
