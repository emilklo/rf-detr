# RF-DETR Maritime Obstacle Detection

Training and evaluation of [RF-DETR](https://github.com/roboflow/rf-detr) instance segmentation models on the LaRS maritime obstacle detection dataset.

## Overview

This repository implements **real-time instance segmentation** for maritime obstacle detection using RF-DETR (Roboflow DETR), a state-of-the-art transformer-based model. The model is trained on the [LaRS dataset](https://lojzezust.github.io/lars-dataset/), a diverse panoptic maritime obstacle detection dataset.

### What is Instance Segmentation?

Instance segmentation combines object detection with pixel-level segmentation:
- **Detects objects** with bounding boxes
- **Generates pixel-level masks** for each individual object instance
- **Classifies** each instance into categories (e.g., boats, static obstacles, water, sky)

Unlike semantic segmentation which labels all pixels of the same class identically, instance segmentation distinguishes between separate instances of the same class.

## RF-DETR Architecture

RF-DETR Seg uses a **DETR-based (Detection Transformer)** architecture with the following key components:

### Core Architecture
- **Backbone**: DINOv2 Vision Transformer (ViT)
- **Decoder**: Deformable cross-attention transformer decoder
- **Masking Head**: Inspired by MaskDINO with memory-efficient loss implementation

### How It Works
1. **Feature Extraction**: DINOv2 ViT backbone extracts visual features from input images
2. **Deformable Attention**: Cross-attention mechanism focuses on relevant spatial locations
3. **Bilinear Upsampling**: Compensates for non-hierarchical ViT backbone to recover high-resolution features
4. **Layer-wise Mask Refinement**: Each decoder layer refines segmentation masks progressively
5. **End-to-End Training**: Object detection and segmentation trained jointly

### Performance
- **Speed**: 170 FPS on T4 GPU (5.6ms latency)
- **Accuracy**: State-of-the-art on COCO instance segmentation
- **Efficiency**: 3x faster than YOLO11 with higher accuracy

Learn more: [RF-DETR Segmentation](https://blog.roboflow.com/rf-detr-segmentation-preview/)

## Dataset

### LaRS Dataset
The **LaRS (Large-scale Annotated Real-world Scenes)** dataset is a panoptic maritime obstacle detection dataset developed by the University of Ljubljana.

**Categories**:
- Static Obstacle
- Boat/Ship
- Water
- Sky
- And more...

**Format**: COCO JSON with instance segmentation annotations

**Source**: [LaRS Dataset Website](https://lojzezust.github.io/lars-dataset/)

## Repository Structure

```
rf-detr/
├── train.py                          # Main training script with YAML config support
├── train_slurm.sh                    # SLURM job script for cluster training
├── visualize_model_comparison.py     # Compare two model checkpoints side-by-side
├── visualize_comparison_slurm.sh     # SLURM job for visualization
├── config/
│   └── train_config.yaml             # Training configuration (hyperparameters, paths)
├── datasets/
│   └── lars_rfdetr/                  # LaRS dataset in COCO format
│       ├── train/
│       │   ├── images/
│       │   └── _annotations.coco.json
│       └── valid/
│           ├── images/
│           └── _annotations.coco.json
├── models/
│   ├── output/                       # First trained model checkpoints
│   └── output_lr0.001_bs16_ep100/    # Second trained model (lr=1e-3, bs=16, 100 epochs)
└── visualizations/
    └── visualizations_model_comparison_20251125_131443/  # Model comparison outputs
```

## Training

### Configuration

Training is configured via YAML files in `config/`:

```yaml
# config/train_config.yaml
dataset:
  dir: "datasets/lars_rfdetr"

training:
  epochs: 100
  batch_size: 8
  grad_accum_steps: 2
  lr: 1.0e-3

output:
  dir: "auto"  # Auto-generates output_lr{lr}_bs{batch}_ep{epochs}

profiles:
  a100:
    batch_size: 8
    grad_accum_steps: 2
  v100:
    batch_size: 4
    grad_accum_steps: 4
```

### Running Training

**Local:**
```bash
# Use config file with auto-detected GPU profile
python train.py --config config/train_config.yaml --profile a100

# Override hyperparameters
python train.py --config config/train_config.yaml --epochs 50 --lr 5e-4
```

**SLURM Cluster:**
```bash
# Auto-detects GPU type and applies appropriate profile
sbatch train_slurm.sh
```

### Output Directory Naming

The training script auto-generates descriptive output directories:
- Format: `output_lr{lr}_bs{effective_batch}_ep{epochs}`
- Example: `output_lr0.001_bs16_ep100` means:
  - Learning rate: 0.001
  - Effective batch size: 16
  - Epochs: 100

## Trained Models

### Model Checkpoints

Each training run produces:
- `checkpoint_best_total.pth` - Best overall model (smallest total loss)
- `checkpoint_best_ema.pth` - Best EMA model
- `checkpoint_best_regular.pth` - Best regular model
- `checkpoint00XX.pth` - Periodic checkpoints every 10 epochs
- `results.json` - Training metrics
- `log.txt` - Full training logs

### Available Models

Two trained models are included:
1. **`models/output/`** - Baseline model
2. **`models/output_lr0.001_bs16_ep100/`** - Optimized model (lr=1e-3, 100 epochs)

## Model Comparison Visualization

Compare two model checkpoints side-by-side with ground truth:

```bash
# Local
python visualize_model_comparison.py \
    --checkpoint-a models/output/checkpoint_best_total.pth \
    --checkpoint-b models/output_lr0.001_bs16_ep100/checkpoint_best_total.pth \
    --images-dir datasets/lars_rfdetr/valid/images \
    --annotations datasets/lars_rfdetr/valid/_annotations.coco.json \
    --output-dir visualizations/comparison \
    --max-images 20 \
    --threshold 0.4 \
    --label-a "Baseline" \
    --label-b "Optimized"

# SLURM cluster
sbatch visualize_comparison_slurm.sh
```

**Output**: 2x2 grids with:
- Top-left: Original image
- Top-right: Ground truth annotations
- Bottom-left: Model A predictions
- Bottom-right: Model B predictions

## Usage

### Install Dependencies

```bash
# Using uv (recommended)
uv sync

# Using pip
pip install -r requirements.txt
```

### Inference

```python
from rfdetr import RFDETRSegPreview
import cv2

# Load trained model
model = RFDETRSegPreview(
    pretrain_weights="models/output_lr0.001_bs16_ep100/checkpoint_best_total.pth"
)
model.optimize_for_inference()

# Run inference
image = cv2.imread("path/to/image.jpg")
detections = model.predict(image, threshold=0.4)

# detections contains:
# - xyxy: bounding boxes
# - class_id: class labels
# - confidence: detection scores
# - mask: instance segmentation masks
```

## Requirements

- Python >= 3.11
- PyTorch with CUDA support
- rfdetr >= 1.3.0
- supervision >= 0.27.0
- PyYAML >= 6.0
- OpenCV
- Matplotlib

See `pyproject.toml` for full dependency list.

## Hardware Requirements

### Recommended
- **GPU**: NVIDIA A100 (40GB) or V100 (32GB)
- **Memory**: 64GB RAM
- **Storage**: 100GB+ for datasets and model checkpoints

### Minimum
- **GPU**: Any CUDA-capable GPU with 16GB+ VRAM
- **Memory**: 32GB RAM

## References

- [RF-DETR GitHub](https://github.com/roboflow/rf-detr)
- [RF-DETR Segmentation Blog Post](https://blog.roboflow.com/rf-detr-segmentation-preview/)
- [How to Train RF-DETR](https://blog.roboflow.com/train-rf-detr-segmentation/)
- [LaRS Dataset](https://lojzezust.github.io/lars-dataset/)
- [LaRS Paper (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Zust_LaRS_A_Diverse_Panoptic_Maritime_Obstacle_Detection_Dataset_and_Benchmark_ICCV_2023_paper.pdf)

## License

This project uses:
- RF-DETR: [Apache 2.0 License](https://github.com/roboflow/rf-detr/blob/main/LICENSE)
- LaRS Dataset: Check [dataset website](https://lojzezust.github.io/lars-dataset/) for terms

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{zust2023lars,
  title={LaRS: A Diverse Panoptic Maritime Obstacle Detection Dataset and Benchmark},
  author={Žust, Lojze and Žust, Luka and Kristan, Matej},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}

@misc{roboflow2024rfdetr,
  title={RF-DETR: Real-Time Object Detection and Segmentation},
  author={Roboflow},
  year={2024},
  url={https://github.com/roboflow/rf-detr}
}
```
