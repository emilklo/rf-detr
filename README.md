# RF-DETR Training on LaRS Dataset

This project trains [RF-DETR](https://github.com/roboflow/rf-detr) (Roboflow's Detection Transformer) for instance segmentation on the [LaRS (Labeled Maritime Segmentation)](https://lojzezust.github.io/lars-dataset/) dataset.

## Overview

RF-DETR is a real-time object detection and segmentation model that achieves state-of-the-art performance on COCO. This project fine-tunes RF-DETR Seg (Preview) on maritime obstacle detection using the LaRS dataset, which contains panoptic annotations for varied maritime environments.

## Dataset

**LaRS v1.0.0** - Panoptic dataset for obstacle detection in maritime environments

- **Training images**: 2,605 (16,951 annotations)
- **Validation images**: 198 (1,667 annotations)
- **Image resolution**: 1278 x 958 pixels

### Categories (11 classes)

1. Static Obstacle
2. Water
3. Sky
4. Boat/ship
5. Row boats
6. Paddle board
7. Buoy
8. Swimmer
9. Animal
10. Float
11. Other

## Setup

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended)
- [uv](https://github.com/astral-sh/uv) package manager 

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rf-detr
```

2. Install dependencies:
```bash
# Using uv (recommended)
uv sync
```

### Dataset Organization

The LaRS dataset should be organized as follows:

```
datasets/
├── lars_v1.0.0_images/
│   ├── train/images/
│   └── val/images/
├── lars_v1.0.0_annotations/
│   ├── train/lars_train_instances_all.json
│   └── val/lars_val_instances_all.json
└── lars_rfdetr/  (created automatically)
    ├── train/
    │   ├── images -> ../../lars_v1.0.0_images/train/images
    │   └── _annotations.coco.json
    └── valid/
        ├── images -> ../../lars_v1.0.0_images/val/images
        └── _annotations.coco.json
```

## Training

### Quick Start

Train using the standalone script:

```bash
uv run python train_rfdetr.py
```

### Training Configuration

Default training parameters:
- **Epochs**: 100
- **Batch size**: 4
- **Gradient accumulation steps**: 4
- **Learning rate**: 1e-4
- **Effective batch size**: 16 (batch_size × grad_accum_steps)
- **Output directory**: `output/`

### Using Jupyter Notebook

An interactive notebook is provided for exploratory training:

```bash
jupyter notebook notebooks/rf-detr.ipynb
```

The notebook includes:
1. Dataset verification and statistics
2. Sample image visualization
3. Training with progress monitoring
4. Inference and result visualization

### Custom Training

```python
from rfdetr import RFDETRSegPreview

model = RFDETRSegPreview()
model.train(
    dataset_dir="datasets/lars_rfdetr",
    epochs=100,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="output"
)
```

### Training Notes

- **GPU Memory**: Batch size of 4 works on T4 GPUs (16GB). For A100, you can increase to `batch_size=16, grad_accum_steps=1`
- **Training Time**: ~50-100 epochs recommended for good performance (several hours on modern GPUs)
- **Checkpoints**: Best model saved as `output/checkpoint_best_total.pth`

## Inference

### Command Line Inference

Run inference on a single image:

```bash
uv run python inference_rfdetr.py \
    --checkpoint output/checkpoint_best_total.pth \
    --image path/to/image.jpg \
    --output result.jpg \
    --threshold 0.5
```

### Programmatic Inference

```python
import cv2
from rfdetr import RFDETRSegPreview
import supervision as sv

# Load model
model = RFDETRSegPreview(pretrain_weights="output/checkpoint_best_total.pth")
model.optimize_for_inference()

# Run inference
image = cv2.imread("path/to/image.jpg")
detections = model.predict(image, threshold=0.5)

# Visualize results
mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()

annotated = mask_annotator.annotate(scene=image.copy(), detections=detections)
annotated = label_annotator.annotate(scene=annotated, detections=detections)

cv2.imwrite("output.jpg", annotated)
```

## Project Structure

```
rf-detr/
├── README.md                   # This file
├── pyproject.toml             # Project dependencies
├── train_rfdetr.py            # Training script
├── inference_rfdetr.py        # Inference script
├── notebooks/
│   └── rf-detr.ipynb          # Interactive training notebook
├── datasets/
│   ├── lars_v1.0.0_images/    # Original LaRS images
│   ├── lars_v1.0.0_annotations/ # Original LaRS annotations
│   └── lars_rfdetr/           # RF-DETR formatted dataset
└── output/                     # Training outputs (created during training)
    ├── checkpoint_best_total.pth
    └── logs/
```

## Model Architecture

RF-DETR (Roboflow Detection Transformer) combines:
- Transformer-based architecture
- Real-time inference capabilities
- State-of-the-art segmentation performance
- Optimized for fine-tuning on custom datasets

## Performance Tips

1. **Batch Size**: Adjust based on GPU memory. Keep `batch_size × grad_accum_steps = 16` for equivalent training
2. **Learning Rate**: Default `1e-4` works well for fine-tuning. Lower if training is unstable
3. **Epochs**: Start with 50-100 epochs. Monitor validation metrics to avoid overfitting
4. **Data Augmentation**: RF-DETR includes built-in augmentation during training

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size:
```python
model.train(batch_size=2, grad_accum_steps=8)  # Keeps effective batch size at 16
```

### Slow Training

- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Use A100 or V100 GPUs for faster training
- Enable mixed precision training (enabled by default in RF-DETR)

### Poor Results

- Train for more epochs (100+)
- Check dataset annotations for errors
- Adjust confidence threshold during inference
- Review class balance in dataset

## References

- [RF-DETR GitHub Repository](https://github.com/roboflow/rf-detr)
- [RF-DETR Training Guide](https://blog.roboflow.com/train-rf-detr-segmentation/)
- [LaRS Dataset](https://lojzezust.github.io/lars-dataset/)
- [RF-DETR Paper](https://arxiv.org/abs/2410.13159)

## Citation

If you use this code or the LaRS dataset, please cite:

```bibtex
@inproceedings{kristan2023lars,
  title={The LaRS Dataset - Labeled Maritime Segmentation},
  author={Kristan, Matej and others},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2023}
}

@article{rfdetr2024,
  title={RF-DETR: Towards Real-Time Detection Transformer},
  author={Roboflow},
  year={2024}
}
```

## License

Please refer to the respective licenses of RF-DETR and the LaRS dataset.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements.
