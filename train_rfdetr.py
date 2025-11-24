"""
Train RF-DETR segmentation model on LaRS dataset.

Based on the official Roboflow guide:
https://blog.roboflow.com/train-rf-detr-segmentation/
"""

import os
from pathlib import Path
from rfdetr import RFDETRSegPreview

def main():
    dataset_dir = "datasets/lars_rfdetr"
    output_dir = "output"

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    if not (dataset_path / "train" / "_annotations.coco.json").exists():
        raise ValueError(f"Train annotations not found in {dataset_dir}/train/")

    if not (dataset_path / "valid" / "_annotations.coco.json").exists():
        raise ValueError(f"Valid annotations not found in {dataset_dir}/valid/")

    print("=" * 60)
    print("RF-DETR Training on LaRS Dataset")
    print("=" * 60)
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print()

    model = RFDETRSegPreview()

    print("Starting training...")
    print("Training parameters:")
    print("  - Epochs: 100")
    print("  - Batch size: 4")
    print("  - Gradient accumulation steps: 4")
    print("  - Learning rate: 1e-4")
    print("  - Effective batch size: 16 (batch_size * grad_accum_steps)")
    print()

    model.train(
        dataset_dir=dataset_dir,
        epochs=100,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir=output_dir
    )

    print()
    print("=" * 60)
    print("Training completed!")
    print(f"Model checkpoints saved to: {output_dir}/")
    print("Best model: checkpoint_best_total.pth")
    print("=" * 60)

if __name__ == "__main__":
    main()
