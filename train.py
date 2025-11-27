"""
Train RF-DETR segmentation model on LaRS dataset.

Based on the official Roboflow guide:
https://blog.roboflow.com/train-rf-detr-segmentation/

Usage:
    python train.py --config config/train_config.yaml
    python train.py --config config/train_config.yaml --profile a100
    python train.py --config config/train_config.yaml --epochs 50 --batch-size 4
"""

import argparse
from pathlib import Path
import yaml
from rfdetr import RFDETRSegPreview


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_output_dir(config):
    """
    Generate output directory name based on training parameters.

    Format: output_lr<lr>_bs<effective_batch>_ep<epochs>
    Example: output_lr0.001_bs16_ep100
    """
    training = config['training']
    lr = training['lr']
    batch_size = training['batch_size']
    grad_accum = training['grad_accum_steps']
    epochs = training['epochs']

    effective_batch = batch_size * grad_accum

    # Format learning rate nicely (remove trailing zeros)
    lr_str = f"{lr:.6f}".rstrip('0').rstrip('.')

    output_dir = f"output_lr{lr_str}_bs{effective_batch}_ep{epochs}"

    return output_dir


def merge_args_with_config(config, args):
    """Merge command-line arguments with config file."""
    # Apply profile if specified
    if args.profile and args.profile in config.get('profiles', {}):
        profile = config['profiles'][args.profile]
        print(f"Applying profile: {args.profile}")
        config['training'].update(profile)

    # Override with command-line arguments
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.grad_accum_steps is not None:
        config['training']['grad_accum_steps'] = args.grad_accum_steps
    if args.lr is not None:
        config['training']['lr'] = args.lr
    if args.dataset_dir is not None:
        config['dataset']['dir'] = args.dataset_dir
    if args.output_dir is not None:
        config['output']['dir'] = args.output_dir
    if args.checkpoint is not None:
        config['model']['pretrain_weights'] = args.checkpoint

    return config


def main():
    parser = argparse.ArgumentParser(description="Train RF-DETR on LaRS dataset")

    # Config file
    parser.add_argument(
        '--config',
        type=str,
        default='config/train_config.yaml',
        help='Path to config file'
    )

    # Profile selection
    parser.add_argument(
        '--profile',
        type=str,
        choices=['v100', 'a100', 'cpu'],
        help='Hardware profile (v100, a100, cpu)'
    )

    # Training parameters (override config)
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--grad-accum-steps', type=int, help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, help='Learning rate')

    # Paths (override config)
    parser.add_argument('--dataset-dir', type=str, help='Dataset directory')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--checkpoint', type=str, help='Path to pretrained checkpoint')

    args = parser.parse_args()

    # Load and merge config
    config = load_config(args.config)
    config = merge_args_with_config(config, args)

    # Extract settings
    dataset_dir = config['dataset']['dir']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    grad_accum_steps = config['training']['grad_accum_steps']
    lr = config['training']['lr']
    pretrain_weights = config['model']['pretrain_weights']

    # Auto-generate output directory based on training params if not specified
    if args.output_dir is None and (config['output']['dir'] == 'output' or config['output']['dir'] == 'auto'):
        output_dir = generate_output_dir(config)
        print(f"Auto-generated output directory: {output_dir}")
    else:
        output_dir = config['output']['dir']

    # Validate dataset
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")

    if not (dataset_path / "train" / "_annotations.coco.json").exists():
        raise ValueError(f"Train annotations not found in {dataset_dir}/train/")

    if not (dataset_path / "valid" / "_annotations.coco.json").exists():
        raise ValueError(f"Valid annotations not found in {dataset_dir}/valid/")

    # Print configuration
    print("=" * 60)
    print("RF-DETR Training on LaRS Dataset")
    print("=" * 60)
    print(f"Configuration: {args.config}")
    if args.profile:
        print(f"Profile: {args.profile}")
    print()
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print()
    print("Training parameters:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation steps: {grad_accum_steps}")
    print(f"  Effective batch size: {batch_size * grad_accum_steps}")
    print(f"  Learning rate: {lr}")
    if pretrain_weights:
        print(f"  Pretrained weights: {pretrain_weights}")
    print()

    # Initialize model
    if pretrain_weights:
        model = RFDETRSegPreview(pretrain_weights=pretrain_weights)
    else:
        model = RFDETRSegPreview()

    # Train
    print("Starting training...")
    model.train(
        dataset_dir=dataset_dir,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        lr=lr,
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
