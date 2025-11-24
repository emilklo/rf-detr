"""
Inference script for RF-DETR segmentation model.

Load a trained model and run inference on images.
"""

import argparse
from pathlib import Path
import cv2
from rfdetr import RFDETRSegPreview
import supervision as sv

def main():
    parser = argparse.ArgumentParser(description="Run RF-DETR inference on images")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/checkpoint_best_total.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_image.jpg",
        help="Path to save output image"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for detections"
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint not found: {args.checkpoint}")

    image_path = Path(args.image)
    if not image_path.exists():
        raise ValueError(f"Image not found: {args.image}")

    print(f"Loading model from: {args.checkpoint}")
    model = RFDETRSegPreview(pretrain_weights=args.checkpoint)

    print("Optimizing model for inference...")
    model.optimize_for_inference()

    print(f"Loading image: {args.image}")
    image = cv2.imread(args.image)

    print(f"Running inference (threshold={args.threshold})...")
    detections = model.predict(image, threshold=args.threshold)

    print(f"Found {len(detections)} detections")

    mask_annotator = sv.MaskAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = mask_annotator.annotate(
        scene=image.copy(),
        detections=detections
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=detections
    )

    cv2.imwrite(args.output, annotated_image)
    print(f"Saved annotated image to: {args.output}")

if __name__ == "__main__":
    main()
