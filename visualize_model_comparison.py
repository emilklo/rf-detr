"""
Create 2x2 grids comparing two RF-DETR checkpoints against ground truth.

Top-left: raw image
Top-right: ground truth
Bottom-left: predictions from checkpoint A
Bottom-right: predictions from checkpoint B
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import supervision as sv
from rfdetr import RFDETRSegPreview


def empty_detections() -> sv.Detections:
    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        class_id=np.array([], dtype=int),
        confidence=np.array([], dtype=np.float32),
        mask=None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two RF-DETR models side by side.")
    parser.add_argument(
        "--checkpoint-a",
        type=str,
        required=True,
        help="Path to first checkpoint (bottom-left panel).",
    )
    parser.add_argument(
        "--checkpoint-b",
        type=str,
        required=True,
        help="Path to second checkpoint (bottom-right panel).",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory with input images.",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="COCO annotations json with ground truth.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations_comparison",
        help="Directory to save 2x2 grids.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Number of images to visualize.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Confidence threshold for predictions.",
    )
    parser.add_argument(
        "--label-a",
        type=str,
        default="Model A",
        help="Label for checkpoint A in the grid title.",
    )
    parser.add_argument(
        "--label-b",
        type=str,
        default="Model B",
        help="Label for checkpoint B in the grid title.",
    )
    return parser.parse_args()


def load_coco_annotations(
    annotations_path: Path,
) -> Tuple[Dict[str, int], Dict[int, List[dict]], Dict[int, str]]:
    with annotations_path.open() as f:
        coco = json.load(f)

    image_id_by_file = {
        Path(img["file_name"]).name: img["id"] for img in coco.get("images", [])
    }

    anns_by_image: Dict[int, List[dict]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    category_names = {
        int(cat["id"]): cat.get("name", str(cat["id"]))
        for cat in coco.get("categories", [])
    }

    return image_id_by_file, anns_by_image, category_names


def load_model(checkpoint_path: Path) -> RFDETRSegPreview:
    print(f"Loading model from {checkpoint_path}")
    model = RFDETRSegPreview(pretrain_weights=str(checkpoint_path))
    model.optimize_for_inference()
    return model


def annotations_to_detections(
    anns: List[dict],
) -> sv.Detections:
    xyxy = []
    class_ids = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        xyxy.append([x, y, x + w, y + h])
        class_ids.append(int(ann["category_id"]))

    if not xyxy:
        return empty_detections()

    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
        confidence=np.ones(len(xyxy), dtype=np.float32),
    )


def annotate_image(
    image_bgr: np.ndarray,
    detections: sv.Detections,
    category_names: Dict[int, str],
    show_scores: bool,
    empty_message: str,
) -> np.ndarray:
    annotated = image_bgr.copy()

    if detections is None or len(detections) == 0:
        cv2.putText(
            annotated,
            empty_message,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels = []
    for class_id, conf in zip(detections.class_id, detections.confidence):
        name = category_names.get(int(class_id), f"id {class_id}")
        if show_scores and conf is not None:
            labels.append(f"{name} {conf:.2f}")
        else:
            labels.append(name)

    mask_data = getattr(detections, "mask", None)
    if mask_data is not None:
        annotated = mask_annotator.annotate(scene=annotated, detections=detections)
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(
        scene=annotated, detections=detections, labels=labels
    )
    return annotated


def compose_grid(
    original_bgr: np.ndarray,
    gt_bgr: np.ndarray,
    model_a_bgr: np.ndarray,
    model_b_bgr: np.ndarray,
    titles: Tuple[str, str, str, str],
    save_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    panels = [original_bgr, gt_bgr, model_a_bgr, model_b_bgr]

    for ax, panel, title in zip(axes.flat, panels, titles):
        ax.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    images_dir = Path(args.images_dir)
    annotations_path = Path(args.annotations)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_id_by_file, anns_by_image, category_names = load_coco_annotations(
        annotations_path
    )
    total_annotations = sum(len(v) for v in anns_by_image.values())
    print(
        f"Loaded annotations for {len(image_id_by_file)} images "
        f"with {total_annotations} objects."
    )
    if total_annotations == 0:
        print("Warning: no ground truth annotations found; GT panels will be empty.")
    if not category_names:
        print("Warning: no categories found in annotations file.")

    checkpoint_a = Path(args.checkpoint_a)
    checkpoint_b = Path(args.checkpoint_b)
    model_a = load_model(checkpoint_a)
    model_b = load_model(checkpoint_b)

    image_files = sorted(
        list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    )
    if args.max_images:
        image_files = image_files[: args.max_images]

    print(f"Found {len(image_files)} images. Saving results to {output_dir}")

    for idx, image_path in enumerate(image_files, start=1):
        print(f"[{idx}/{len(image_files)}] {image_path.name}")
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  Skipping: unable to read {image_path}")
            continue

        image_id = image_id_by_file.get(image_path.name)
        if image_id is None:
            gt_dets = empty_detections()
        else:
            gt_dets = annotations_to_detections(anns_by_image.get(image_id, []))

        # Run inference
        pred_a = model_a.predict(image, threshold=args.threshold)
        pred_b = model_b.predict(image, threshold=args.threshold)

        original_panel = image
        gt_panel = annotate_image(
            image, gt_dets, category_names, show_scores=False, empty_message="No ground truth"
        )
        model_a_panel = annotate_image(
            image,
            pred_a,
            category_names,
            show_scores=True,
            empty_message="No predictions",
        )
        model_b_panel = annotate_image(
            image,
            pred_b,
            category_names,
            show_scores=True,
            empty_message="No predictions",
        )

        titles = (
            "Original",
            "Ground Truth",
            f"{args.label_a} ({checkpoint_a.parent.name or checkpoint_a.name})",
            f"{args.label_b} ({checkpoint_b.parent.name or checkpoint_b.name})",
        )

        save_path = output_dir / f"{image_path.stem}_comparison.jpg"
        compose_grid(original_panel, gt_panel, model_a_panel, model_b_panel, titles, save_path)
        print(f"  Saved to {save_path}")


if __name__ == "__main__":
    main()
