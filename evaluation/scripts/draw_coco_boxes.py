from pathlib import Path
import argparse
import json
import cv2


def draw_coco_boxes(dataset_dir: Path, split: str):
    images_dir = dataset_dir / "images" / split
    annotations_file = dataset_dir / "annotations" / f"instances_{split}.json"
    output_dir = dataset_dir / "debug_overlays" / split

    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if not annotations_file.exists():
        raise FileNotFoundError(
            f"COCO annotations not found: {annotations_file}"
        )

    # load COCO JSON
    with open(annotations_file, "r") as f:
        coco = json.load(f)

    categories = {
        category["id"]: category["name"]
        for category in coco["categories"]
    }

    print("Categories:")
    for cid, name in categories.items():
        print(f"  {cid}: {name}")

    images = {
        image["id"]: image
        for image in coco["images"]
    }

    annotations_by_image = {}

    for annotation in coco["annotations"]:
        annotations_by_image.setdefault(
            annotation["image_id"],
            []
        ).append(annotation)

    colors = {}

    for category_id, class_name in categories.items():
        if class_name == "crop":
            colors[category_id] = (0, 255, 0)      # green
        elif class_name == "weed":
            colors[category_id] = (0, 0, 255)      # red
        else:
            colors[category_id] = (255, 255, 0)

    print()
    print(f"Split: {split}")
    print(f"Images in COCO JSON: {len(images)}")
    print(f"Annotations: {len(coco['annotations'])}")
    print()

    # Draw
    for image_id, image_info in images.items():

        filename = image_info["file_name"]
        image_path = images_dir / Path(filename).name

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[WARN] Cannot read: {image_path}")
            continue

        height, width = image.shape[:2]

        annotations = annotations_by_image.get(
            image_id,
            []
        )

        for annotation in annotations:

            category_id = annotation["category_id"]

            # COCO bbox: [x_min, y_min, width, height]
            x, y, box_w, box_h = annotation["bbox"]

            x1 = int(round(x))
            y1 = int(round(y))
            x2 = int(round(x + box_w))
            y2 = int(round(y + box_h))

            # Clamp
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            color = colors.get(
                category_id,
                (255, 255, 0)
            )

            class_name = categories.get(
                category_id,
                f"class_{category_id}"
            )

            # bounding box
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            text = (
                f"{class_name} "
                f"{int(round(box_w))}x{int(round(box_h))}px"
            )

            (text_w, text_h), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                1
            )

            text_y = max(
                y1 - 5,
                text_h + 5
            )

            # label background
            cv2.rectangle(
                image,
                (x1, text_y - text_h - 4),
                (x1 + text_w + 4, text_y + 2),
                color,
                -1
            )

            # label
            cv2.putText(
                image,
                text,
                (x1 + 2, text_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        output_path = output_dir / Path(filename).name

        cv2.imwrite(
            str(output_path),
            image
        )

        print(
            f"[OK] {Path(filename).name}: "
            f"{len(annotations)} boxes"
        )

    print()
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Root dataset directory"
    )

    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="train",
        help="Dataset split to visualize"
    )

    args = parser.parse_args()

    draw_coco_boxes(
        args.dataset_dir,
        args.split
    )


if __name__ == "__main__":
    main()
