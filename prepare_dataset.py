"""Готує одно-класовий датасет `bird` у форматі YOLO з COCO 2017.

Завантажує через FiftyOne лише зображення, що містять птахів (без повних 19 ГБ COCO),
залишає лише клас bird, ділить на train/val (80/20, фіксований seed) і експортує у YOLO.
"""
import argparse
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="datasets/bird", help="куди експортувати датасет")
    p.add_argument("--max-samples", type=int, default=3000, help="скільки зображень узяти")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ds = foz.load_zoo_dataset(
        "coco-2017",
        splits=["train", "validation"],
        label_types=["detections"],
        classes=["bird"],
        max_samples=args.max_samples,
        shuffle=True,
        seed=args.seed,
        dataset_name="coco-bird",
        drop_existing_dataset=True,
    )

    # лишаємо тільки bird, без crowd-розмітки
    view = ds.filter_labels(
        "ground_truth", (F("label") == "bird") & (F("iscrowd") != 1), only_matches=True
    )
    view = view.shuffle(seed=args.seed)

    total = len(view)
    n_val = int(total * args.val_ratio)
    splits = {"val": view.limit(n_val), "train": view.skip(n_val)}

    out = Path(args.out).resolve()
    for name, sv in splits.items():
        sv.export(
            export_dir=str(out),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            classes=["bird"],
            split=name,
        )
        n_boxes = sum(len(s.ground_truth.detections) for s in sv)
        print(f"{name}: {len(sv)} зображень, {n_boxes} боксів")

    print(f"Готово: {out / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
