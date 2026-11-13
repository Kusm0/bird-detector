"""Донавчання YOLO26n для одного класу `bird`."""
import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="datasets/bird/dataset.yaml")
    p.add_argument("--model", default="yolo26n.pt", help="попередньо навчені ваги")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None, help="напр. 0, cpu, mps; за замовчуванням авто")
    p.add_argument("--workers", type=int, default=8, help="потоки завантаження даних")
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="bird")
    p.add_argument("--weights-dir", default="weights", help="куди копіювати best.pt і metrics.json")
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        seed=0,
        patience=15,
    )

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    out_dir = Path(args.weights_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out_dir / "best.pt")
    print(f"Збережено {out_dir / 'best.pt'}")

    # фінальна оцінка найкращих ваг на validation
    m = YOLO(str(out_dir / "best.pt")).val(data=args.data, imgsz=args.imgsz, device=args.device)
    metrics = {
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "mAP50": float(m.box.map50),
        "mAP50-95": float(m.box.map),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(metrics)


if __name__ == "__main__":
    main()
