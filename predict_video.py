"""Детекція птахів на відео: малює bbox, клас і confidence, зберігає результат у файл."""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

COLOR = (0, 200, 0)


def draw(frame, box, label):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2)
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    top = max(y1 - th - bl - 4, 0)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 4, top + th + bl + 4), COLOR, -1)
    cv2.putText(frame, label, (x1 + 2, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 2, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="шлях до вхідного відео")
    p.add_argument("--output", required=True, help="шлях до вихідного відео")
    p.add_argument("--weights", default="weights/best.pt")
    p.add_argument("--conf", type=float, default=0.25, help="поріг confidence")
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise SystemExit(f"Не вдалося відкрити відео: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    model = YOLO(args.weights)
    n = detections = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                            device=args.device, verbose=False)[0]
        for box, conf, cls in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(),
                                  res.boxes.cls.tolist()):
            draw(frame, box, f"{res.names[int(cls)]} {conf:.2f}")
            detections += 1
        writer.write(frame)
        n += 1
        if n % 50 == 0:
            print(f"{n}/{total} кадрів", flush=True)

    cap.release()
    writer.release()
    print(f"Готово: {n} кадрів, {detections} детекцій -> {args.output}")


if __name__ == "__main__":
    main()
