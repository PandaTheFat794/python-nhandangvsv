"""
Evaluate a trained YOLO model.

Example:
  python -m src.evaluate --weights runs/detect/bacterial_colony_24cls/weights/best.pt --data data/yolo_dataset/data.yaml --split test
"""

from __future__ import annotations

import argparse
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    ap.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", type=str, default="0")
    args = ap.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split, imgsz=args.imgsz, device=args.device)
    print(metrics)


if __name__ == "__main__":
    main()
