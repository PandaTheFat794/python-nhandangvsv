"""
Train a multi-class YOLO detector (detection + classification per object) with Ultralytics.

Example:
  python -m src.train_yolo --data data/yolo_dataset/data.yaml --model yolov8n.pt --epochs 100 --imgsz 1024 --batch 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="Model checkpoint (e.g., yolov8n.pt)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", type=str, default="0", help="0,1,... or 'cpu'")
    ap.add_argument("--project", type=str, default="runs/detect")
    ap.add_argument("--name", type=str, default="bacterial_colony_24cls")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        workers=args.workers,
        seed=args.seed,
    )

    # Ultralytics saves weights under: {project}/{name}/weights/best.pt
    out_dir = Path(args.project) / args.name
    print(f"✅ Training finished. Outputs: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
