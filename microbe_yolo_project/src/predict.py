"""
Run inference and save visualizations.

Example:
  python -m src.predict --weights runs/detect/bacterial_colony_24cls/weights/best.pt --source path/to/images --conf 0.25 --imgsz 1024
"""

from __future__ import annotations

import argparse
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--source", type=str, required=True, help="Image file/dir/video/0(webcam)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--save", action="store_true", default=True)
    ap.add_argument("--project", type=str, default="runs/predict")
    ap.add_argument("--name", type=str, default="inference")
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=args.save,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
