"""
Export a trained Ultralytics YOLO model to ONNX (or other formats).

Example:
  python -m src.export_model --weights runs/detect/bacterial_colony_24cls/weights/best.pt --format onnx --imgsz 1024
"""

from __future__ import annotations

import argparse
from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--format", type=str, default="onnx", help="onnx, torchscript, openvino, engine, ...")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--half", action="store_true", default=False)
    ap.add_argument("--dynamic", action="store_true", default=False)
    args = ap.parse_args()

    model = YOLO(args.weights)
    model.export(format=args.format, imgsz=args.imgsz, device=args.device, half=args.half, dynamic=args.dynamic)


if __name__ == "__main__":
    main()
