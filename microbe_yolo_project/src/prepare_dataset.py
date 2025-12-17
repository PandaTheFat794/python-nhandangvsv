"""
Prepare the Makrai et al. (Scientific Data 2023) bacterial colony dataset for Ultralytics YOLO training.

Expected inputs (raw_dir):
- Many *.jpg images (e.g., sp21_img04.jpg)
- annot_YOLO.zip (YOLO-format bounding boxes)
Optionally:
- images.xls (metadata)

This script:
1) Unzips annot_YOLO.zip
2) Locates label *.txt files
3) Ensures YOLO class indices are 0-based (shifts if needed)
4) Creates a train/val/test split stratified by species id inferred from filename (spXX)
5) Writes Ultralytics dataset structure:
   out_dir/
     images/{train,val,test}/*.jpg
     labels/{train,val,test}/*.txt
     data.yaml
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# From Table 1 in the Nature Scientific Data paper (IDs sp01..sp24)
DEFAULT_CLASS_NAMES = [
    "Actinobacillus equuli",
    "Actinobacillus pleuropneumoniae",
    "Aeromonas hydrophila",
    "Bacillus cereus",
    "Bibersteinia trehalosi",
    "Bordetella bronchiseptica",
    "Brucella ovis",
    "Clostridium perfringens",
    "Corynebacterium pseudotuberculosis",
    "Erysipelothrix rhusiopathiae",
    "Escherichia coli",
    "Glaesserella parasuis",
    "Klebsiella pneumoniae",
    "Listeria monocytogenes",
    "Paenibacillus larvae",
    "Pasteurella multocida",
    "Proteus mirabilis",
    "Pseudomonas aeruginosa",
    "Rhodococcus equi",
    "Salmonella enterica",
    "Staphylococcus aureus",
    "Staphylococcus hyicus",
    "Streptococcus agalactiae",
    "Trueperella pyogenes",
]


SPECIES_RE = re.compile(r"^(sp\d{2})_img\d+", re.IGNORECASE)


def infer_species_id(filename: str) -> Optional[str]:
    m = SPECIES_RE.match(Path(filename).stem)
    return m.group(1).lower() if m else None


def unzip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


def find_label_files(labels_root: Path) -> List[Path]:
    txts = []
    for p in labels_root.rglob("*.txt"):
        # Exclude common class-name files
        if p.name.lower() in {"classes.txt", "obj.names"}:
            continue
        txts.append(p)
    return sorted(txts)


def find_classes_file(labels_root: Path) -> Optional[Path]:
    for name in ["classes.txt", "obj.names"]:
        p = labels_root / name
        if p.exists():
            return p
    # try deep search
    for p in labels_root.rglob("classes.txt"):
        return p
    for p in labels_root.rglob("obj.names"):
        return p
    return None


def load_class_names(labels_root: Path) -> List[str]:
    classes_file = find_classes_file(labels_root)
    if classes_file and classes_file.exists():
        lines = [ln.strip() for ln in classes_file.read_text(encoding="utf-8", errors="ignore").splitlines()]
        lines = [ln for ln in lines if ln]
        if lines:
            return lines
    return DEFAULT_CLASS_NAMES


def parse_label_line(line: str) -> Tuple[int, float, float, float, float]:
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"Bad label line: {line!r}")
    cls = int(float(parts[0]))
    x, y, w, h = map(float, parts[1:5])
    return cls, x, y, w, h


def scan_class_range(label_files: List[Path]) -> Tuple[int, int]:
    mn, mx = 10**9, -10**9
    for lf in label_files:
        for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            cls, *_ = parse_label_line(line)
            mn = min(mn, cls)
            mx = max(mx, cls)
    return mn, mx


def rewrite_labels_to_out(
    label_files: List[Path],
    image_stems: set,
    out_labels_dir: Path,
    class_shift: int,
) -> Dict[str, Path]:
    """
    Copies/re-writes labels to out_labels_dir with same stem name, applying class_shift.
    Only keeps labels whose stem exists in image_stems.
    Returns mapping stem -> output label path.
    """
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    mapping: Dict[str, Path] = {}

    for lf in tqdm(label_files, desc="Rewriting labels"):
        stem = lf.stem
        if stem not in image_stems:
            continue

        out_path = out_labels_dir / f"{stem}.txt"
        out_lines = []
        for line in lf.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            cls, x, y, w, h = parse_label_line(line)
            cls2 = cls + class_shift
            if cls2 < 0:
                raise ValueError(f"After shift, class < 0 (cls={cls} shift={class_shift}) in {lf}")
            out_lines.append(f"{cls2} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        out_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
        mapping[stem] = out_path

    return mapping


def copy_split(
    image_paths: List[Path],
    labels_by_stem: Dict[str, Path],
    images_out: Path,
    labels_out: Path,
) -> None:
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for img in tqdm(image_paths, desc=f"Copying {images_out.name}"):
        stem = img.stem
        shutil.copy2(img, images_out / img.name)
        # Some images may legitimately have no labels (shouldn't here, but be safe)
        lf = labels_by_stem.get(stem)
        if lf and lf.exists():
            shutil.copy2(lf, labels_out / f"{stem}.txt")
        else:
            (labels_out / f"{stem}.txt").write_text("", encoding="utf-8")


def write_data_yaml(out_dir: Path, class_names: List[str]) -> Path:
    data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(class_names)},
    }
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return yaml_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=str, required=True, help="Directory with images + annot_YOLO.zip")
    ap.add_argument("--out-dir", type=str, default="data/yolo_dataset", help="Output dataset directory")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Images
    image_paths = sorted([p for p in raw_dir.glob("*.jpg")] + [p for p in raw_dir.glob("*.JPG")])
    if not image_paths:
        # maybe images are nested
        image_paths = sorted(list(raw_dir.rglob("*.jpg")) + list(raw_dir.rglob("*.JPG")))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found under {raw_dir}")

    # 2) Unzip labels
    yolo_zip = raw_dir / "annot_YOLO.zip"
    if not yolo_zip.exists():
        # try fuzzy search
        zips = list(raw_dir.rglob("*YOLO*.zip"))
        if zips:
            yolo_zip = zips[0]
        else:
            raise FileNotFoundError(f"annot_YOLO.zip not found in {raw_dir}")

    labels_root = out_dir / "_labels_raw"
    if not labels_root.exists():
        print(f"Unzipping labels: {yolo_zip} -> {labels_root}")
        unzip(yolo_zip, labels_root)

    label_files = find_label_files(labels_root)
    if not label_files:
        raise FileNotFoundError(f"No label .txt files found inside {labels_root}")

    # 3) Class names
    class_names = load_class_names(labels_root)
    nc = len(class_names)

    # 4) Determine class shift to make indices 0..nc-1
    mn, mx = scan_class_range(label_files)
    class_shift = 0
    if mn == 1 and mx == nc:
        class_shift = -1  # 1-based -> 0-based
    elif mn == 0 and mx == nc - 1:
        class_shift = 0
    else:
        # Try to infer: if max equals nc and min==0, maybe there's a stray; otherwise leave as-is and warn
        print(f"[warn] Unexpected class id range: min={mn}, max={mx}, nc={nc}. Keeping shift=0.")

    # 5) Rewrite labels to a clean flat folder
    image_stems = {p.stem for p in image_paths}
    labels_clean_dir = out_dir / "_labels_clean"
    labels_by_stem = rewrite_labels_to_out(label_files, image_stems, labels_clean_dir, class_shift)

    # 6) Split by image-level species id (from filename prefix spXX)
    y = []
    for p in image_paths:
        sid = infer_species_id(p.name) or "unknown"
        y.append(sid)

    test_size = args.test_ratio
    val_size = args.val_ratio / (1.0 - test_size)

    X_trainval, X_test, y_trainval, _ = train_test_split(
        image_paths, y, test_size=test_size, random_state=args.seed, stratify=y
    )
    X_train, X_val, _, _ = train_test_split(
        X_trainval, y_trainval, test_size=val_size, random_state=args.seed, stratify=y_trainval
    )

    # 7) Copy into Ultralytics folder structure
    for split_name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        copy_split(
            X,
            labels_by_stem=labels_by_stem,
            images_out=out_dir / "images" / split_name,
            labels_out=out_dir / "labels" / split_name,
        )

    yaml_path = write_data_yaml(out_dir, class_names)
    print(f"✅ Dataset ready: {out_dir.resolve()}")
    print(f"✅ data.yaml: {yaml_path.resolve()}")
    print(f"Splits: train={len(X_train)} val={len(X_val)} test={len(X_test)}")


if __name__ == "__main__":
    main()
