# Nhận dạng & phân loại khuẩn lạc vi sinh vật trên đĩa thạch bằng YOLO (Ultralytics)

Repo này cung cấp pipeline **YOLO object detection + classification theo lớp (multi-class)** cho bài toán:
- **Nhận dạng (detect)** vị trí khuẩn lạc (bounding box)
- **Phân loại (classify)** khuẩn lạc theo **24 loài vi khuẩn** (class id trong annotation)

Dataset tham khảo: *Annotated dataset for deep-learning-based bacterial colony detection* (Makrai et al., Scientific Data 2023).
- 369 ảnh, 24 loài, 56,865 khuẩn lạc (bbox + label)
- Có sẵn annotation dạng **YOLO** trong file `annot_YOLO.zip`.

> Lưu ý: Figshare đôi khi chặn truy cập qua web UI. Vì vậy repo có script tải bằng **Figshare REST API**.

---

## 1) Cấu trúc repo

```
microbe_yolo_project/
  src/
    download_figshare.py
    prepare_dataset.py
    train_yolo.py
    evaluate.py
    predict.py
    export_model.py
  requirements.txt
  README.md
```

---

## 2) Cài đặt (Local)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## 3) Tải dữ liệu từ Figshare (tùy chọn)

### 3.1 Tải annotation + (toàn bộ) ảnh JPG
```bash
python -m src.download_figshare   --article-id 22022540   --out-dir data/raw   --include "annot_YOLO\.zip" ".*\.jpg$" "images\.xls"
```

> Nếu bạn muốn thử nhanh trước: thêm `--max-files 50` để tải 50 ảnh đầu.

---

## 4) Chuẩn hoá dataset sang format Ultralytics YOLO

```bash
python -m src.prepare_dataset   --raw-dir data/raw   --out-dir data/yolo_dataset   --val-ratio 0.15   --test-ratio 0.15
```

Kết quả tạo ra:
- `data/yolo_dataset/data.yaml`
- `data/yolo_dataset/images/{train,val,test}/...`
- `data/yolo_dataset/labels/{train,val,test}/...`

---

## 5) Train YOLO (Detect + 24-class)

```bash
python -m src.train_yolo   --data data/yolo_dataset/data.yaml   --model yolov8n.pt   --epochs 100   --imgsz 1024   --batch 4
```

Weights:
- `runs/detect/bacterial_colony_24cls/weights/best.pt`

---

## 6) Đánh giá

```bash
python -m src.evaluate   --weights runs/detect/bacterial_colony_24cls/weights/best.pt   --data data/yolo_dataset/data.yaml   --split test   --imgsz 1024
```

---

## 7) Inference

```bash
python -m src.predict   --weights runs/detect/bacterial_colony_24cls/weights/best.pt   --source data/yolo_dataset/images/test   --conf 0.25   --imgsz 1024
```

---

## 8) Export model (ONNX)

```bash
python -m src.export_model   --weights runs/detect/bacterial_colony_24cls/weights/best.pt   --format onnx   --imgsz 1024
```

---

## 9) Chạy trên Kaggle

- Tạo Kaggle Notebook, bật **GPU**.
- Bật **Internet** nếu muốn tải dữ liệu từ Figshare qua API.
- Hoặc: upload dataset vào Kaggle Datasets rồi đổi `--raw-dir` sang `/kaggle/input/<ten-dataset>/...`.

---

## Gợi ý cải thiện (nếu muốn làm tốt hơn)

- Ảnh đĩa thạch có rất nhiều vật thể nhỏ ⇒ cân nhắc:
  - tăng `imgsz` (1024–1280)
  - hoặc **chia patch/tiling** (cắt ảnh lớn thành nhiều ô nhỏ) để tăng độ phân giải biểu kiến.
- Class imbalance: một số loài có nhiều colony hơn; dùng stratified split theo `spXX` đã giúp ổn định.

---

## Citation

Nếu dùng dataset/ý tưởng trong báo cáo, hãy trích dẫn bài Scientific Data 2023 của Makrai et al.
