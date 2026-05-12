# Vietnamese Handwriting OCR Pipeline

Dự án nhận diện chữ viết tay tiếng Việt với pipeline 4 giai đoạn:

```
PaddleOCR  ->  VietOCR  ->  PhoBERT / GLiNER  ->  ViT5
(detect)      (recognize)   (NER / NLU)          (correct / generate)
```

Repo này hiện đang ở **Phase 1**: ghép PaddleOCR (chỉ detection) với VietOCR (recognition).
Các phase sau (NER và sửa lỗi chính tả) sẽ được bổ sung dần.

---

## Cấu trúc thư mục

```
handw/
  configs/
    default.yaml                # cấu hình inference pipeline
    train_vietocr.yaml          # cấu hình fine-tune VietOCR
  data/
    samples/                    # ảnh mẫu để test
    outputs/                    # JSON + ảnh visualize (gitignored)
    data_line/                  # InkData (training set, ~7.3k ảnh)
  models/
    vietocr/                    # checkpoint fine-tuned (gitignored)
  src/
    detector.py                 # PaddleOCR wrapper (det-only)
    recognizer.py               # VietOCR wrapper (hỗ trợ weights_path)
    pipeline.py                 # ghép detector + recognizer
    cli.py                      # CLI entry point
    train/
      prepare_data.py           # split train/val + check vocab
      train_vietocr.py          # driver gọi VietOCR Trainer
      evaluate.py               # CER/WER/EM trên test set
    utils/
      image.py                  # warp, sort reading order
      viz.py                    # vẽ bbox + chữ Việt
  scripts/
    make_sample_image.py        # sinh ảnh demo
    smoke_train.py              # dry-run fine-tune ngay tại máy
    inspect_data.py             # check format dataset
  notebooks/
    01_paddle_vietocr_demo.ipynb   # Phase 1 inference demo
    02_finetune_vietocr_colab.ipynb # Phase 1.5 fine-tune (Colab)
  app/
    streamlit_app.py            # web UI
  tests/
    test_pipeline.py
  requirements.txt
```

---

## Cài đặt

### Yêu cầu

- Python **3.8 – 3.10** (PaddleOCR chưa tương thích tốt với 3.11+ trên Windows)
- Windows / Linux / macOS, hoặc Google Colab

### Local (Windows)

```powershell
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Local (Linux/macOS)

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Google Colab (GPU)

```python
!pip install -q paddlepaddle-gpu==2.6.1 paddleocr==2.7.3 vietocr==0.3.13
!pip install -q "numpy<2.0" "opencv-python==4.10.0.84"
```

> Lần đầu chạy, VietOCR và PaddleOCR sẽ tự tải weights (~150–300 MB).

---

## Cách dùng

### 1. CLI

```bash
python -m src.cli --image data/samples/note.jpg --save-viz
python -m src.cli --image data/samples/note.jpg --rec-model vgg_seq2seq --output-json out.json
```

Tham số:
- `--image` (bắt buộc): đường dẫn ảnh đầu vào
- `--config`: file YAML cấu hình (mặc định `configs/default.yaml`)
- `--rec-model`: `vgg_transformer` (chính xác) hoặc `vgg_seq2seq` (nhanh)
- `--device`: `auto`, `cpu`, `cuda:0`...
- `--save-viz`: lưu ảnh có bbox vào `data/outputs/`
- `--output-json`: ghi kết quả JSON ra file

### 2. Streamlit web app

```bash
streamlit run app/streamlit_app.py
```

Mở `http://localhost:8501`, upload ảnh, chọn backbone, xem kết quả.

### 3. Notebook

```bash
jupyter notebook notebooks/01_paddle_vietocr_demo.ipynb
```

Notebook chạy từng bước: detect -> visualize bbox -> crop -> recognize -> kết quả.

### 4. Dùng như thư viện Python

```python
from src.pipeline import HandwritingOCRPipeline

pipe = HandwritingOCRPipeline.from_config("configs/default.yaml")
result = pipe.run("data/samples/note.jpg")

print(result["full_text"])
for item in result["items"]:
    print(item["bbox"], item["text"], item["confidence"])
```

---

## Lưu ý quan trọng

- VietOCR pretrained được train chủ yếu trên **chữ in**. Trên chữ viết tay
  thật, độ chính xác sẽ thấp. Phase tiếp theo sẽ **fine-tune** trên dataset
  chữ viết tay tiếng Việt công khai (VNOnDB, Cinnamon AI HWR...).
- PaddleOCR cần `numpy < 2.0`.
- Trên Windows nếu thiếu font Unicode khi vẽ kết quả, hệ thống sẽ fallback
  sang `DejaVuSans` hoặc font mặc định của PIL.

---

## Phase 1.5 — Fine-tune VietOCR trên InkData

Dataset `data/data_line/` (InkData line-level + 15 ảnh địa chỉ bổ sung):

- `train_line_annotation.txt` — 5,498 dòng (path + label, separator TAB hoặc 2+ spaces)
- `test_line_annotation.txt`  — 1,813 dòng (giữ riêng cho đánh giá cuối)
- `InkData_line_processed/`   — 7,311 ảnh (.png/.jpg)

### Bước 1: Split train/val 90/10 và check vocab

```powershell
python -m src.train.prepare_data --config configs/train_vietocr.yaml
```

Tạo `train_split.txt` (~4948 dòng) và `val_split.txt` (~550 dòng). Cảnh báo nếu có ký tự ngoài vocab VietOCR mặc định.

### Bước 2: Smoke test local (tuỳ chọn, không cần GPU)

```powershell
python scripts/smoke_train.py --iters 50 --batch-size 4 --device cpu
```

Subsample 100 train / 20 val, chạy 50 iter để kiểm tra wiring trước khi đốt thời gian GPU.

### Bước 3: Fine-tune trên Colab GPU (chính)

1. Nén thư mục `data/data_line/` và upload lên Google Drive tại `MyDrive/handw/data_line/`
2. Nén toàn bộ repo thành `handw.zip` và upload tại `MyDrive/handw/handw.zip`
3. Mở `notebooks/02_finetune_vietocr_colab.ipynb` trên Colab GPU (T4 đủ dùng)
4. Run all → checkpoint xuất ra `MyDrive/handw/models/vietocr/vietocr_seq2seq_inkdata.pth`

Thời gian: ~1–2 giờ trên T4 với mặc định `iters=20000`, `batch_size=32`.

### Bước 4: Đánh giá CER/WER

```powershell
python -m src.train.evaluate `
    --config configs/train_vietocr.yaml `
    --weights models/vietocr/vietocr_seq2seq_inkdata.pth `
    --save-predictions data/outputs/test_predictions.json
```

In ra:
- **CER** — Character Error Rate (Levenshtein-based)
- **WER** — Word Error Rate (whitespace tokens)
- **EM**  — Exact Match
- File JSON kèm dự đoán per-sample để phân tích lỗi

### Bước 5: Dùng checkpoint trong pipeline inference

Sau khi copy `.pth` về local, sửa `configs/default.yaml`:

```yaml
recognizer:
  model: vgg_seq2seq
  weights_path: models/vietocr/vietocr_seq2seq_inkdata.pth
```

Hoặc truyền thẳng qua CLI:

```powershell
python -m src.cli --image data\samples\sample.png --rec-model vgg_seq2seq `
    --weights models\vietocr\vietocr_seq2seq_inkdata.pth --save-viz
```

---

## Roadmap

- [x] **Phase 1**: PaddleOCR (detect) + VietOCR (recognize) + CLI/UI
- [x] **Phase 1.5**: Fine-tune VietOCR trên InkData (scripts + Colab notebook + eval)
- [ ] **Phase 2**: PhoBERT / GLiNER cho NER trên text đã OCR
- [ ] **Phase 3**: ViT5 sửa lỗi chính tả / chuẩn hoá văn bản
- [ ] **Phase 4**: Đánh giá end-to-end + tối ưu / deploy
