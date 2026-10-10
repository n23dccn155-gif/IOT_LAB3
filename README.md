# LAB3 – nhận diện trạng thái đeo khẩu trang trên Cloud

LAB3 triển khai toàn bộ luồng: dataset Kaggle + XML → crop và chia tập theo ảnh gốc → MobileNetV2 ba nhãn → FastAPI nạp model `.keras` → giao diện camera gửi ảnh và tọa độ từng khuôn mặt.

LAB3 độc lập với LAB2. Ba nhãn và thứ tự đầu ra cố định:

| Chỉ số | Nhãn | Hiển thị |
|---:|---|---|
| 0 | `without_mask` | Không đeo khẩu trang |
| 1 | `with_mask` | Đeo khẩu trang đúng |
| 2 | `mask_weared_incorrect` | Đeo khẩu trang sai |

## Cài môi trường

Tại thư mục `LAB3`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

Code dùng Keras 3 với backend PyTorch. Model nhận RGB `float32` trong khoảng `[0,255]`; lớp chuẩn hóa nằm bên trong model.

## Dataset và huấn luyện

Tải bộ [Face Mask Detection trên Kaggle](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) được dẫn trong slide LAB3. Giải nén và đặt ảnh/annotation theo cấu trúc dưới đây. Dataset được tải riêng; repo Git chứa model đã train và báo cáo để chạy thử ngay sau khi cài môi trường.

Dataset thô có cấu trúc:

```text
data/raw/images/*.png
data/raw/annotations/*.xml
```

Chuẩn bị crop khuôn mặt và chia train/val/test theo nhóm ảnh gốc:

```powershell
python prepare_dataset.py
```

Kết quả nằm trong `data/processed/`; `summary.json` chứa số mẫu và `manifest.json` lưu ảnh nguồn, nhãn, box, crop và split. Script không ghi đè một thư mục output đã có dữ liệu.

Huấn luyện baseline:

```powershell
python train.py --epochs 20 --batch-size 16
```

Model dùng MobileNetV2 `alpha=0.35`, ImageNet pretrained, input 224×224, class weights và augmentation chỉ trên train. Checkpoint được chọn bằng `val_loss`; test chỉ được đánh giá sau khi chọn model. Artifact nằm trong `outputs/run1/`:

```text
mask_model.keras
labels.txt
metadata.json
classification_report.txt
classification_report.json
confusion_matrix.csv
training_log.csv
history.json
training_curves.png
```

Xem macro-F1 và precision/recall/F1 từng nhãn, đặc biệt `mask_weared_incorrect`; accuracy tổng không đủ để đánh giá dữ liệu lệch lớp. Thử model với một crop hoặc thư mục crop:

```powershell
python predict.py data\processed\test\mask_weared_incorrect
```

Fine-tune tùy chọn sang output mới:

```powershell
python train.py --epochs 20 --fine-tune-epochs 5 --out outputs/run2
```

## API contract

`POST /api/v1/predict` nhận `multipart/form-data`:

- `image`: JPEG hoặc PNG chứa ảnh gốc.
- `faces`: chuỗi JSON array; mỗi phần tử là `{ "id", "x", "y", "width", "height" }`, đơn vị pixel trên ảnh gốc.

Response:

```json
{
  "request_id": "uuid",
  "image": { "width": 1280, "height": 720 },
  "predictions": [
    {
      "id": "face-1",
      "label": "with_mask",
      "confidence": 0.97,
      "box": { "x": 20, "y": 30, "width": 100, "height": 120 },
      "inference_ms": 12.3
    }
  ],
  "total_inference_ms": 14.8
}
```

Lỗi dùng envelope FastAPI `{ "detail": ... }`. `GET /health` cho biết service và model đã sẵn sàng hay chưa.

## Chạy thử ứng dụng

Trước tiên cài môi trường theo mục [Cài môi trường](#cài-môi-trường). Mở **hai terminal PowerShell**, chuyển cả hai terminal đến thư mục `LAB3`:

```powershell
cd "D:\N23DCCN155- Dang Van Hiep\HK7\IOT\lab2\LAB3"
```

Ở terminal thứ nhất, khởi động backend (giữ terminal này mở):

```powershell
python -m backend.app --host 127.0.0.1 --port 8000
```

Ở terminal thứ hai, khởi động frontend và proxy API (giữ terminal này mở):

```powershell
python frontend\serve.py --port 5500 --api http://127.0.0.1:8000 --no-browser
```

Mở `http://localhost:5500` trong trình duyệt. Frontend proxy `/api/...` và `/health` sang FastAPI cùng origin; mặc định ô địa chỉ API để trống để dùng proxy này. Nếu trước đây đã lưu địa chỉ API trong trình duyệt, xóa ô đó rồi bấm lưu cấu hình. `localhost` cũng là secure context cho quyền camera.

Các bước thử nhận diện nhiều người:

1. Cho phép trình duyệt truy cập camera.
2. Chọn detector tự động. `FaceDetector` chỉ có trên một số trình duyệt; nếu không được hỗ trợ, chọn `MediaPipe` (cần Internet để tải thư viện từ CDN).
3. Đặt nhiều khuôn mặt rõ ràng trong khung hình. Mỗi request gửi một frame cùng toàn bộ box; mặc định frontend xử lý tối đa 20 mặt, backend tối đa 64 mặt.
4. Kiểm tra box và nhãn từng khuôn mặt: xanh là `with_mask` (đeo đúng), đỏ là `without_mask` (không đeo), vàng là `mask_weared_incorrect` (đeo sai). Nếu detector không phát hiện đủ mặt hoặc không tải được, có thể vẽ nhiều box thủ công trên khung hình/ảnh tĩnh.

Giao diện hiển thị confidence, thời gian suy luận server, thời gian gửi–nhận và tổng thời gian. Kết quả phân loại phụ thuộc chất lượng model và kích thước/độ rõ của khuôn mặt trong ảnh. Có thể kiểm tra backend sẵn sàng bằng cách mở `http://127.0.0.1:8000/health`.

Các biến môi trường backend chính:

```text
MASK_API_MODEL_PATH       đường dẫn model, mặc định outputs/run1/mask_model.keras
MASK_API_LOAD_AT_STARTUP  nạp model lúc startup
MASK_API_REQUIRE_MODEL    dừng startup nếu thiếu model
MASK_API_HOST / MASK_API_PORT / MASK_API_WORKERS
```

## Kiểm thử

Tại thư mục `LAB3`, chạy bộ test backend hiện có:

```powershell
python -m pytest backend\tests -q --basetemp backend\.pytest-local
```

Kiểm tra cú pháp frontend và server tĩnh:

```powershell
node --check frontend\config.js
node --check frontend\detectors.js
node --check frontend\api.js
node --check frontend\ui.js
node --check frontend\app.js
python -m py_compile frontend\serve.py
```

Backend import và `/health` vẫn hoạt động khi chưa có model; endpoint predict trả `503` rõ ràng. Sau khi huấn luyện xong, khởi động lại API để nạp checkpoint thật rồi kiểm tra camera toàn trình.
