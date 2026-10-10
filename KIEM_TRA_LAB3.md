# Đối chiếu LAB3 — 10/10/2026

Nguồn yêu cầu: `../3- Phat trien he thong IoT LAB 3-2.pptx`, slide 1–3.
Đối chiếu bổ sung nguyên tắc đo độ trễ ở LAB1, slide 6–7.

**Kết luận: bản local có luồng camera → phát hiện nhiều mặt → API → phân loại/vẽ kết quả, nhưng chưa đủ bằng chứng hoàn thành toàn bộ LAB3.** Các việc chính còn lại: triển khai model/API trên Cloud; sửa phép đo toàn trình; xác minh ba trạng thái xuất hiện và được phân loại đúng trong cùng một khung hình, đặc biệt lớp đeo sai.

## Đối chiếu yêu cầu

| Yêu cầu trong slide | Trạng thái | Bằng chứng / việc còn lại |
|---|---|---|
| Dataset ảnh từ LAB2 hoặc nguồn khác | Có | Dataset Kaggle, ảnh và XML ở `data/raw`; chia tập theo ảnh gốc và hash nội dung. Kiểm tra manifest không thấy hash ảnh xuất hiện ở nhiều split. |
| Ba nhãn: đeo đúng, không đeo, đeo sai; tối thiểu 20 mẫu khác nhau mỗi nhãn | Có | Train: 2.239 / 497 / 86 crop; lần lượt 529 / 194 / 69 hash ảnh nguồn. |
| Huấn luyện model học sâu nhẹ | Có | MobileNetV2 alpha=0.35, input 224×224; checkpoint khoảng 4,2 MB. Đã huấn luyện 9 epoch; fine-tune chưa chạy nhưng slide không bắt buộc fine-tune. |
| Xuất model thành file | Có | `outputs/run1/mask_model.keras`, kèm labels, metadata, báo cáo phân loại, confusion matrix và đường cong train. |
| Lưu model trên Cloud, suy luận trên Cloud | Chưa xác nhận | API đang kiểm tra ở `127.0.0.1:8000`; frontend proxy ở `127.0.0.1:5500`. Model đang nạp từ đĩa laptop. Chưa tìm thấy URL hay cấu hình triển khai Cloud trong LAB3. Bản local chưa chứng minh yêu cầu Cloud. |
| Camera thường hoặc camera IoT | Có với laptop | Camera laptop hợp lệ theo lựa chọn “camera thường/camera IoT” ở slide 1. Không bắt buộc bổ sung ESP32 chỉ để đáp ứng lựa chọn này. |
| Giao diện hiển thị | Có | Camera, box từng mặt, nhãn, confidence, cấu hình, ảnh tĩnh, tình trạng API và các số đo thời gian. Lỗi đổi tọa độ/lật chữ đã sửa ở lượt trước. |
| Nhận dạng cùng lúc ba loại nhãn khác nhau | Hỗ trợ về code, chưa xác minh đạt ổn định | API trả nhiều prediction trong một request. Người dùng xác nhận camera nhận đúng hai mặt. Trong kiểm tra ba mặt thuộc ba trạng thái, lớp đeo sai bị đoán sai; cần demo ba trạng thái cùng khung hình. |
| Cung cấp API | Có | `POST /api/v1/predict`, `GET /health`; backend có OpenAPI tại `/docs`. |
| Thời gian suy luận trên Cloud | Đã đo model trên local | `total_inference_ms` đo thời gian `model.predict`; cần chạy API trên Cloud để lấy số đo đúng môi trường yêu cầu. Đây là thời gian model, không phải toàn bộ decode/crop/xử lý HTTP trên server. |
| Phản hồi local ↔ Cloud, từ gửi ảnh đến nhận kết quả | Đã đo RTT trên local | Browser đo thời gian bắt đầu request đến nhận/đọc JSON. Cần lấy số đo khi API ở Cloud. RTT không phải độ trễ upload một chiều. |
| Độ trễ toàn trình phản ánh thời gian chờ | Chưa đo đủ | `captureMs` kết thúc ngay sau `drawImage`, trước `frameToBlob`; `totalMs = captureMs + RTT` bỏ sót mã hóa JPEG. Cũng chưa lấy mốc kết thúc sau khi cập nhật/vẽ kết quả. |

## Kiểm tra đã chạy

- 165 test backend không cần nạp Keras/checkpoint: qua. Các test này dùng model stub để kiểm tra contract, validation, crop, batching và xử lý lỗi; không chứng minh độ chính xác của model thật.
- Hai bộ regression frontend: tọa độ, lật gương/chữ, DPR, letterboxing, resize và health/model readiness: qua.
- `/health` của API trực tiếp và frontend proxy: `ready=true`, backend Torch, checkpoint `outputs/run1/mask_model.keras`.
- 16 request vào API đang chạy bằng model thật: một request ba trạng thái và 15 request đo thời gian với 1/2/3 mặt. API trả đúng số kết quả, đúng ID, nhãn hợp lệ và confidence trong [0,1].
- Ảnh kiểm tra lấy từ test: `maksssksksss156.png`; box lấy từ manifest/VOC, **không** dùng bộ phát hiện của browser trong phép kiểm tra API này.

| Nhãn thực tế của ba mặt | Kết quả API | Confidence |
|---|---|---:|
| Không đeo | Không đeo | 75,5% |
| Đeo đúng | Đeo đúng | 99,3% |
| Đeo sai | **Đeo đúng — sai** | 63,8% |

Báo cáo test có sẵn trên 599 crop: accuracy 86,48%; macro-F1 63,62%; F1 không đeo 73,28%, đeo đúng 91,76%, đeo sai 25,81%. Recall đeo sai 23,53% trên 17 mẫu test. Model có đủ ba đầu ra, nhưng accuracy tổng che khuất điểm yếu của lớp đeo sai. Slide không quy định ngưỡng accuracy; các số này cho thấy cần cải thiện/xác minh trước khi khẳng định demo ổn định.

| Số mặt/request | RTT trung bình qua proxy local | Suy luận model trung bình |
|---:|---:|---:|
| 1 | 84,78 ms | 58,59 ms |
| 2 | 100,24 ms | 66,36 ms |
| 3 | 117,82 ms | 78,63 ms |

Mỗi hàng chỉ gồm 5 request, dùng cùng ảnh có sẵn, trong lúc server local đang hoạt động. Không gồm chụp/mã hóa ảnh ở browser; không phải số liệu Cloud hay benchmark chuẩn cho mọi điều kiện.

Chi tiết và script tái chạy: [JSON](outputs/audit/local_api_audit.json), [CSV](outputs/audit/local_benchmark.csv), [script](outputs/audit/audit_live.py).

## Các việc nên làm tiếp

1. **Sửa phép đo toàn trình.** Đặt mốc bắt đầu trước lấy/mã hóa frame; kết thúc khi kết quả được hiển thị. Ghi rõ có bao gồm phát hiện khuôn mặt hay không. Giữ RTT và inference riêng; không cộng inference thêm vào RTT vì RTT đã bao gồm chờ server. Code liên quan: `frontend/app.js` phần `sendFrame`/`applyResult`, `frontend/api.js` phần `predict`.
2. **Cải thiện và kiểm tra lớp đeo sai.** Train hiện có 86 crop lớp này so với 2.239 đeo đúng. Bổ sung dữ liệu đeo dưới mũi/dưới cằm phù hợp webcam, giữ test độc lập; thử fine-tune hoặc điều chỉnh huấn luyện và chọn model bằng validation. Không dùng test để lựa chọn checkpoint. Demo ba người/ba trạng thái trong cùng frame; nếu thiếu người có thể kiểm tra thêm bằng ảnh tĩnh ba trạng thái, nhưng cần ghi rõ hình thức demo.
3. **Đưa model và API lên Cloud**, rồi cho frontend gọi API đó và đo lại cả ba thời gian. Cần có bằng chứng URL, model nạp thành công, request thực từ laptop và kết quả nhiều mặt. Kiểm tra đường model và lệnh chạy trước khi deploy: `backend/app/config.py::env_path` đang trả default tương đối theo working directory mặc dù docstring nói luôn theo project root; hiện README yêu cầu chạy trong LAB3 nên bản local vẫn chạy được.
4. **Lưu bằng chứng cho báo cáo:** ảnh/video ba trạng thái, kết quả đánh giá model và bảng thời gian khi chạy Cloud. Xuất log CSV/thống kê nhiều lần là bổ sung hữu ích; slide không quy định bắt buộc CSV, database, Docker hay provider Cloud cụ thể. Log UI hiện chỉ giữ 25 dòng trong bộ nhớ.

Nguyên tắc LAB1 có thể áp dụng: gắn ID cho mỗi lần xử lý, định nghĩa mốc thời gian rõ, đo chênh lệch bằng cùng đồng hồ cho RTT và E2E. `request_id` đã có trong API. Không áp nguyên xi các mốc relay/email/MQTT của LAB1 vào LAB3; muốn đo upload một chiều giữa hai máy phải xử lý đồng bộ đồng hồ, không suy ra trực tiếp từ RTT.

Lượt kiểm tra này chỉ thêm báo cáo và artifact kiểm tra, không đổi model hoặc code ứng dụng và không triển khai lên dịch vụ ngoài.
