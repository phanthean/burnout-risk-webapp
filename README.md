# Burnout Risk Lookup: web app demo cho Final Project APM17

Nhập **Person_ID** của một người trong bộ dữ liệu, web app sẽ trả về:

- **Có / không** thuộc nhóm nguy cơ burnout cao, kèm xác suất của 3 mức Low / Moderate / High.
- **Tầng ưu tiên** (Cao / Trung bình / Thấp) theo ngưỡng P(High) đã chọn trong notebook.
- **Đối chiếu** với nhãn thực tế, và cho biết người đó thuộc tập train, valid hay test.
- **Yếu tố đẩy về High**: đóng góp của 9 đặc trưng, tính từ hệ số Logistic Regression.


## Cấu trúc

```
Final Project/
├── Dataset/mental_health_burnout_prediction_dataset.csv   <- web app đọc dữ liệu ở đây
├── Report/Code/Burnout_EDA_ML.ipynb                        <- nơi xây dựng mô hình
└── WebApp/
    ├── main.py              # điểm khởi chạy
    ├── run_app.bat          # nhấp đúp để chạy trên Windows
    ├── requirements.txt
    ├── models/burnout_model.joblib
    ├── src/
    │   ├── config.py        # đặc trưng, ngưỡng, cột loại bỏ: giống notebook
    │   ├── train.py         # huấn luyện lại mô hình (mục B2–B6)
    │   ├── predictor.py     # tra cứu theo ID + dự đoán + giải thích
    │   └── api.py           # FastAPI
    └── static/index.html    # giao diện
```

## Chạy

**Cách nhanh (Windows):** nhấp đúp `run_app.bat`. Trình duyệt sẽ mở `http://localhost:8000`.

**Bằng lệnh:**

```bash
cd "D:\DA\Kỳ 3\Project\Final Project\WebApp"
pip install -r requirements.txt
python main.py                 # chạy web app -> http://localhost:8000
python main.py --mode train    # huấn luyện lại mô hình từ dataset
python main.py --mode all      # huấn luyện lại rồi chạy web app
```

Nếu file `burnout_model.joblib` không nạp được (thường do khác phiên bản scikit-learn), app **tự huấn luyện lại** khi khởi động, mất vài giây. Kết quả giống hệt mô hình trong notebook vì dùng cùng phép chia và cùng tham số. Bản đã kiểm tra: Macro F1 = 0,781 và Recall High = 0,891 trên tập test.

## API

| Endpoint | Mô tả |
|---|---|
| `GET /` | Giao diện tra cứu |
| `GET /predict/{person_id}` | Kết quả dự đoán dạng JSON |
| `GET /random-id?split=test&label=High` | Lấy ngẫu nhiên một ID (lọc theo tập và nhãn thực tế) |
| `GET /model-info` | Thông tin mô hình, ngưỡng, odds ratio |
| `GET /health` | Kiểm tra trạng thái |
| `GET /docs` | Tài liệu API tự sinh (Swagger) |

Ví dụ `GET /predict/28`:

```json
{"person_id": 28, "split": "test", "p_high": 0.5412, "prediction": "High",
 "at_risk": true, "tier": "Cao", "actual": "High", "correct": true, ...}
```

## Gợi ý khi demo

1. Bấm **Thực tế High** (lấy từ tập test): mô hình thường cảnh báo đúng, và phần giải thích cho thấy điểm lo âu, trầm cảm cao.
2. Bấm **Thực tế Moderate** vài lần: sẽ gặp ca bị cảnh báo nhầm thành High. Đây là cái giá của ngưỡng 0,25 (ưu tiên không bỏ sót).
3. Nhập một ID thuộc tập **train** để giải thích vì sao nên demo bằng tập test.

## Giới hạn

Dữ liệu mô phỏng. Mô hình dựa vào điểm khảo sát tâm lý và thói quen, không phải công cụ chẩn đoán y khoa.
