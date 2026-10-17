"""Cấu hình chung của web app — giữ đúng các lựa chọn đã chốt trong notebook Burnout_EDA_ML.ipynb."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent            # .../Final Project/WebApp

# Dataset nằm ở .../Final Project/Dataset (cùng cấp với WebApp và Report)
DATA_CANDIDATES = [
    ROOT.parent / 'Dataset' / 'mental_health_burnout_prediction_dataset.csv',
    ROOT / 'data' / 'mental_health_burnout_prediction_dataset.csv',
]
MODEL_PATH = ROOT / 'models' / 'burnout_model.joblib'

RANDOM_STATE = 42
ID_COL = 'Person_ID'
TARGET = 'Burnout_Risk'
ORDER = ['Low', 'Moderate', 'High']

# Mục B2: các cột bị loại vì rò rỉ hoặc là đầu ra sau đánh giá
DROP = ['Person_ID', 'Burnout_Score', 'Mental_Health_Status', 'AI_Wellness_Recommendation',
        'Stress_Level', 'Chronic_Stress']

# Mục B3: 9 đặc trưng còn lại sau 4 quy tắc chọn biến
FEATURES = ['Work_Hours_Per_Week', 'Sleep_Hours', 'Anxiety_Score', 'Depression_Score', 'Mood_Score',
            'Physical_Activity_Hours', 'Meditation_Minutes', 'Screen_Time_Hours', 'Coffee_Cups_Per_Day']

# Mục B5: Logistic Regression, C = 10 (GridSearchCV)
BEST_C = 10

# Mục B6: ngưỡng cảnh báo chọn trên tập valid (Recall High >= 0,90)
THRESHOLD_HIGH = 0.25
TIER_CUTOFFS = {'Cao': 0.25, 'Trung bình': 0.10}

# Tên hiển thị tiếng Việt cho giao diện
LABELS_VI = {
    'Work_Hours_Per_Week': ('Giờ làm / tuần', 'giờ'),
    'Sleep_Hours': ('Giờ ngủ / ngày', 'giờ'),
    'Anxiety_Score': ('Điểm lo âu', '/100'),
    'Depression_Score': ('Điểm trầm cảm', '/100'),
    'Mood_Score': ('Điểm tâm trạng', '/100'),
    'Physical_Activity_Hours': ('Vận động / tuần', 'giờ'),
    'Meditation_Minutes': ('Thiền / ngày', 'phút'),
    'Screen_Time_Hours': ('Thời gian màn hình', 'giờ'),
    'Coffee_Cups_Per_Day': ('Cà phê / ngày', 'cốc'),
}
CONTEXT_COLS = ['Age', 'Gender', 'Country', 'Occupation', 'Employment_Status']


def data_path() -> Path:
    for p in DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError('Không tìm thấy dataset. Đặt file csv vào: ' + ' hoặc '.join(map(str, DATA_CANDIDATES)))
