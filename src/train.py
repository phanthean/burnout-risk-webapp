"""Huấn luyện lại mô hình cuối (giống mục B2–B6 của notebook) và lưu vào models/burnout_model.joblib.

Chạy lại bước này khi máy có phiên bản scikit-learn khác với máy đã xuất mô hình,
vì file joblib chỉ nạp ổn định trên cùng phiên bản thư viện.
"""
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C


def split_ids(df: pd.DataFrame) -> pd.Series:
    """Tái tạo đúng phép chia 70/20/10 của notebook, trả về nhãn tập cho từng dòng."""
    X, y = df.drop(columns=C.DROP + [C.TARGET]), df[C.TARGET]
    X_tmp, X_test, y_tmp, _ = train_test_split(X, y, test_size=0.10, stratify=y, random_state=C.RANDOM_STATE)
    X_train, X_valid, _, _ = train_test_split(X_tmp, y_tmp, test_size=2 / 9, stratify=y_tmp,
                                              random_state=C.RANDOM_STATE)
    split = pd.Series('train', index=df.index)
    split[X_valid.index] = 'valid'
    split[X_test.index] = 'test'
    return split


def build_model() -> Pipeline:
    pre = ColumnTransformer([('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                                               ('scale', StandardScaler())]), C.FEATURES)])
    return Pipeline([('pre', pre), ('m', LogisticRegression(C=C.BEST_C, max_iter=2000))])


def predict_with_threshold(proba, classes, threshold):
    hi = classes.index('High')
    rest = proba.copy()
    rest[:, hi] = -1
    return np.where(proba[:, hi] >= threshold, 'High', np.array(classes)[rest.argmax(axis=1)])


def main():
    df = pd.read_csv(C.data_path())
    split = split_ids(df)
    tr, te = df[split == 'train'], df[split == 'test']

    model = build_model().fit(tr[C.FEATURES], tr[C.TARGET])
    classes = list(model.classes_)

    proba = model.predict_proba(te[C.FEATURES])
    pred = predict_with_threshold(proba, classes, C.THRESHOLD_HIGH)
    metrics = {
        'Macro F1': round(f1_score(te[C.TARGET], pred, average='macro'), 3),
        'Recall High': round(recall_score(te[C.TARGET], pred, labels=['High'], average='macro'), 3),
    }
    info = tr[C.FEATURES].describe().T[['min', '50%', 'max', 'mean']].rename(columns={'50%': 'median'}).round(2)

    artifact = {
        'model': model, 'model_name': 'Logistic Regression', 'classes': classes, 'features': C.FEATURES,
        'feature_info': info.to_dict('index'), 'threshold_high': C.THRESHOLD_HIGH,
        'tier_cutoffs': C.TIER_CUTOFFS, 'test_metrics_threshold': metrics,
        'sklearn_version': sklearn.__version__,
    }
    C.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, C.MODEL_PATH)
    print(f'Đã lưu {C.MODEL_PATH} (scikit-learn {sklearn.__version__})')
    print(f'Kiểm tra trên tập test, ngưỡng {C.THRESHOLD_HIGH}: {metrics}')


if __name__ == '__main__':
    main()
