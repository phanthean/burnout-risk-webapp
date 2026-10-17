"""Nạp dữ liệu + mô hình một lần, rồi tra cứu và dự đoán theo Person_ID."""
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src import config as C
from src.train import split_ids


class BurnoutPredictor:
    def __init__(self):
        self.df = pd.read_csv(C.data_path())
        self.df['_split'] = split_ids(self.df)
        self.df = self.df.set_index(C.ID_COL, drop=False)
        self.art = self._load_model()

        self.model = self.art['model']
        self.classes = list(self.art['classes'])
        self.hi, self.lo = self.classes.index('High'), self.classes.index('Low')
        self.threshold = float(self.art['threshold_high'])
        self.tiers = self.art['tier_cutoffs']

        # Hệ số Logistic Regression: so sánh High với Low trên thang đã chuẩn hóa (giống odds ratio ở mục B8)
        lr = self.model.named_steps['m']
        self.coef_hl = lr.coef_[self.hi] - lr.coef_[self.lo]
        self.scaler = self.model.named_steps['pre'].named_transformers_['num'].named_steps['scale']
        self.imputer = self.model.named_steps['pre'].named_transformers_['num'].named_steps['impute']

    def _load_model(self):
        """Nạp joblib; nếu thiếu file hoặc lệch phiên bản scikit-learn thì tự huấn luyện lại."""
        try:
            art = joblib.load(C.MODEL_PATH)
            art['model'].predict_proba(self.df[C.FEATURES].head(2))   # thử dự đoán để chắc chắn tương thích
            return art
        except Exception as e:  # noqa: BLE001
            print(f'[!] Không dùng được {C.MODEL_PATH.name} ({type(e).__name__}): huấn luyện lại...')
            from src.train import main as retrain
            retrain()
            return joblib.load(C.MODEL_PATH)

    # ---------------------------------------------------------------- tra cứu
    @property
    def id_range(self):
        return int(self.df[C.ID_COL].min()), int(self.df[C.ID_COL].max())

    def random_id(self, split: Optional[str] = None, label: Optional[str] = None) -> int:
        pool = self.df
        if split:
            pool = pool[pool['_split'] == split]
        if label:
            pool = pool[pool[C.TARGET] == label]
        return int(pool.sample(1)[C.ID_COL].iloc[0])

    def tier(self, p_high: float) -> str:
        if p_high >= self.tiers['Cao']:
            return 'Cao'
        if p_high >= self.tiers['Trung bình']:
            return 'Trung bình'
        return 'Thấp'

    def predict(self, person_id: int) -> Optional[dict]:
        if person_id not in self.df.index:
            return None
        row = self.df.loc[[person_id]]
        x = row[C.FEATURES]

        proba = self.model.predict_proba(x)[0]
        p_high = float(proba[self.hi])
        argmax = self.classes[int(proba.argmax())]
        if p_high >= self.threshold:
            label = 'High'
        else:
            rest = proba.copy(); rest[self.hi] = -1
            label = self.classes[int(rest.argmax())]

        # Đóng góp từng biến vào log-odds High so với Low, tính từ người "trung bình" (z = 0)
        z = self.scaler.transform(self.imputer.transform(x))[0]
        contrib = self.coef_hl * z

        r = row.iloc[0]
        features = []
        for j, f in enumerate(C.FEATURES):
            raw = r[f]
            missing = bool(pd.isna(raw))
            info = self.art['feature_info'][f]
            name, unit = C.LABELS_VI[f]
            features.append({
                'feature': f, 'name': name, 'unit': unit,
                'value': None if missing else float(raw),
                'used_value': float(self.imputer.statistics_[j]) if missing else float(raw),
                'missing': missing, 'median': info['median'], 'min': info['min'], 'max': info['max'],
                'z': round(float(z[j]), 2), 'contribution': round(float(contrib[j]), 3),
            })
        features.sort(key=lambda d: -abs(d['contribution']))

        actual = r[C.TARGET]
        return {
            'person_id': int(person_id),
            'split': r['_split'],
            'context': {c: (None if pd.isna(r[c]) else (r[c].item() if hasattr(r[c], 'item') else r[c]))
                        for c in C.CONTEXT_COLS},
            'probabilities': {c: round(float(proba[self.classes.index(c)]), 4) for c in C.ORDER},
            'p_high': round(p_high, 4),
            'threshold': self.threshold,
            'prediction': label,
            'prediction_argmax': argmax,
            'at_risk': label == 'High',
            'tier': self.tier(p_high),
            'actual': actual,
            'correct': bool(label == actual),
            'features': features,
        }

    def model_info(self) -> dict:
        lr = self.model.named_steps['m']
        return {
            'model': self.art.get('model_name', 'Logistic Regression'), 'C': lr.C,
            'features': C.FEATURES, 'threshold_high': self.threshold, 'tier_cutoffs': self.tiers,
            'n_people': len(self.df), 'id_range': self.id_range,
            'split_sizes': self.df['_split'].value_counts().to_dict(),
            'odds_ratio_high_vs_low': {f: round(float(np.exp(c)), 2) for f, c in zip(C.FEATURES, self.coef_hl)},
        }
