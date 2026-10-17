"""FastAPI: tra cứu nguy cơ burnout theo Person_ID."""
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src import config as C
from src.predictor import BurnoutPredictor

app = FastAPI(title='Burnout Risk Lookup', version='1.0',
              description='Final Project APM17: dự báo nguy cơ burnout theo Person_ID (Logistic Regression, 9 đặc trưng).')
app.mount('/static', StaticFiles(directory=str(C.ROOT / 'static')), name='static')

predictor: Optional[BurnoutPredictor] = None


@app.on_event('startup')
def startup():
    global predictor
    t = time.time()
    predictor = BurnoutPredictor()
    print(f'Đã nạp {len(predictor.df):,} người và mô hình trong {time.time() - t:.1f}s')


@app.get('/', include_in_schema=False)
def index():
    return FileResponse(str(C.ROOT / 'static' / 'index.html'))


@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': predictor is not None}


@app.get('/model-info')
def model_info():
    return predictor.model_info()


@app.get('/predict/{person_id}')
def predict(person_id: int):
    t = time.time()
    lo, hi = predictor.id_range
    result = predictor.predict(person_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f'Không có Person_ID = {person_id}. ID hợp lệ: {lo}–{hi}.')
    result['latency_ms'] = round((time.time() - t) * 1000, 1)
    return result


@app.get('/random-id')
def random_id(split: Optional[str] = Query(None, pattern='^(train|valid|test)$'),
              label: Optional[str] = Query(None, pattern='^(Low|Moderate|High)$')):
    return {'person_id': predictor.random_id(split, label)}
