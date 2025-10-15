# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import os
import joblib
import numpy as np

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURE_ORDER = [
 'age', 'appl_rej_cnt', 'Score_bki', 'out_request_cnt', 'region_rating',
 'home_address_cd', 'work_address_cd', 'income', 'first_time_cd',
 'education_cd_ACD', 'education_cd_GRD', 'education_cd_NONE',
 'education_cd_PGR', 'education_cd_SCH', 'education_cd_UGR',
 'car_own_flg_N', 'car_own_flg_Y', 'car_type_flg_N', 'car_type_flg_Y',
 'Air_flg_N', 'Air_flg_Y'
]

EDU_OPTIONS = ['ACD','GRD','NONE','PGR','SCH','UGR']
BOOL_FLAGS = {
    'car_own': ('car_own_flg_N', 'car_own_flg_Y'),
    'car_type': ('car_type_flg_N', 'car_type_flg_Y'),
    'air': ('Air_flg_N', 'Air_flg_Y')
}

BEST_THRESH = 0.5

class LoanRequest(BaseModel):
    age: int = 0
    appl_rej_cnt: int = 0
    Score_bki: float = 0.0
    out_request_cnt: int = 0
    region_rating: int = 0
    home_address_cd: str | int | None = None
    work_address_cd: str | int | None = None
    income: float = 0.0
    first_time_cd: str | int | None = None
    education: str | None = None   
    car_own: str | None = None     
    car_type: str | None = None    
    air: str | None = None         


model = joblib.load('/Users/denisgusin/Desktop/code/Projects/Credit Scoring/Credit-Scoring/backend/logreg_model.pkl')

if not hasattr(model, "predict_proba"):
    raise RuntimeError("Loaded model does not have predict_proba method. Expect sklearn-like estimator.")

@app.post("/predict")
def predict(data: LoanRequest):
    payload = data.model_dump() if hasattr(data, "model_dump") else data.dict()
    df = pd.DataFrame([{c: 0 for c in FEATURE_ORDER}])

    direct_numeric = {
        'age': None, 'appl_rej_cnt': None, 'Score_bki': None,
        'out_request_cnt': None, 'region_rating': None,
        'home_address_cd': None, 'work_address_cd': None,
        'income': None, 'first_time_cd': None
    }
    for key in direct_numeric.keys():
        if key in payload and payload[key] is not None:
            if key in ['home_address_cd','work_address_cd','first_time_cd']:
                try:
                    df.loc[0, key] = int(payload[key])
                except Exception:
                    try:
                        df.loc[0, key] = int(str(payload[key]))
                    except Exception:
                        df.loc[0, key] = 0
            else:
                try:
                    df.loc[0, key] = float(payload[key])
                except Exception:
                    df.loc[0, key] = 0.0

    chosen_edu = (payload.get("education") or "").strip()
    for opt in EDU_OPTIONS:
        col = f"education_cd_{opt}"
        df.loc[0, col] = 1 if chosen_edu.upper() == opt else 0

    for key, (colN, colY) in BOOL_FLAGS.items():
        v = (payload.get(key) or "").strip().upper()
        if v == 'Y':
            df.loc[0, colY] = 1
            df.loc[0, colN] = 0
        elif v == 'N':
            df.loc[0, colY] = 0
            df.loc[0, colN] = 1
        else:
            df.loc[0, colY] = 0
            df.loc[0, colN] = 1

    df = df[FEATURE_ORDER]

    try:
        prob = float(model.predict_proba(df)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model predict_proba error: {e}")

    pred = 1 if (prob > BEST_THRESH or df['income'][0] < 30000) else 0

    prob_pct = round(prob * 100, 2)
    if pred == 1:
        message = f"Вероятность дефолта {prob_pct}%. Заявка отклонена."
    else:
        message = f"Вероятность дефолта {prob_pct}%. Заявка одобрена."

    return {"probability": prob, "prediction": int(pred), "message": message, "threshold": BEST_THRESH}
