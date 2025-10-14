from fastapi import FastAPI
from catboost import CatBoostClassifier
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

class LoanRequest(BaseModel):
    Month:int
    Name:str=None
    Age:int
    SSN:str=None
    Occupation:str=None
    Annual_Income:float
    Num_Bank_Accounts:int=None
    Num_Credit_Card:int=None
    Interest_Rate:float=None
    Num_of_Loan:int=None
    Type_of_Loan:str=None
    Delay_from_due_date:int=None
    Num_of_Delayed_Payment:int=None
    Changed_Credit_Limit:float=None
    Num_Credit_Inquiries:int=None
    Credit_Mix:str=None
    Outstanding_Debt:float=None
    Credit_Utilization_Ratio:float=None
    Payment_of_Min_Amount:str=None
    Total_EMI_per_month:float=None
    Amount_invested_monthly:float=None
    Payment_Behaviour:str=None
    Monthly_Balance:float=None
    Credit_History_Years:int=None
    Credit_History_Months:int=None

model = CatBoostClassifier()
model.load_model("Credit-Scoring/app/credit_model.cbm")

@app.post("/predict")
def predict(data: LoanRequest):
    df = pd.DataFrame([data.model_dump()])
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].fillna('missing').astype(str)
    for c in df.select_dtypes(include=['float64','int64']).columns:
        df[c] = df[c].fillna(0)
    pred = model.predict(df)[0]
    return {"prediction": pred, "message": f"Ваша заявка: {pred}"}