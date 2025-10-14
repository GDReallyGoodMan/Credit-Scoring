from fastapi import FastAPI
from catboost import CatBoostClassifier
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           
    allow_credentials=True,
    allow_methods=["*"],            
    allow_headers=["*"],
)

FEATURE_ORDER = [
 'Month','Name','SSN','Occupation','Type_of_Loan','Credit_Mix',
 'Payment_of_Min_Amount','Payment_Behaviour',
 'Age','Annual_Income','Num_Bank_Accounts','Num_Credit_Card',
 'Interest_Rate','Num_of_Loan','Delay_from_due_date',
 'Num_of_Delayed_Payment','Changed_Credit_Limit','Num_Credit_Inquiries',
 'Outstanding_Debt','Credit_Utilization_Ratio','Total_EMI_per_month',
 'Amount_invested_monthly','Monthly_Balance','Credit_History_Years',
 'Credit_History_Months'
]


CAT_FEATURES = [
 'Month','Name','SSN','Occupation','Type_of_Loan',
 'Credit_Mix','Payment_of_Min_Amount','Payment_Behaviour'
]


class LoanRequest(BaseModel):
    Month: str | int = 0
    Name: str | None = None
    SSN: str | None = None
    Occupation: str | None = None
    Type_of_Loan: str | None = None
    Credit_Mix: str | None = None
    Payment_of_Min_Amount: str | None = None
    Payment_Behaviour: str | None = None
    Age: int = 0
    Annual_Income: float = 0.0
    Num_Bank_Accounts: int = 0
    Num_Credit_Card: int = 0
    Interest_Rate: float | None = None
    Num_of_Loan: int = 0
    Delay_from_due_date: int = 0
    Num_of_Delayed_Payment: int = 0
    Changed_Credit_Limit: float | None = None
    Num_Credit_Inquiries: int = 0
    Outstanding_Debt: float | None = None
    Credit_Utilization_Ratio: float | None = None
    Total_EMI_per_month: float | None = None
    Amount_invested_monthly: float | None = None
    Monthly_Balance: float | None = None
    Credit_History_Years: int = 0
    Credit_History_Months: int = 0

model = CatBoostClassifier()
model.load_model("/Users/denisgusin/Desktop/code/Projects/Credit Scoring/Credit-Scoring/backend/credit_model.cbm")

@app.post("/predict")
def predict(data: LoanRequest):
    payload = data.model_dump() if hasattr(data, "model_dump") else data.dict()
    df = pd.DataFrame([payload])

    for c in FEATURE_ORDER:
        if c not in df.columns:
            df[c] = None
    df = df[FEATURE_ORDER]

    for c in CAT_FEATURES:
        df[c] = df[c].where(df[c].notna(), "missing").astype(str)
    
    for c in df.select_dtypes(include=['object']).columns:
        if c not in CAT_FEATURES:
            df[c] = df[c].where(df[c].notna(), "missing").astype(str)
    
    for c in df.columns:
        if c not in CAT_FEATURES:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    pred = model.predict(df)[0]
    pred = str(pred) 

    if pred == "Poor":
        message = "К сожалению, мы не можем выдать вам кредит."
    elif pred == "Standard":
        message = "Ваша заявка на рассмотрении."
    else:
        message = "Ваша заявка по кредиту одобрена. Пожалуйста, свяжитесь с банком."
    
    return {"prediction": pred, "message": message}

