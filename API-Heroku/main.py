import pandas as pd
import uvicorn
import joblib 
import json
from sklearn.ensemble import RandomForestClassifier
import xgboost 
from typing import List
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pydantic import annotated_types

# Creating FastAPI 
app = FastAPI()

templates = Jinja2Templates(directory="templates")

class rf(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    DebtRatio: float 
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int 
    NumberRealEstateLoansOrLines: int
    NumberOfDependents: float


@app.get('/')
async def root():
    """
    Verifies API is deployed, links to docs
    """ 
    return HTMLResponse("""
    <h1>XGBoost model API</h1>
    <p>Go to <a href="/docs">/docs</a> for documentation. </p>
    """)
@app.post('/predict')
async def model_predict(data: List[rf]):
    df = pd.DataFrame([dict(item) for item in data])
    model = joblib.load('model.pkl')
    prediction = model.predict_proba(df)
    print(prediction)

    prediction_json = json.dumps(prediction.tolist())

    return Response(prediction_json)

if __name__ == '__main__': 
        uvicorn.run('main:app', host='127.0.0.1', port=8006, reload=True)

