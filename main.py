import os
import requests
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Telecom Churn Predictor")

# Get absolute path and join it with the "templates" folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

AZURE_ENDPOINT_URL = os.getenv("AZURE_ENDPOINT_URL")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")

class CustomerData(BaseModel):
    account_weeks: int
    contract_renewal: int
    data_plan: int
    data_usage: float
    cust_serv_calls: int
    day_mins: float
    day_calls: int
    monthly_charge: float
    overage_fee: float
    roam_mins: float

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict")
def predict(data: CustomerData):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_API_KEY}"
    }

    payload = {
        "input_data": {
            "columns": [
                "AccountWeeks", "ContractRenewal", "DataPlan", "DataUsage",
                "CustServCalls", "DayMins", "DayCalls", "MonthlyCharge",
                "OverageFee", "RoamMins"
            ],
            "data": [[
                data.account_weeks, data.contract_renewal, data.data_plan,
                data.data_usage, data.cust_serv_calls, data.day_mins,
                data.day_calls, data.monthly_charge, data.overage_fee,
                data.roam_mins
            ]]
        },
        "params": {}
    }

    try:
        response = requests.post(AZURE_ENDPOINT_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        raw_result = response.json()
        
        prediction_val = raw_result[0] if isinstance(raw_result, list) else raw_result
        churn_risk = bool(prediction_val == 1)

        return {
            "success": True,
            "prediction": prediction_val,
            "churn_risk": churn_risk,
            "message": "High Churn Risk" if churn_risk else "Customer Likely to Stay"
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}