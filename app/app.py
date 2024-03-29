# Some Example customer JSON's:

# customer0 = Customer(customerID="BEEP", gender="Male", SeniorCitizen="No", Partner = "No", Dependents = "Yes", tenure = 5, PhoneService = "Yes",
#                     MultipleLines = "No", InternetService = "DSL", OnlineSecurity = "No", OnlineBackup = "No", DeviceProtection = "No", TechSupport = "No",
#                     StreamingTV = "No", StreamingMovies = "Yes", Contract = "One year", PaperlessBilling = "Yes", PaymentMethod = "Bank transfer (automatic)",
#                     MonthlyCharges = 15.0, TotalCharges = 15.0*5)

# customer1 = Customer(customerID="BEEP", gender="Male", SeniorCitizen="No", Partner = "No", Dependents = "No", tenure = 1, PhoneService = "Yes",
#                     MultipleLines = "No", InternetService = "No", OnlineSecurity = "No internet service", OnlineBackup = "No internet service", DeviceProtection = "No internet service", TechSupport = "No internet service",
#                     StreamingTV = "No internet service", StreamingMovies = "No internet service", Contract = "Month-to-month", PaperlessBilling = "No", PaymentMethod = "Mailed check",
#                     MonthlyCharges = 20.15, TotalCharges = 20.15)

from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, Field

import torch
import pandas as pd
from typing import Literal, Optional
from app_models import predict

# Instanciate the FastAPI application
app = FastAPI()


# Define the pydantic dataclasses used by the application
class Customer(BaseModel):
    customerID: str
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal["Yes", "No"]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["No", "DSL", "Fiber optic"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float
    TotalCharges: float


class Responce(BaseModel):
    churn: Literal["Yes", "No"]


@app.get("/")
def index():
    return {"Health check": "OK"}


@app.post("/predict", response_model=Responce)
def predict_churn(payload: Customer):
    churn = predict(payload.model_dump())
    return {"churn": churn}
