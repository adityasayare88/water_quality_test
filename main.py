from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
from data_model import Water

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = FastAPI(
    title="Water Potability Prediction",
    description="Predicting Water Potability"
)


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def index():
    return "Welcome to Water Quality Prediction using FastAPI"

@app.post("/predict")
def model_predict(water: Water):
    sample = pd.DataFrame({
        'ph': [water.ph],
        'Hardness': [water.Hardness],
        'Solids': [water.Solids],
        'Chloramines': [water.Chloramines],
        'Sulfate': [water.Sulfate],
        'Conductivity': [water.Conductivity],
        'Organic_carbon': [water.Organic_carbon],
        'Trihalomethanes': [water.Trihalomethanes],
        'Turbidity': [water.Turbidity]
    })
    
    predicted_value = model.predict(sample)
    prediction = predicted_value[0]  # Extract the first element from numpy array
    
    if prediction == 1:
        return "Water is potable"
    else:
        return "Water is not potable"
