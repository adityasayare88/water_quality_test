from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
from data_model import Water

app = FastAPI(
    title="Water Potability Prediction",
    description="Predicting Water Potability"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def index():
    return {"message": "Welcome to Water Quality Prediction using FastAPI"}

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
    
    # Return JSON response
    return {
        "prediction": int(predicted_value[0]),
        "message": "Water is potable" if predicted_value[0] == 1 else "Water is not potable"
    }
