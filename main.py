from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("rf_model.joblib") # loaded ONCE

labels = ["Healthy", "Obese", "Overweight", "Underweight"]

# Pydantic model = automatic input validation
class PersonInfo(BaseModel):
    Height_cm: float
    Weight_kg: float
    BMI: float
    Protein_Intake_g: float
    Carbohydrate_Intake_g: float
    Fat_Intake_g: float
    Water_Intake_Liters: float
    Calorie_Difference: int 
    Activity_Lvl_Enc: int

@app.post("/predict")
def predict(data: PersonInfo):  
    features = [[data.Height_cm, data.Weight_kg, data.BMI, data.Protein_Intake_g, data.Carbohydrate_Intake_g,
                 data.Fat_Intake_g, data.Water_Intake_Liters, data.Calorie_Difference, data.Activity_Lvl_Enc]]
    
    pred = model.predict(features)[0]
    return {"prediction": labels[pred]}
