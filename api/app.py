from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Annotated
import pickle
import pandas as pd
from fastapi.responses import JSONResponse

with open('model.pkl', 'rb') as f:
    bundle = pickle.load(f)

scaler = bundle["scaler"]
models = bundle["models"]
disease_columns = bundle["disease_columns"]

app = FastAPI()

class UserInput(BaseModel):
    age: Annotated[int, Field(..., gt=0, lt=120, description='AGE of user')]
    gender: Annotated[str, Field(..., description="gender of user")]
    blood_pressure: Annotated[str, Field(..., description="blood_pressure of user")]
    smoking: Annotated[str, Field(..., description="user is smoker?")]
    alcohol_consumption: Annotated[str, Field(..., description="user do alcohol_consumption?")]
    exercise: Annotated[str, Field(..., description="user do Exercise?")]
    cholesterol: Annotated[str, Field(..., description="cholesterol of user")]
    glucose: Annotated[str, Field(..., description="glucose of user")]
    family_history: Annotated[str, Field(..., description="family_disease_history")]
    height: Annotated[int, Field(..., gt=0, description='user height in cm')]
    weight: Annotated[int, Field(..., gt=0, description='user weight in kg')]

def calc_bmi_category(height_cm, weight_kg):
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        return 2   # Underweight
    elif bmi < 25:
        return 0   # Normal
    else:
        return 1   # Overweight

@app.get("/")
def home():
    return {"message": "Disease Prediction API is running! ✅"}

@app.post('/predict')
def predict(data: UserInput):
    try:                                          
        mapping1 = {"Yes": 1, "No": 0}
        mapping2 = {"Low": 0, "Normal": 1, "High": 2}
        gender_map = {"Male": 1, "Female": 0}

        bmi_category = calc_bmi_category(data.height, data.weight)

        input_df = pd.DataFrame([{              
            'age':                 data.age,
            'gender':              gender_map[data.gender],
            'blood_pressure':      mapping2[data.blood_pressure],
            'cholesterol':         mapping2[data.cholesterol],
            'glucose':             mapping2[data.glucose],
            'smoking':             mapping1[data.smoking],
            'alcohol_consumption': mapping1[data.alcohol_consumption],
            'exercise':            mapping1[data.exercise],
            'family_history':      mapping1[data.family_history],
            'bmi_category':        bmi_category,
        }])

        X_scaled = scaler.transform(input_df)   

        predictions = {                         
            disease: int(models[disease].predict(X_scaled)[0])
            for disease in disease_columns
        }

        return JSONResponse(status_code=200, content={   
            "predictions": predictions
        })

    except Exception as e:                      
        return JSONResponse(status_code=500, content={"error": str(e)})


     
     