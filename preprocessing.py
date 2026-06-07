from database import df
import pandas as pd
df["bmi_category"] = df['bmi'].apply(
    lambda x: "Underweight" if x < 18.5 else ("Normal" if x < 25 else "Overweight")
)
df = df.drop("bmi", axis=1) 


mapping2={
    "Normal":0,
    "Overweight":1,
    "Underweight":2
}
mapping1={
   "Yes":1,
   "No":0
}

df["gender"] = df["gender"].map({
     "Male": 1,
    "Female": 0
})

mapping= {
    "Low":0,
    "Normal":1,
    "High":2
}
# Smoking
df["smoking"] = df["smoking"].map(mapping1)

# Alcohol
df["alcohol_consumption"] = df["alcohol_consumption"].map(mapping1)

# Exercise
df["exercise"] = df["exercise"].map(mapping1)


df["blood_pressure"]=df['blood_pressure'].map(mapping)
   

df["cholesterol"]=df['cholesterol'].map(mapping)

    

df["glucose"]=df['glucose'].map(mapping)

    
df["bmi_category"]=df['bmi_category'].map(mapping2)

df["family_history"]=df['family_history'].map(mapping1)
disease_columns = [
    'heart_disease',
    'diabetes',
    'stroke',
    'kidney_disease',
    'cancer',
    'alzheimers_disease',
    'copd',
    'liver_disease',
    'parkinsons_disease',
    'tuberculosis'
]




 
x= df.drop(disease_columns,axis=1)
y=df[disease_columns]


#print(x.columns.tolist())
# print(df.head())

