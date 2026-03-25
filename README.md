# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
Load the weather dataset using pandas.
Preprocess the data by handling missing values and sorting by time.
Select features and create lag variables for temperature and PM2.5.
Train Random Forest models to predict temperature and PM2.5 and save the models.
```

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: Mohamed Ukkas R
RegisterNumber:  25007472
*/


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("weather-station-eee-block_2024_07_13.csv")
df.columns = df.columns.str.strip()
df['time'] = pd.to_datetime(df['time'], errors='coerce')

print("Original rows:", len(df))

# Only drop if target missing
df = df.dropna(subset=['tem', 'pm2_5'])

# Fill feature columns instead of dropping
df['hum'] = df['hum'].fillna(df['hum'].mean())
df['pressure'] = df['pressure'].fillna(df['pressure'].mean())
df['wind_speed'] = df['wind_speed'].fillna(df['wind_speed'].mean())
df['co2'] = df['co2'].fillna(df['co2'].mean())

# Sort by time
df = df.sort_values('time')

# Create lag features
df['Temp_Lag1'] = df['tem'].shift(1)
df['PM_Lag1'] = df['pm2_5'].shift(1)

# Only remove first row created by shift
df = df.iloc[1:]

print("Rows after preprocessing:", len(df))

# Features
X = df[['hum', 'pressure', 'wind_speed', 'co2',
        'Temp_Lag1', 'PM_Lag1']]

y_temp = df['tem']
y_pm = df['pm2_5']

print("Training samples:", len(X))

# Train models
model_temp = RandomForestRegressor(n_estimators=300, random_state=42)
model_pm = RandomForestRegressor(n_estimators=300, random_state=42)

model_temp.fit(X, y_temp)
model_pm.fit(X, y_pm)

# Save models
joblib.dump(model_temp, "temperature_model.pkl")
joblib.dump(model_pm, "pm25_model.pkl")

print("Models trained and saved successfully!")

```

## Output:
<img width="1248" height="114" alt="564772029-51249e05-937e-464b-bf10-5159308b65a5" src="https://github.com/user-attachments/assets/8bc24ae7-7840-48d3-99ff-a5acc55d2e30" />
<img width="1263" height="463" alt="564772123-e25cc9b6-aa96-4a70-b77c-aa1d44041d72" src="https://github.com/user-attachments/assets/b5ccec2c-ba93-4e45-b759-feda534057bd" />
<img width="1268" height="460" alt="564772217-54b97dae-8691-4c37-a28d-5fa4c8b87096" src="https://github.com/user-attachments/assets/e7210362-e219-46b2-9721-438497c0fa2b" />
<img width="1271" height="465" alt="564772371-16f0abfb-3d60-4857-b285-a973cc664f12" src="https://github.com/user-attachments/assets/8a501263-7faa-468f-a5a8-31846e80bca7" />
<img width="1246" height="96" alt="564772487-477ca07a-1a8c-4284-989f-885e15cf4590" src="https://github.com/user-attachments/assets/966e45da-74fb-4362-a716-d1eb27e6ed30" />


## Result:
The Random Forest model successfully predicted temperature, PM2.5 pollution, and solar radiation using weather sensor data with good accuracy. The system also generated next-step predictions and visual graphs comparing actual vs predicted values and showing feature importance.
