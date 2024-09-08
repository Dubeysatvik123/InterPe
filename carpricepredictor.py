import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder


@st.cache_data
def load_data():
    url = 'https://drive.google.com/uc?id=17wcOi_UoPklpR2R2a2OPv9RjOvCnLxWj'
    data = pd.read_csv(url)
    return data

df = load_data()


df = df.dropna()  

df['kms_driven'] = df['kms_driven'].str.replace(' kms', '').str.replace(',', '').astype(int)

df['Price'] = df['Price'].str.replace(',', '')  

df['Price'] = pd.to_numeric(df['Price'], errors='coerce')  
df = df.dropna(subset=['Price'])  
df['Price'] = df['Price'].astype(int) 

le_company = LabelEncoder()
le_fuel_type = LabelEncoder()

df['company'] = le_company.fit_transform(df['company'])
df['fuel_type'] = le_fuel_type.fit_transform(df['fuel_type'])

X = df[['year', 'kms_driven', 'company', 'fuel_type']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
st.write(f"Model Performance: Mean Absolute Error = {mae:.2f}")

st.title("Car Price Predictor")

year = st.slider("Car Year", int(df['year'].min()), int(df['year'].max()), step=1)
kms_driven = st.number_input("Kilometers Driven", min_value=0, max_value=int(df['kms_driven'].max()), step=1000)
company = st.selectbox("Company", le_company.classes_)
fuel_type = st.selectbox("Fuel Type", le_fuel_type.classes_)

input_data = pd.DataFrame({
    'year': [year],
    'kms_driven': [kms_driven],
    'company': [le_company.transform([company])[0]],
    'fuel_type': [le_fuel_type.transform([fuel_type])[0]]
})

if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"The predicted price for the car is ₹{prediction[0]:,.2f}")
