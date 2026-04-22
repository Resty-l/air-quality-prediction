import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import joblib

# ----------------------------
# PYTORCH MODEL CLASS DEFINITION
# ----------------------------
class AirQualityLSTM(nn.Module):
    def __init__(self, input_size=14, hidden_size=128):
        super(AirQualityLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.network = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.network(h_n[-1])

# ----------------------------
# LOAD ASSETS (Optimized for Streamlit)
# ----------------------------
@st.cache_resource
def load_resources():
    # Use relative paths for GitHub deployment
    model = AirQualityLSTM(input_size=14)
    model.load_state_dict(torch.load('airqo_model.pth', map_location=torch.device('cpu')))
    model.eval()
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_resources()

features = ['latitude', 'longitude', 'humidity', 'temperature',
            's5p_no2', 's5p_ai', 'lst_temp_k', 'wind_u', 'wind_v',
            'day_sin', 'day_cos', 'pm2_5_lag_1', 'pm2_5_lag_2', 'pm2_5_lag_3']

def predict_pm25(lat, lon):
    current_date = datetime.now()
    day_of_year = current_date.timetuple().tm_yday
    day_sin = np.sin(2 * np.pi * day_of_year / 365)
    day_cos = np.cos(2 * np.pi * day_of_year / 365)

    # Note: Using 0.0 for missing sensors/meteorology
    input_vals = [lat, lon, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, day_sin, day_cos, 0.0, 0.0, 0.0]
    input_df = pd.DataFrame([input_vals], columns=features)
    
    scaled_input = scaler.transform(input_df)
    tensor_input = torch.FloatTensor(scaled_input).reshape(1, 1, 14)

    with torch.no_grad():
        prediction = model(tensor_input).item()
    
    return round(max(0, float(prediction)), 2) # Prevent negative values

# ----------------------------
# AQI LOGIC & UI (Remains mostly same)
# ----------------------------
def categorize_aqi(pm25):
    if pm25 <= 12: return "Good", "Green", "Perfect day for outdoor activities!"
    elif pm25 <= 35.4: return "Moderate", "Yellow", "Sensitive groups should limit exertion."
    elif pm25 <= 55.4: return "Unhealthy (Sensitive)", "Orange", "Wear a mask if you have asthma."
    else: return "Unhealthy", "Red", "Avoid outdoor activity; air quality is poor."

st.set_page_config(layout="wide", page_title="Uganda Air Quality")
st.title("🌍 Uganda Air Quality Monitoring")

tab1, tab2 = st.tabs(["📍 Local Check", "📊 Planner Dashboard"])

with tab1:
    st.subheader("Real-time Estimation")
    col1, col2 = st.columns(2)
    lat = col1.number_input("Latitude", value=0.3476, format="%.4f")
    lon = col2.number_input("Longitude", value=32.5825, format="%.4f")

    if st.button("Predict PM2.5"):
        pm25 = predict_pm25(lat, lon)
        cat, col, advice = categorize_aqi(pm25)
        st.metric("PM2.5 Level", f"{pm25} µg/m³")
        if col == "Green": st.success(advice)
        elif col == "Yellow": st.warning(advice)
        elif col == "Orange": st.warning(advice)
        else: st.error(advice)

with tab2:
    st.subheader("Regional Analysis")
    # Heatmap logic remains identical to your script
    num_points = 100
    data = pd.DataFrame({
        "lat": np.random.uniform(0.1, 0.5, num_points),
        "lon": np.random.uniform(32.4, 32.7, num_points),
        "pm25": np.random.uniform(10, 80, num_points)
    })
    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer("HeatmapLayer", data=data, get_position='[lon, lat]', get_weight="pm25", radiusPixels=50)],
        initial_view_state=pdk.ViewState(latitude=0.34, longitude=32.58, zoom=10)
    ))
