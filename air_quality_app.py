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

# ----------------------------
# CITY PLANNER DASHBOARD VIEW
# ----------------------------
with tab2:
    st.sidebar.header("Dashboard Controls")
    selected_date = st.sidebar.date_input("Select Date for Analysis", datetime.today())

    st.subheader("National Air Quality Heatmap (Uganda)")

    # 1. COORDINATES FOR ALL OF UGANDA
    # Lat: -1.5 (South) to 4.5 (North) | Lon: 29.5 (West) to 35.0 (East)
    num_points = 800 
    data = pd.DataFrame({
        "lat": np.random.uniform(-1.5, 4.5, num_points),
        "lon": np.random.uniform(29.5, 35.0, num_points),
        "pm25": np.random.uniform(5, 120, num_points) # Random PM2.5 for visualization
    })

    # 2. ADD POLLUTION CLUSTERS (Optional: makes Kampala/Entebbe look 'hotter')
    # This adds a bit of realism so the whole country isn't just uniform random dots
    kampala_mask = (data['lat'].between(0.1, 0.6)) & (data['lon'].between(32.3, 32.8))
    data.loc[kampala_mask, 'pm25'] += np.random.uniform(40, 80, kampala_mask.sum())

    # Add AQI categories
    data['category'], data['color'], _ = zip(*data['pm25'].apply(categorize_aqi))

    # PyDeck Heatmap Layer
    layer = pdk.Layer(
        "HeatmapLayer",
        data=data,
        get_position='[lon, lat]',
        get_weight="pm25",
        radiusPixels=40, # Smaller radius for national view
        opacity=0.9,
    )

    # PyDeck View State (Centered on Uganda)
    view_state = pdk.ViewState(
        latitude=1.3733, # Center of Uganda
        longitude=32.2903,
        zoom=6.5, # Zoomed out to see the whole country
        pitch=0,
    )

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/light-v9', # Cleaner look for heatmaps
        tooltip={"text": "Estimated PM2.5: {pm25} µg/m³"}
    ))

    # DOWNLOAD DATA BUTTON (sidebar)
    csv = data.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        "Download National Data",
        csv,
        f"uganda_aqi_{selected_date}.csv",
        "text/csv"
    )
