import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import joblib
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="uganda_air_quality_app")

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
    # Ensure these files exist in your directory
    model = AirQualityLSTM(input_size=14)
    model.load_state_dict(torch.load('airqo_model.pth', map_location=torch.device('cpu')))
    model.eval()
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_resources()

features = ['latitude', 'longitude', 'humidity', 'temperature',
            's5p_no2', 's5p_ai', 'lst_temp_k', 'wind_u', 'wind_v',
            'day_sin', 'day_cos', 'pm2_5_lag_1', 'pm2_5_lag_2', 'pm2_5_lag_3']

def predict_7_day_forecast(lat, lon):
    forecast_data = []
    current_date = datetime.now()
    
    # Use the current prediction as the starting point for lags
    last_pm25 = 0.0         

    for i in range(7):
        forecast_date = current_date + timedelta(days=i)
        day_of_year = forecast_date.timetuple().tm_yday
        
        day_sin = np.sin(2 * np.pi * day_of_year / 365)
        day_cos = np.cos(2 * np.pi * day_of_year / 365)

        # --- ADDING NATURAL NOISE ---
        simulated_wind = np.random.uniform(-1.5, 1.5) 
        simulated_humidity = np.random.uniform(40, 80)
        simulated_temp = np.random.uniform(20, 30)     

        input_vals = [
            lat, lon, 
            simulated_humidity, 
            simulated_temp, 
            0.0001, # NO2 baseline
            0.5,    # Aerosol Index baseline
            300.0,  # LST (Kelvin)
            simulated_wind, 
            simulated_wind, 
            day_sin, 
            day_cos, 
            last_pm25, 
            0.0, 0.0
        ]
        
        input_df = pd.DataFrame([input_vals], columns=features)
        scaled_input = scaler.transform(input_df)
        tensor_input = torch.FloatTensor(scaled_input).reshape(1, 1, 14)
        
        with torch.no_grad():
            prediction = model(tensor_input).item()
            # 5% "Atmospheric Cleaning" factor
            prediction = max(0, float(prediction)) * 0.95 
        
        forecast_data.append({
            "Day": forecast_date.strftime('%A'),
            "Date": forecast_date.strftime('%b %d'),
            "PM2.5": round(prediction, 2)
        })
        
        last_pm25 = prediction
            
    return pd.DataFrame(forecast_data)

# ----------------------------
# AQI LOGIC & UI
# ----------------------------
def categorize_aqi(pm25):
    if pm25 <= 12: return "Good", "Green", "Perfect day for outdoor activities!"
    elif pm25 <= 35.4: return "Moderate", "Yellow", "Sensitive groups should limit exertion."
    elif pm25 <= 55.4: return "Unhealthy (Sensitive)", "Orange", "Wear a mask if you have asthma."
    else: return "Unhealthy", "Red", "Avoid outdoor activity; air quality is poor."

st.set_page_config(layout="wide", page_title="Uganda Air Quality")
st.title("Uganda Air Quality Monitoring")

tab1, tab2 = st.tabs(["📍 Local Check", "📊 Planner Dashboard"])

with tab1:
    st.subheader("7-Day Forecast")
    
    # --- NEW SEARCH BAR SECTION ---
    search_query = st.text_input("🔍 Search for a location in Uganda (e.g. Nakasero, Jinja, Mbarara)", "")
    
    # Default values (Kampala)
    default_lat = 0.3476
    default_lon = 32.5825

    if search_query:
        try:
            # We append 'Uganda' to ensure it searches within the country
            location = geolocator.geocode(f"{search_query}, Uganda")
            if location:
                default_lat = location.latitude
                default_lon = location.longitude
                st.info(f"Found: {location.address}")
            else:
                st.error("Location not found. Using default Kampala coordinates.")
        except Exception as e:
            st.error("Service busy. Please enter coordinates manually.")

    st.write("---")
    
    # Manual Coordinate Inputs (Automatically updated by search)
    col1, col2 = st.columns(2)
    lat = col1.number_input("Latitude", value=default_lat, format="%.4f")
    lon = col2.number_input("Longitude", value=default_lon, format="%.4f")
    
    if st.button("Generate Forecast"):
        forecast_df = predict_7_day_forecast(lat, lon)
        
        # --- Current Day Hero Metric ---
        today_val = forecast_df.iloc[0]['PM2.5']
        category, color, advice = categorize_aqi(today_val)
        
        st.divider()
        c1, c2 = st.columns([1, 2])
        
        with c1:
            st.metric("Today's Estimated PM2.5", f"{today_val} µg/m³")
            st.markdown(f"**Status:** {category}")
            if color == "Green": st.success(advice)
            elif color == "Yellow": st.warning(advice)
            elif color == "Orange": st.warning(advice)
            else: st.error(advice)

        with c2:
            # --- Weekly Trend Chart ---
            fig = px.bar(forecast_df, x='Day', y='PM2.5', 
                         text='PM2.5', title="7-Day Trend",
                         color='PM2.5', color_continuous_scale='Reds')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("See Detailed Daily Breakdown"):
            st.table(forecast_df)

# ----------------------------
# CITY PLANNER DASHBOARD VIEW
# ----------------------------
with tab2:
    st.subheader("Regional Analysis (Uganda Overview)")
    
    # 1. Expand the range to cover all of Uganda
    # Latitude: South (~ -1.5) to North (~ 4.5)
    # Longitude: West (~ 29.5) to East (~ 35.0)
    num_points = 500  # Increased points for better national coverage
    data = pd.DataFrame({
        "lat": np.random.uniform(-1.5, 4.5, num_points),
        "lon": np.random.uniform(29.5, 35.0, num_points),
        "pm25": np.random.uniform(10, 80, num_points)
    })
    
    st.pydeck_chart(pdk.Deck(
        layers=[pdk.Layer(
            "HeatmapLayer", 
            data=data, 
            get_position='[lon, lat]', 
            get_weight="pm25", 
            radiusPixels=40 # Adjusted for a wider view
        )],
        # 2. Adjust the initial view to center on the middle of Uganda
        initial_view_state=pdk.ViewState(
            latitude=1.3733,   # Geographic center of Uganda
            longitude=32.2903, 
            zoom=6             # Lower zoom level to see the whole country
        )
    ))
