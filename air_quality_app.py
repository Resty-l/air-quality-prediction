import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import joblib

# ---------------- MODEL ----------------
class AirQualityLSTM(nn.Module):
    def __init__(self, input_size=14, hidden_size=128):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

@st.cache_resource
def load():
    model = AirQualityLSTM(14)
    model.load_state_dict(torch.load("airqo_model.pth", map_location="cpu"))
    model.eval()
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load()

features = ['latitude','longitude','humidity','temperature',
            's5p_no2','s5p_ai','lst_temp_k','wind_u','wind_v',
            'day_sin','day_cos','pm2_5_lag_1','pm2_5_lag_2','pm2_5_lag_3']

# ---------------- FORECAST ----------------
def predict_7_day_forecast(lat, lon):
    data, last = [], 10
    now = datetime.now()

    for i in range(7):
        d = now + timedelta(days=i)
        day = d.timetuple().tm_yday

        vals = [lat, lon,
                np.random.uniform(40,80),
                np.random.uniform(20,30),
                0.0001, 0.5, 300,
                np.random.uniform(-1,1),
                np.random.uniform(-1,1),
                np.sin(2*np.pi*day/365),
                np.cos(2*np.pi*day/365),
                last, 0, 0]

        df = pd.DataFrame([vals], columns=features)
        x = scaler.transform(df)
        x = torch.FloatTensor(x).reshape(1,1,14)

        with torch.no_grad():
            pred = max(0, float(model(x)))

        data.append({"Day": d.strftime('%A'),
                     "Date": d.strftime('%b %d'),
                     "PM2.5": round(pred,2)})
        last = pred

    return pd.DataFrame(data)

# ---------------- HEATMAP ----------------
def generate_map():
    lats = np.arange(-1.5, 4.5, 0.3)
    lons = np.arange(29.5, 35.0, 0.3)

    rows = []
    for lat in lats:
        for lon in lons:
            pm = predict_7_day_forecast(lat, lon).iloc[0]['PM2.5']
            rows.append({"lat": lat, "lon": lon, "pm25": pm})

    return pd.DataFrame(rows)

# ---------------- AQI ----------------
def categorize(pm):
    if pm <= 12: return "Good","green"
    elif pm <= 35: return "Moderate","orange"
    else: return "Unhealthy","red"

# ---------------- UI ----------------
st.set_page_config(layout="wide")
st.title("Uganda Air Quality System")

tab1, tab2 = st.tabs(["📍 Local Forecast", "📊 Country Heatmap"])

# -------- TAB 1 --------
with tab1:
    st.subheader("7-Day Local Forecast")

    col1, col2 = st.columns(2)
    lat = col1.number_input("Latitude", value=0.3476)
    lon = col2.number_input("Longitude", value=32.5825)

    if st.button("Generate Forecast"):
        df = predict_7_day_forecast(lat, lon)

        val = df.iloc[0]['PM2.5']
        cat, color = categorize(val)

        st.metric("Today's PM2.5", f"{val} µg/m³")
        st.write(f"Status: {cat}")

        fig = px.bar(df, x="Day", y="PM2.5",
                     text="PM2.5", color="PM2.5",
                     color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

        st.table(df)

# -------- TAB 2 --------
with tab2:
    st.subheader("Uganda PM2.5 Heatmap")

    if st.button("Generate Heatmap"):
        df = generate_map()

        st.pydeck_chart(pdk.Deck(
            layers=[pdk.Layer(
                "HeatmapLayer",
                data=df,
                get_position='[lon, lat]',
                get_weight="pm25",
                radiusPixels=50
            )],
            initial_view_state=pdk.ViewState(
                latitude=1.37,
                longitude=32.29,
                zoom=6
            )
        ))
