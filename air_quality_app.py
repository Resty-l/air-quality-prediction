
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
from datetime import datetime, timedelta
import tensorflow as tf
import joblib
from sklearn.preprocessing import StandardScaler

# ----------------------------
# LOAD MODEL AND SCALER
# ----------------------------
model_path = "/content/drive/My Drive/capstone/airqo_prediction_model_lstm.keras"
model = tf.keras.models.load_model(model_path)

scaler_path = "/content/drive/My Drive/capstone/scaler.pkl"
scaler = joblib.load(scaler_path)

# Define the features list used during training
features = ['latitude', 'longitude', 'humidity', 'temperature',
            's5p_no2', 's5p_ai', 'lst_temp_k', 'wind_u', 'wind_v',
            'day_sin', 'day_cos', 'pm2_5_lag_1', 'pm2_5_lag_2', 'pm2_5_lag_3']

def predict_pm25(lat, lon):
    # Get current date for cyclical features
    current_date = datetime.now()
    day_of_year = current_date.timetuple().tm_yday
    day_sin = np.sin(2 * np.pi * day_of_year / 365)
    day_cos = np.cos(2 * np.pi * day_of_year / 365)

    # Create a DataFrame for a single prediction point
    # For simplicity, we are setting other features to 0.
    # In a real application, you would fetch real-time data for these.
    input_data = pd.DataFrame([[
        lat, lon, 0.0, 0.0, # humidity, temperature (placeholders)
        0.0, 0.0, 0.0, 0.0, 0.0, # s5p_no2, s5p_ai, lst_temp_k, wind_u, wind_v (placeholders)
        day_sin, day_cos,
        0.0, 0.0, 0.0 # pm2_5_lag_1, pm2_5_lag_2, pm2_5_lag_3 (placeholders)
    ]], columns=features)

    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Reshape for LSTM (samples, timesteps, features)
    reshaped_input = scaled_input.reshape(1, 1, scaled_input.shape[1])

    # Make prediction
    prediction = model.predict(reshaped_input)[0][0]
    return round(float(prediction), 2)

# ----------------------------
# AQI LOGIC (WHO standards) with advice
# ----------------------------
def categorize_aqi(pm25):
    if pm25 <= 12:
        return "Good", "Green", "Perfect day for outdoor activities, air quality is excellent!"
    elif pm25 <= 35.4:
        return "Moderate", "Yellow", "Sensitive groups should limit prolonged exertion outdoors."
    elif pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups", "Orange", "Consider wearing a mask; active children and adults, and people with respiratory disease, such as asthma, should avoid prolonged outdoor exertion."
    else:
        return "Unhealthy", "Red", "Avoid outdoor activity; everyone may begin to experience health effects, and members of sensitive groups may experience more serious health effects."

# ----------------------------
# UI CONFIGURATION
# ----------------------------
st.set_page_config(layout="wide", page_title="Air Quality Monitoring App")

st.title("🌍 Air Quality Monitoring App")

tab1, tab2 = st.tabs(["Citizen Mobile App", "City Planner Dashboard"])

# ----------------------------
# CITIZEN MOBILE APP VIEW
# ----------------------------
with tab1:
    st.subheader("Check Air Quality Near You")

    st.write("Enter your location to get a real-time (simulated) PM2.5 reading and health advice.")

    col1, col2 = st.columns(2)
    lat = col1.number_input("Latitude", value=0.3476, format="%.4f")
    lon = col2.number_input("Longitude", value=32.5825, format="%.4f")

    if st.button("Get Air Quality"): # Add a button to trigger prediction
        pm25 = predict_pm25(lat, lon)
        category, color, advice = categorize_aqi(pm25)

        st.metric("Current PM2.5 Level", f"{pm25} µg/m³")

        if color == "Green":
            st.success(f"**{category}** – {advice}")
        elif color == "Yellow":
            st.warning(f"**{category}** – {advice}")
        elif color == "Orange":
            st.warning(f"**{category}** – {advice}")
        else: # Red
            st.error(f"**{category}** – {advice}")

# ----------------------------
# CITY PLANNER DASHBOARD VIEW
# ----------------------------
with tab2:
    st.sidebar.header("Dashboard Controls")
    selected_date = st.sidebar.date_input("Select Date for Analysis", datetime.today())

    st.subheader("Air Quality Heatmap Across the Region")

    # Simulated dataset for heatmap (replace with actual data loading/filtering for selected_date)
    num_points = 500
    data = pd.DataFrame({
        "lat": np.random.uniform(-1.5, 1.5, num_points) + 0.3, # Centered around Uganda
        "lon": np.random.uniform(31, 34, num_points), # Centered around Uganda
        "pm25": np.random.uniform(5, 150, num_points)
    })
    # Add AQI categories to simulated data
    data['category'], data['color'], data['advice'] = zip(*data['pm25'].apply(categorize_aqi))

    # DOWNLOAD DATA BUTTON
    csv = data.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        "Download Data as CSV",
        csv,
        f"air_quality_data_{selected_date}.csv",
        "text/csv",
        key='download-csv'
    )

    # PyDeck Heatmap Layer
    layer = pdk.Layer(
        "HeatmapLayer",
        data=data,
        get_position='[lon, lat]',
        get_weight="pm25",
        radiusPixels=60,
        opacity=0.8,
    )

    # PyDeck View State
    view_state = pdk.ViewState(
        latitude=data["lat"].mean(),
        longitude=data["lon"].mean(),
        zoom=6,
        pitch=45,
    )

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": "<b>PM2.5:</b> {pm25} µg/m³<br/><b>Category:</b> {category}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }
    ))

    # ----------------------------
    # 24-HOUR POLLUTION TREND
    # ----------------------------
    st.subheader("24-Hour Pollution Trend (Simulated)")

    # Simulated trend data (replace with actual time-series data)
    hours = pd.date_range(datetime.now() - timedelta(days=1), periods=24, freq='h')
    trend = pd.DataFrame({
        "time": hours,
        "pm25": np.random.uniform(10, 100, 24)
    })

    fig = px.line(trend, x="time", y="pm25", title="PM2.5 Trend Over Last 24 Hours")
    st.plotly_chart(fig, use_container_width=True)
