import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
from twilio.rest import Client

# Load the dataset
data = pd.read_csv('data/weather_data.csv')

# Ensure 'YEAR', 'MO', and 'DY' are integers
data['YEAR'] = pd.to_numeric(data['YEAR'], errors='coerce').fillna(0).astype(int)
data['MO'] = pd.to_numeric(data['MO'], errors='coerce').fillna(0).astype(int)
data['DY'] = pd.to_numeric(data['DY'], errors='coerce').fillna(0).astype(int)

# Function to create a date
def create_date(row):
    try:
        year = int(row['YEAR'])
        month = int(row['MO'])
        day = int(row['DY'])
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return pd.NaT

# Apply the function to create a new date column
data['Date'] = data.apply(create_date, axis=1)

# Extract features
X = data[['LAT', 'LON', 'Date']].copy()
X['Year'] = X['Date'].dt.year
X['Month'] = X['Date'].dt.month
X['Day'] = X['Date'].dt.day

# Define target variables
y_temp = data['Temperature']
y_wind = data['Wind(m/s)']
y_rainfall = data['Precipitation(mm/day)']
y_humidity = data['Humidity (%)']

# Drop any rows with NaN values in features or target
X = X.dropna()
y_temp = y_temp[X.index]
y_wind = y_wind[X.index]
y_rainfall = y_rainfall[X.index]
y_humidity = y_humidity[X.index]

# Split the dataset into training and testing sets
X_train_temp_wind, X_test_temp_wind, y_train_temp, y_test_temp = train_test_split(X[['LAT', 'LON', 'Year', 'Month', 'Day']], y_temp, test_size=0.2, random_state=42)
X_train_wind, X_test_wind, y_train_wind, y_test_wind = train_test_split(X[['LAT', 'LON', 'Year', 'Month', 'Day']], y_wind, test_size=0.2, random_state=42)
X_train_rain, X_test_rain, y_train_rain, y_test_rain = train_test_split(X[['LAT', 'LON', 'Year', 'Month', 'Day']], y_rainfall, test_size=0.2, random_state=42)
X_train_hum, X_test_hum, y_train_hum, y_test_hum = train_test_split(X[['LAT', 'LON', 'Year', 'Month', 'Day']], y_humidity, test_size=0.2, random_state=42)

# Initialize Random Forest models
model_wind_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_temp_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rain_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_hum_rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the models
model_wind_rf.fit(X_train_wind, y_train_wind)
model_temp_rf.fit(X_train_temp_wind, y_train_temp)
model_rain_rf.fit(X_train_rain, y_train_rain)
model_hum_rf.fit(X_train_hum, y_train_hum)


# Initialize Twilio Client
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Function to send SMS alert
def send_sms_alert(condition, value):
    message = f"Alert: Extreme {condition} detected! Value: {value}. Stay Safe!"
    client.messages.create(
        body=message,
        from_=TWILIO_PHONE_NUMBER,
        to=USER_PHONE_NUMBER
    )

# Function to predict wind, temperature, rainfall, and humidity
def predict_all(lat, lon, date):
    date_obj = pd.to_datetime(date)
    features = np.array([[lat, lon, date_obj.year, date_obj.month, date_obj.day]])

    predicted_wind_rf = model_wind_rf.predict(features)
    predicted_temp_rf = model_temp_rf.predict(features)
    predicted_rain_rf = model_rain_rf.predict(features)
    predicted_hum_rf = model_hum_rf.predict(features)

    return predicted_wind_rf[0], predicted_temp_rf[0], predicted_rain_rf[0], predicted_hum_rf[0]

def predict_and_alert(lat, lon, date):
            predicted_wind, predicted_temp, predicted_rain, predicted_hum = predict_all(lat, lon, date)

            # Define extreme weather thresholds
            if predicted_wind > 30:  # Example threshold for extreme wind max-30
                send_sms_alert("Wind", predicted_wind)
            if predicted_temp > 40:  # Example threshold for extreme temperature max-40
                send_sms_alert("Temperature", predicted_temp)
            if predicted_rain > 100:  # Example threshold for extreme rainfall max-100
                send_sms_alert("Rainfall", predicted_rain)
            if predicted_hum > 60:  # Example threshold for extreme humidity max-90
                send_sms_alert("Humidity", predicted_hum)

            return predicted_wind, predicted_temp, predicted_rain, predicted_hum

# Streamlit interface
st.set_page_config(page_title="Weather Prediction App", layout="wide")

# Set background color
st.markdown(
    """
    <style>
    .reportview-container {
        background-color: #87CEEB;  /* Sky Blue */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header/navbar
st.markdown(
    """
    <h1 style="text-align: center;">Weather Prediction App</h1>
    <nav style="text-align: center;">
            <a class="weatherwidget-io" href="https://forecast7.com/en/19d0872d88/mumbai/" data-label_1="MUMBAI" data-label_2="WEATHER" data-icons="Climacons Animated" data-theme="retro-sky" >MUMBAI WEATHER</a>
    <script>
    !function(d,s,id){var js,fjs=d.getElementsByTagName(s)[0];if(!d.getElementById(id)){js=d.createElement(s);js.id=id;js.src='https://weatherwidget.io/js/widget.min.js';fjs.parentNode.insertBefore(js,fjs);}}(document,'script','weatherwidget-io-js');
    </script>
    </nav>
    """,
    unsafe_allow_html=True
)

# User inputs
st.write("Enter the date, latitude, and longitude for weather predictions.")
col1, col2 = st.columns([2, 1])  # Two columns layout

with col1:
    lat_input = st.number_input("Enter Latitude:", value=22.75)
    lon_input = st.number_input("Enter Longitude:", value=74.25)
    date_str = st.date_input("Select Date:")

    # Convert date to string in the required format
    date_str = date_str.strftime('%Y-%m-%d')

    # Display inputs
    st.write(f"Latitude: {lat_input}, Longitude: {lon_input}, Date: {date_str}")

    # Button to trigger prediction
    if st.button("Predict"):
        # Add error handling
        try:
            # Ensure inputs are in correct format
            lat = float(lat_input)
            lon = float(lon_input)

            predicted_wind_rf, predicted_temp_rf, predicted_rain_rf, predicted_hum_rf = predict_and_alert(lat, lon, date_str)

            # Display the results
            st.success(f'Predicted Wind: {predicted_wind_rf:.2f} m/s')
            st.success(f'Predicted Temperature: {predicted_temp_rf:.2f}°C')
            st.success(f'Predicted Rainfall: {predicted_rain_rf:.2f} mm/day')
            st.success(f'Predicted Humidity: {predicted_hum_rf:.2f} g/kg')

            # Display predictions
            st.subheader("Predicted Weather Conditions")
            st.write(f"Predicted Wind: {predicted_wind_rf:.2f} m/s")
            st.write(f"Predicted Temperature: {predicted_temp_rf:.2f} °C")
            st.write(f"Predicted Rainfall: {predicted_rain_rf:.2f} mm/day")
            st.write(f"Predicted Humidity: {predicted_hum_rf:.2f} g/kg")

            # Visualization
            st.subheader("Prediction Visualizations")
            fig, ax = plt.subplots(2, 2, figsize=(10, 8))

            # Wind
            sns.histplot(y_wind, ax=ax[0, 0], kde=True, color='blue')
            ax[0, 0].axvline(predicted_wind_rf, color='red', linestyle='dashed', linewidth=1)
            ax[0, 0].set_title('Wind Prediction Distribution')
            ax[0, 0].set_xlabel('Wind (m/s)')
            ax[0, 0].set_ylabel('Frequency')

            # Temperature
            sns.histplot(y_temp, ax=ax[0, 1], kde=True, color='orange')
            ax[0, 1].axvline(predicted_temp_rf, color='red', linestyle='dashed', linewidth=1)
            ax[0, 1].set_title('Temperature Prediction Distribution')
            ax[0, 1].set_xlabel('Temperature (°C)')
            ax[0, 1].set_ylabel('Frequency')

            # Rainfall
            sns.histplot(y_rainfall, ax=ax[1, 0], kde=True, color='green')
            ax[1, 0].axvline(predicted_rain_rf, color='red', linestyle='dashed', linewidth=1)
            ax[1, 0].set_title('Rainfall Prediction Distribution')
            ax[1, 0].set_xlabel('Rainfall (mm/day)')
            ax[1, 0].set_ylabel('Frequency')

            # Humidity
            sns.histplot(y_humidity, ax=ax[1, 1], kde=True, color='purple')
            ax[1, 1].axvline(predicted_hum_rf, color='red', linestyle='dashed', linewidth=1)
            ax[1, 1].set_title('Humidity Prediction Distribution')
            ax[1, 1].set_xlabel('Humidity (g/kg)')
            ax[1, 1].set_ylabel('Frequency')

            plt.tight_layout()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            print(e)  # Print the error for debugging in the console

with col2:
    st.subheader("Additional Information")
    st.markdown("This section can contain additional information or features.")
    
    # Iframe for external content
    st.markdown(
        """
<iframe src="https://www.ventusky.com/?p=18.9;80.9;5&l=satellite&w=dark" 
            width="100%" height="400" frameborder="0" style="border:none; overflow:hidden;" allowTransparency="true"></iframe>        """,
        unsafe_allow_html=True
    )

# To plot the weather prediction results on the map, add this function after your predictions
def plot_weather_on_map(lat, lon, predictions):
    # Plotting code using your .shp file goes here
    # You can use geopandas to read the shapefile and plot predictions
    pass
