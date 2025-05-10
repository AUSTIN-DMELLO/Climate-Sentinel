import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('data/cleaned_Indian_earthquake_data.csv')
    df['Magnitude'] = pd.to_numeric(df['Magnitude'], errors='coerce')
    df['Depth'] = pd.to_numeric(df['Depth'], errors='coerce')
    return df

df = load_data()

st.title('Earthquake Visualization and Prediction Dashboard')

# Sidebar for filtering
st.sidebar.header('Filters')
min_magnitude = st.sidebar.slider('Minimum Magnitude', float(df['Magnitude'].min()), float(df['Magnitude'].max()), float(df['Magnitude'].min()))
depth_range = st.sidebar.slider('Depth Range (km)', float(df['Depth'].min()), float(df['Depth'].max()), (float(df['Depth'].min()), float(df['Depth'].max())))

# Filter the dataframe based on magnitude and depth
filtered_df = df[(df['Magnitude'] >= min_magnitude) & 
                 (df['Depth'] >= depth_range[0]) & 
                 (df['Depth'] <= depth_range[1])]

# Map
st.subheader('Earthquake Locations')
m = folium.Map(location=[20.5937, 78.9629], zoom_start=4)

for idx, row in filtered_df.iterrows():
    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=row['Magnitude'] * 2,
        color='red' if row['Magnitude'] >= 4 else 'orange',
        fill=True,
        fill_opacity=0.6,
        popup=f"{row['Location']}, Magnitude: {row['Magnitude']}, Depth: {row['Depth']} km",
    ).add_to(m)

folium_static(m)

# Magnitude Distribution
st.subheader('Magnitude Distribution')
fig_mag = px.histogram(filtered_df, x='Magnitude', nbins=20, title='Distribution of Earthquake Magnitudes')
st.plotly_chart(fig_mag)

# Depth vs Magnitude
st.subheader('Depth vs Magnitude')
fig_depth = px.scatter(filtered_df, x='Depth', y='Magnitude', hover_data=['Location'], 
                      title='Earthquake Depth vs Magnitude')
st.plotly_chart(fig_depth)

# Top 10 strongest earthquakes
st.subheader('Top 10 Strongest Earthquakes')
top_10 = filtered_df.nlargest(10, 'Magnitude')
st.table(top_10[['Magnitude', 'Depth', 'Location']])

# Prediction Section
st.subheader('Predict Earthquake Magnitude')

# Machine Learning Model (Random Forest Regressor)
# Features: Latitude, Longitude, Depth; Target: Magnitude
X = df[['Latitude', 'Longitude', 'Depth']].dropna()
y = df['Magnitude'].dropna()

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"Model Mean Squared Error: {mse:.2f}")

# User input for prediction
st.write('Enter the latitude, longitude, and depth to predict earthquake magnitude:')
lat_input = st.number_input('Latitude', value=20.5937)
long_input = st.number_input('Longitude', value=78.9629)
depth_input = st.number_input('Depth (km)', value=10.0)

# Prediction button
if st.button('Predict Magnitude'):
    input_data = pd.DataFrame({'Latitude': [lat_input], 'Longitude': [long_input], 'Depth': [depth_input]})
    predicted_magnitude = model.predict(input_data)[0]
    st.write(f'Predicted Earthquake Magnitude: {predicted_magnitude:.2f}')

# Download filtered data
st.subheader('Download Filtered Data')
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="Download filtered data as CSV",
    data=csv,
    file_name="filtered_earthquake_data.csv",
    mime="text/csv",
)