import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data
def load_data():
    data = pd.read_csv('data/subnationalforestcover.csv')
    return data

data = load_data()

le = LabelEncoder()
data['country_encoded'] = le.fit_transform(data['country'])
data['subnational1_encoded'] = le.fit_transform(data['subnational1'])
data['subnational2_encoded'] = le.fit_transform(data['subnational2'])

features = ['country_encoded', 'subnational1_encoded', 'subnational2_encoded', 'threshold', 'area_ha', 'extent_2000_ha', 'extent_2010_ha', 'gain_2000-2020_ha'] + [f'tc_loss_ha_{year}' for year in range(2001, 2021)]

X = data[features]
y = data['tc_loss_ha_2023']

@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model, X_test, y_test

rf_model, X_test, y_test = train_model()

st.title('Forest Cover Loss Prediction for India')

st.write("""
This application predicts forest cover loss across India using machine learning. 
Explore predictions for different years and visualize the results!
""")

# Model performance
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.header('Model Performance')
col1, col2 = st.columns(2)
col1.metric("Mean Squared Error", f"{mse:.2f}")
col2.metric("R-squared Score", f"{r2:.2f}")

# Prediction function
def predict_forest_cover_loss_india(year):
    input_data = data[features].values
    years_diff = year - 2023
    for i in range(years_diff):
        input_data[:, -(i+1)] = rf_model.predict(input_data)
    predicted_losses = rf_model.predict(input_data)
    total_loss = np.sum(predicted_losses)
    
    state_losses = data.groupby('subnational1')['subnational2'].count().reset_index()
    state_losses['predicted_loss'] = data.groupby('subnational1').apply(lambda x: np.sum(predicted_losses[x.index])).values
    state_losses = state_losses.sort_values('predicted_loss', ascending=False)
    
    return total_loss, state_losses

# User input
st.header('Predict Forest Cover Loss')
year = st.slider('Select a year to predict', min_value=2024, max_value=2030, value=2025)

# Make prediction
total_loss, state_losses = predict_forest_cover_loss_india(year)

# Display results
st.subheader(f'Predicted Forest Cover Loss for India in {year}')
st.metric("Total Predicted Loss", f"{total_loss:.2f} ha")

# Visualizations
st.subheader('Visualizations')

# Bar plot
st.write("Predicted Forest Cover Loss by State")
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='subnational1', y='predicted_loss', data=state_losses, ax=ax)
plt.xticks(rotation=45, ha='right')
plt.xlabel('State')
plt.ylabel('Predicted Loss (ha)')
st.pyplot(fig)

# Pie chart
st.write("Top 5 States - Predicted Forest Cover Loss Share")
fig, ax = plt.subplots(figsize=(10, 8))
top_5_states = state_losses.head()
plt.pie(top_5_states['predicted_loss'], labels=top_5_states['subnational1'], autopct='%1.1f%%')
plt.axis('equal')
st.pyplot(fig)

# Multi-year prediction
st.header('Multi-Year Prediction')
start_year = st.number_input('Start Year', min_value=2024, max_value=2029, value=2024)
end_year = st.number_input('End Year', min_value=start_year+1, max_value=2030, value=2030)

if st.button('Generate Multi-Year Prediction'):
    years = range(start_year, end_year + 1)
    total_losses = []
    state_losses_over_time = []

    for year in years:
        total_loss, state_losses = predict_forest_cover_loss_india(year)
        total_losses.append(total_loss)
        state_losses_over_time.append(state_losses)

    # Total loss trend
    st.subheader('Total Predicted Forest Cover Loss Trend')
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(years, total_losses, marker='o')
    plt.title('Predicted Total Forest Cover Loss in India Over Years')
    plt.xlabel('Year')
    plt.ylabel('Predicted Loss (ha)')
    plt.grid(True)
    st.pyplot(fig)

    # Stacked area chart for top 5 states
    st.subheader('Top 5 States - Predicted Forest Cover Loss Trends')
    top_5_states = state_losses_over_time[-1].nlargest(5, 'predicted_loss')['subnational1']
    stacked_data = pd.DataFrame({year: df.set_index('subnational1')['predicted_loss'] 
                                 for year, df in zip(years, state_losses_over_time)})
    stacked_data = stacked_data.loc[top_5_states]

    fig, ax = plt.subplots(figsize=(12, 6))
    stacked_data.T.plot(kind='area', stacked=True, ax=ax)
    plt.title('Predicted Forest Cover Loss Trends for Top 5 States')
    plt.xlabel('Year')
    plt.ylabel('Predicted Loss (ha)')
    plt.legend(title='State', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)

st.write("""
### About this Project
