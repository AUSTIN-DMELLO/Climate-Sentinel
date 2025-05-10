import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
import logging

st.set_page_config(page_title="Climate Sentinel", layout="wide")

st.title("Climate Sentinel")

# Sidebar for model selection
st.sidebar.title("Model Selection")
selected_model = st.sidebar.radio("Choose a model:", (
    "Renewable Energy Prediction", 
    "Forest Cover Loss Prediction",
    "Earthquake Prediction",
    "Tree Cover Loss and Afforestation"
))

# Function definitions for Renewable Energy Prediction
@st.cache_data
def load_renewable_data():
    df = pd.read_csv('data/dailypowerrenewable.csv')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    columns_to_convert = ['NR', 'WR', 'SR', 'ER', 'NER', 'All India']
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    return df

@st.cache_resource
def train_renewable_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Function definitions for Forest Cover Loss Prediction
@st.cache_data
def load_forest_data():
    data = pd.read_csv('data/subnationalforestcover.csv')
    return data

@st.cache_resource
def train_forest_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model, X_test, y_test

# Function definitions for Earthquake Prediction
@st.cache_data
def load_earthquake_data():
    df = pd.read_csv('data/cleaned_Indian_earthquake_data.csv')
    df['Magnitude'] = pd.to_numeric(df['Magnitude'], errors='coerce')
    df['Depth'] = pd.to_numeric(df['Depth'], errors='coerce')
    return df

# Function definitions for Tree Cover Loss and Afforestation
def load_and_preprocess_data(data):
    required_columns = ['country', 'tc_loss_ha_2003', 'tc_loss_ha_2023']
    if not all(col in data.columns for col in required_columns):
        st.error("Missing required columns in the dataset")
        return None

    le = LabelEncoder()
    data['country_encoded'] = le.fit_transform(data['country'].astype(str))
    return data

def prepare_features(data):
    features = ['country_encoded', 'threshold']
    current_year = 2023
    loss_columns = [col for col in data.columns if col.startswith('tc_loss_ha_') and int(col.split('_')[-1]) < current_year]
    features.extend(sorted(loss_columns))
    
    X = data[features]
    y = data[f'tc_loss_ha_{current_year}']
    
    return X, y, features

def train_and_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.success(f"Model Performance - MSE: {mse:.2f}, R-squared: {r2:.2f}")
    
    return rf_model

# Main application logic
if selected_model == "Renewable Energy Prediction":
    st.header("Renewable Energy Generation Prediction and Analysis")
    
    df = load_renewable_data()
    df_grouped = df.groupby('date')[['NR', 'WR', 'SR', 'ER', 'NER', 'All India']].sum().reset_index()

    prediction_years = st.sidebar.slider("Select number of years to predict:", 1, 10, 5)

    X = df_grouped.drop(columns=['date', 'All India'])
    y = df_grouped['All India']

    model, X_test, y_test = train_renewable_model(X, y)

    y_pred = model.predict(X_test)

    st.subheader("Model Performance")
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    # Add the rest of the Renewable Energy Prediction visualizations and analysis here
    last_date = df_grouped['date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=prediction_years*365, freq='D')
    future_input = pd.DataFrame({
        'NR': [df_grouped['NR'].mean()] * len(future_dates),
        'WR': [df_grouped['WR'].mean()] * len(future_dates),
        'SR': [df_grouped['SR'].mean()] * len(future_dates),
        'ER': [df_grouped['ER'].mean()] * len(future_dates),
        'NER': [df_grouped['NER'].mean()] * len(future_dates),
    })
    future_predictions = model.predict(future_input)
    
    st.subheader("Renewable Energy Generation Predictions")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_grouped['date'],
        y=df_grouped['All India'],
        mode='lines',
        name='Historical Data'
    ))
    
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=future_predictions,
        mode='lines',
        name='Predictions',
        line=dict(dash='dash')
    ))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Energy Generation (All India)",
        legend_title="Legend",
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Regional Contribution Over Time")
    
    fig_regional = px.area(df_grouped, x='date', y=['NR', 'WR', 'SR', 'ER', 'NER'],
                           labels={'value': 'Energy Generation', 'variable': 'Region'},
                           title='Regional Contribution to Total Energy Generation')
    
    st.plotly_chart(fig_regional, use_container_width=True)
    
    st.subheader("Seasonal Decomposition of Energy Generation")
    
    decomposition = seasonal_decompose(df_grouped.set_index('date')['All India'], model='additive', period=365)
    
    fig_seasonal = go.Figure()
    
    fig_seasonal.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, mode='lines', name='Observed'))
    fig_seasonal.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'))
    fig_seasonal.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'))
    fig_seasonal.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residual'))
    
    fig_seasonal.update_layout(
        title='Seasonal Decomposition of All India Energy Generation',
        xaxis_title='Date',
        yaxis_title='Energy Generation',
        height=800
    )
    
    st.plotly_chart(fig_seasonal, use_container_width=True)
    
    st.subheader("Monthly Average Energy Generation")
    
    df_grouped['month'] = df_grouped['date'].dt.month
    monthly_avg = df_grouped.groupby('month')['All India'].mean().reset_index()
    
    fig_monthly = px.bar(monthly_avg, x='month', y='All India',
                         labels={'All India': 'Average Energy Generation', 'month': 'Month'},
                         title='Average Monthly Energy Generation')
    
    fig_monthly.update_xaxes(tickvals=list(range(1, 13)), 
                             ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig_importance = go.Figure(go.Bar(
        x=feature_importance['importance'],
        y=feature_importance['feature'],
        orientation='h'
    ))
    
    fig_importance.update_layout(
        xaxis_title="Importance",
        yaxis_title="Feature",
        template="plotly_white"
    )
#--------------------------------------------------------
elif selected_model == "Forest Cover Loss Prediction":
    st.header("Forest Cover Loss Prediction for India")

    data = load_forest_data()

    # Preprocess the data
    le = LabelEncoder()
    data['country_encoded'] = le.fit_transform(data['country'])
    data['subnational1_encoded'] = le.fit_transform(data['subnational1'])
    data['subnational2_encoded'] = le.fit_transform(data['subnational2'])

    features = ['country_encoded', 'subnational1_encoded', 'subnational2_encoded', 'threshold', 'area_ha', 'extent_2000_ha', 'extent_2010_ha', 'gain_2000-2020_ha'] + [f'tc_loss_ha_{year}' for year in range(2001, 2021)]

    X = data[features]
    y = data['tc_loss_ha_2023']

    rf_model, X_test, y_test = train_forest_model(X, y)

    # Add the rest of the Forest Cover Loss Prediction visualizations and analysis here
    # Streamlit app
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

#------------------------------------------------

elif selected_model == "Earthquake Prediction":
    st.header('Earthquake Visualization and Prediction Dashboard')
    
    df = load_earthquake_data()
    
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

elif selected_model == "Tree Cover Loss and Afforestation":
    st.header("🌲 Tree Cover Loss Prediction and Afforestation Suggestions")
    
    # File upload for CSV data
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data = load_and_preprocess_data(data)

        if data is not None:
            X, y, features = prepare_features(data)
            model = train_and_evaluate_model(X, y)
            
            # Add the rest of the Tree Cover Loss and Afforestation analysis here
# Function to suggest afforestation based on predictions
def suggest_afforestation(future_data):
    current_year = 2023
    future_loss_columns = [col for col in future_data.columns if col.startswith('tc_loss_ha_') and int(col.split('_')[-1]) > current_year]
    
    future_data['total_future_loss'] = future_data[future_loss_columns].sum(axis=1)
    
    high_loss_areas = future_data.nlargest(10, 'total_future_loss')
    
    st.subheader("🌳 Top 10 Areas for Afforestation Efforts:")
    
    suggestions = []
    
    for index, row in high_loss_areas.iterrows():
        country_info = {
            "country": row['country'],
            "threshold": row['threshold'],
            "predicted_loss": row['total_future_loss'],
            "suggested_afforestation": row['total_future_loss'] * 1.2,
            "recommendations": ""
        }
        
        # Provide different suggestions based on the threshold value
        if row['threshold'] < 20:
            country_info["recommendations"] = "Immediate intervention needed."
        elif 20 <= row['threshold'] < 50:
            country_info["recommendations"] = "Targeted afforestation programs."
        else:
            country_info["recommendations"] = "Maintain existing forest cover."
        
        suggestions.append(country_info)

        # Display suggestions for each area with improved formatting
        st.markdown(f"### **Area:** {row['country']} 🌍")
        st.markdown(f"**Tree Cover Threshold:** {row['threshold']}%")
        st.markdown(f"**Predicted Loss Over Next {len(future_loss_columns)} Years:** {row['total_future_loss']:.2f} ha")
        st.markdown(f"**Suggested Afforestation:** {row['total_future_loss'] * 1.2:.2f} ha")
        
        # Use a colored box for recommendations
        st.markdown("<div style='background-color: #f0f8ff; padding: 10px; border-radius: 5px;'>", unsafe_allow_html=True)
        st.markdown(f"**Recommendations:** {country_info['recommendations']}")
        st.markdown("</div>", unsafe_allow_html=True)

    return suggestions

# Function to visualize data
def visualize_suggestions(suggestions):
    countries = [s["country"] for s in suggestions]
    predicted_losses = [s["predicted_loss"] for s in suggestions]
    suggested_afforestation = [s["suggested_afforestation"] for s in suggestions]

    # Bar chart visualization
    fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.35
    index_bar = np.arange(len(countries))

    bars1 = ax_bar.bar(index_bar, predicted_losses, bar_width, label='Predicted Loss (ha)', color='orange')
    bars2 = ax_bar.bar(index_bar + bar_width, suggested_afforestation, bar_width, label='Suggested Afforestation (ha)', color='green')

    ax_bar.set_xlabel('Countries')
    ax_bar.set_ylabel('Area (ha)')
    ax_bar.set_title('Predicted Tree Cover Loss vs Suggested Afforestation')
    ax_bar.set_xticks(index_bar + bar_width / 2)
    ax_bar.set_xticklabels(countries)
    ax_bar.legend()

    st.pyplot(fig_bar)

    # Pie chart visualization
    total_predicted_loss = sum(predicted_losses)
    total_suggested_afforestation = sum(suggested_afforestation)

    labels_pie = ['Total Predicted Loss', 'Total Suggested Afforestation']
    sizes_pie = [total_predicted_loss, total_suggested_afforestation]
    
    fig_pie, ax_pie = plt.subplots(figsize=(8, 8))
    
    ax_pie.pie(sizes_pie, labels=labels_pie, autopct='%1.1f%%', startangle=90)
    
    ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    
    st.subheader("📊 Proportion of Predicted Loss vs Suggested Afforestation")
    st.pyplot(fig_pie)

st.sidebar.info("Climate Sentinel combines four powerful models to analyze renewable energy generation, forest cover loss, earthquake predictions, and tree cover loss with afforestation suggestions. Use the radio buttons above to switch between the models.")