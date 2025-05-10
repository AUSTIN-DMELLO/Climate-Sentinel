import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(page_title="Renewable Energy Prediction", layout="wide")

st.title("Renewable Energy Generation Prediction and Analysis")

@st.cache_data
def load_data():
    df = pd.read_csv('data/dailypowerrenewable.csv')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    columns_to_convert = ['NR', 'WR', 'SR', 'ER', 'NER', 'All India']
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    return df

df = load_data()

df_grouped = df.groupby('date')[['NR', 'WR', 'SR', 'ER', 'NER', 'All India']].sum().reset_index()

st.sidebar.header("User Input Features")
prediction_years = st.sidebar.slider("Select number of years to predict:", 1, 10, 5)

X = df_grouped.drop(columns=['date', 'All India'])
y = df_grouped['All India']

@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model(X, y)

y_pred = model.predict(X_test)

st.subheader("Model Performance")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

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

st.plotly_chart(fig_importance, use_container_width=True)

