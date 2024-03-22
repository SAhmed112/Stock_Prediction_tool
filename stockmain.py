import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import ParameterGrid
import plotly.graph_objects as go

def load_data(stock_name):
    data = yf.download(stock_name, start='2010-01-01', end=datetime.now().strftime('%Y-%m-%d'))
    data = data.dropna()
    return data

def preprocess_data(data):
    features = data.drop('Close', axis=1)
    target = data['Close']
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

from sklearn.ensemble import RandomForestRegressor

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    fig = go.Figure()
    
    # Add actual data
    fig.add_trace(go.Candlestick(x=y_test.index,
                                 open=y_test.values,
                                 high=y_test.values,
                                 low=y_test.values,
                                 close=y_test.values,
                                 name='Actual',
                                 increasing_line_color= 'cyan', decreasing_line_color= 'cyan'))
    
    # Add predicted data
    fig.add_trace(go.Candlestick(x=y_test.index,
                                 open=y_pred,
                                 high=y_pred,
                                 low=y_pred,
                                 close=y_pred,
                                 name='Predicted',
                                 increasing_line_color= 'orange', decreasing_line_color= 'orange'))
    
    fig.update_layout(title='Stock Price Prediction',
                      xaxis_title='Time',
                      yaxis_title='Stock Price')
    
    st.plotly_chart(fig)
    
    return mse

def predict_stock_price(target):
    target.index = target.index.to_period('D')
    param_grid = {'p': range(0, 3), 'd': range(0, 3), 'q': range(0, 3), 'P': range(0, 3), 'D': range(0, 3), 'Q': range(0, 3), 's': [12]}
    grid = ParameterGrid(param_grid)
    best_params = None
    lowest_aic = np.inf
    for params in grid:
        try:
            model_sarima = SARIMAX(target, order=(params['p'], params['d'], params['q']), seasonal_order=(params['P'], params['D'], params['Q'], params['s']))
            model_sarima_fit = model_sarima.fit(method_kwargs={"warn_convergence": False})
            if model_sarima_fit.aic < lowest_aic:
                best_params = params
                lowest_aic = model_sarima_fit.aic
        except:
            continue
    model_sarima = SARIMAX(target, order=(best_params['p'], best_params['d'], best_params['q']), seasonal_order=(best_params['P'], best_params['D'], best_params['Q'], best_params['s']))
    model_sarima_fit = model_sarima.fit()
    forecast = model_sarima_fit.forecast(steps=1)
    forecasted_value = forecast[0]
    return forecasted_value

def main():
    st.title('Stock Price Prediction')
    stock_name = st.text_input('Enter a stock name (e.g., HDFCBANK.NS):')
    if st.button('Predict'):
        data = load_data(stock_name)
        X_train, X_test, y_train, y_test = preprocess_data(data)
        model = train_model(X_train, y_train)
        mse = evaluate_model(model, X_test, y_test)
        st.write('Mean Squared Error:', mse)
        forecasted_value = predict_stock_price(data['Close'])
        st.write('Forecasted value:', forecasted_value)

if __name__ == '__main__':
    main()