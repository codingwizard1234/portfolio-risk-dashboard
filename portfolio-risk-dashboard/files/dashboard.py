import streamlit as st
import pandas as pd
import plotly.express as px

from data_fetch import fetch_data
from metrics import (
    cumulative_returns,
    volatility,
    sharpe_ratio,
    max_drawdown,
    value_at_risk,
    portfolio_return,
)
from scenario_analysis import scenario_performance
from monte_carlo import monte_carlo_simulation
from predict_volatility import predict_volatility_arima
from lstm_volatility import train_lstm_volatility

# --------------------------
# Streamlit Dashboard
# --------------------------
st.title("📊 Portfolio & Risk Analytics Dashboard")

# Inputs
tickers_input = st.text_input("Enter stock tickers (comma separated):")
weights_input = st.text_input("Enter portfolio weights (comma separated):")
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))

# Store in session state
if "df" not in st.session_state:
    st.session_state.df = None
if "tickers" not in st.session_state:
    st.session_state.tickers = []
if "weights" not in st.session_state:
    st.session_state.weights = []

# --------------------------
# Portfolio Analysis
# --------------------------
if st.button("Analyze Portfolio"):
    if not tickers_input or not weights_input:
        st.error("⚠️ Please enter both tickers and weights.")
    else:
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        try:
            weights = [float(w) for w in weights_input.split(",")]
        except ValueError:
            st.error("⚠️ Weights must be numbers separated by commas.")
            st.stop()

        if len(tickers) != len(weights):
            st.error("⚠️ Number of tickers and weights must match.")
            st.stop()

        # Save in session state
        st.session_state.tickers = tickers
        st.session_state.weights = weights
        st.session_state.df = fetch_data(tickers, start=start_date)

        df = st.session_state.df

        # Portfolio Metrics
        st.subheader("Portfolio Metrics")
        st.write(f"Volatility: {volatility(df, weights):.4f}")
        st.write(f"Sharpe Ratio: {sharpe_ratio(df, weights):.4f}")
        st.write(f"Max Drawdown: {max_drawdown(df, weights):.2%}")
        st.write(f"Value-at-Risk (5%): {value_at_risk(df, weights):.2%}")

        # Plots
        cum_ret = cumulative_returns(df, weights)
        fig = px.line(cum_ret, title="Cumulative Portfolio Returns")
        st.plotly_chart(fig)

        port_ret = portfolio_return(df, weights).cumsum()
        fig2 = px.line(port_ret, title="Portfolio Return Over Time")
        st.plotly_chart(fig2)

# --------------------------
# Scenario Analysis
# --------------------------
st.subheader("Scenario Analysis")
if st.button("Run Scenario Analysis"):
    if st.session_state.df is None:
        st.error("⚠️ Please run portfolio analysis first.")
    else:
        for event, (s, e) in {
            "2008 Crash": ("2008-09-01", "2009-03-01"),
            "COVID Crash": ("2020-02-15", "2020-03-30"),
        }.items():
            perf = scenario_performance(
                st.session_state.tickers, st.session_state.weights, s, e
            )
            st.write(f"{event}: {perf:.2%}")

# --------------------------
# Monte Carlo Simulation
# --------------------------
st.subheader("Monte Carlo Simulation")
if st.button("Run Monte Carlo"):
    if st.session_state.df is None:
        st.error("⚠️ Please run portfolio analysis first.")
    else:
        sim_results = monte_carlo_simulation(
            st.session_state.df, st.session_state.weights
        )
        fig_mc = px.histogram(
            sim_results, nbins=50, title="Monte Carlo Simulated Returns"
        )
        st.plotly_chart(fig_mc)
        st.write(f"5% worst-case return: {sim_results.quantile(0.05):.2%}")

# --------------------------
# Volatility Prediction
# --------------------------
st.subheader("Volatility Prediction")
if st.button("Predict Volatility"):
    if st.session_state.df is None:
        st.error("⚠️ Please run portfolio analysis first.")
    else:
        returns = st.session_state.df.pct_change().dropna()
        arima_forecast = predict_volatility_arima(returns)
        lstm_forecast = train_lstm_volatility(returns)

        st.write("ARIMA Forecast (next 5 days):", arima_forecast)
        st.write(f"LSTM Next-Day Volatility: {lstm_forecast:.4f}")
