import os
import shutil
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from tvDatafeed import TvDatafeed, Interval

tv = TvDatafeed()

# ตั้งค่าหัวข้อ Streamlit
st.title("Crypto Sentiment Analysis with EMA Strategy")

# โหลดข้อมูล
st.sidebar.header("Upload CSV Files")
prices_file = st.sidebar.file_uploader("Upload Prices CSV", type=["csv"])
sentiment_file = st.sidebar.file_uploader("Upload Sentiment CSV", type=["csv"])


def load_crypto_data(data):
    
    data.columns = data.columns.get_level_values(0)
    df = data.copy().reset_index()
    # คำนวณ EMA 7 และ EMA 14
    df['ema14'] = df.ta.ema(length=14)
    df['ema7'] = df.ta.ema(length=7)
    df.rename(columns={"Date": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()  # ตัดเวลาออก
    return df


def merge_data(df_prices, df_sentiment):
    df_prices.rename(columns={"Date": "date"}, inplace=True)
    df_prices["date"] = pd.to_datetime(df_prices["date"]).dt.date
    df_sentiment["date"] = pd.to_datetime(df_sentiment["date"]).dt.date
    return pd.merge(df_prices, df_sentiment, on="date", how="left").fillna(0)

def generate_signal(df):
    df = df.copy()
    df["signal_buy"] = 0
    high_negative_days = df[df["Positive - Negative"] < -30].index
    for idx in high_negative_days:
        future_days = df.loc[idx+1:]
        positive_day = future_days[future_days["Positive - Negative"] > 0]
        if not positive_day.empty:
            first_positive_idx = positive_day.index[0]
            ema_check_days = df.loc[first_positive_idx:first_positive_idx+10]
            for i in range(1, len(ema_check_days)):
                if (ema_check_days["ema7"].iloc[i-1] < ema_check_days["ema14"].iloc[i-1] and
                    ema_check_days["ema7"].iloc[i] > ema_check_days["ema14"].iloc[i]):
                    df.loc[ema_check_days.index[i], "signal_buy"] = 1
                    break
    return df

def calculate_returns(df):
    for days in [1, 3, 5, 10, 15, 30]:
        df[f"return_{days}d"] = None
    signal_days = df[df["signal_buy"] == 1].index
    for idx in signal_days:
        entry_idx = idx + 1
        if entry_idx < len(df):
            entry_price = df.loc[entry_idx, "Open"]
            for days in [1, 3, 5, 10, 15, 30]:
                future_idx = entry_idx + days
                if future_idx < len(df):
                    future_price = df.loc[future_idx, "Close"]
                    return_value = (future_price - entry_price) / entry_price
                    df.loc[idx, f"return_{days}d"] = return_value
    return df

def plot_trading_strategy(df):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["date"], y=df["Close"], mode='lines', name='Close Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ema7"], mode='lines', name='EMA 7', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ema14"], mode='lines', name='EMA 14', line=dict(color='purple', dash='dot')))
    fig.add_trace(go.Bar(x=df["date"], y=df["Positive - Negative"], name='Sentiment', marker_color='orange', opacity=0.5), secondary_y=True)
    signal_buy_df = df[df["signal_buy"] == 1]
    fig.add_trace(go.Scatter(x=signal_buy_df["date"], y=signal_buy_df["Close"], mode='markers', name='Signal Buy',
                             marker=dict(symbol='triangle-up', size=10, color='lime', line=dict(color='black', width=1))))
    fig.update_layout(title="Close Price, EMA 7, EMA 14, Sentiment and Signal Buy", hovermode="x unified", template="plotly_white")
    return fig

def plot_returns1(df):
    df_plot = df[df["signal_buy"] == 1][["date", "return_1d", "return_3d", "return_5d", "return_10d", "return_15d", "return_30d"]].copy()
    df_plot["date"] = pd.to_datetime(df_plot["date"])
    df_melted = df_plot.melt(id_vars=["date"], var_name="Return Period", value_name="Return")
    fig = px.bar(df_melted, x="date", y="Return", color="Return Period", title="Interactive Return After Signal Buy", labels={"Return": "Return (%)", "date": "Date"}, color_discrete_sequence=px.colors.qualitative.Set1)
    return fig

if prices_file and sentiment_file:

    df_prices = pd.read_csv(prices_file)
    df_sentiment = pd.read_csv(sentiment_file)
    df_prices=load_crypto_data(df_prices)
    df_merged = merge_data(df_prices, df_sentiment)
    df_merged = generate_signal(df_merged)
    df_merged = calculate_returns(df_merged)
    
    st.write("## Merged Data Preview")
    st.dataframe(df_merged.head())
    
    st.write("## Trading Strategy Visualization")
    st.plotly_chart(plot_trading_strategy(df_merged))
    
    st.write("## Returns After Signal Buy")
    st.plotly_chart(plot_returns1(df_merged))
    
    st.success("Analysis Completed! ✅")