# build_training_dataset_v6.py
import pandas as pd
import numpy as np
import pandas_ta as ta
import os
from datetime import datetime
from math import log, sqrt, exp
from scipy.stats import norm

# ---------- 帮助函数：BS模型计算Delta/Gamma/IV ----------
def bs_delta(S, K, T, r, sigma, option_type):
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    if option_type.lower() == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

def bs_gamma(S, K, T, r, sigma):
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    return norm.pdf(d1) / (S * sigma * sqrt(T))

# 简单隐含波动率反推（牛逼的做法要用优化器）
def implied_volatility(price, S, K, T, r, option_type):
    # 初始猜测
    sigma = 0.2
    for i in range(50):
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        if option_type.lower() == "call":
            price_calc = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        else:
            price_calc = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        vega = S * norm.pdf(d1) * sqrt(T)
        if vega == 0:
            break
        diff = price_calc - price
        if abs(diff) < 1e-5:
            break
        sigma -= diff / vega
        if sigma <= 0:
            sigma = 1e-5
    return sigma

# ---------- 文件路径 ----------
base_dir = "E:/hhy/trading_ai"
# 文件路径
TRADES_FILE = base_dir+"/data/processed/trades_clean_v2.csv"
STOCK_DIR = base_dir+"/data/cache/stocks_day_k"
INDEX_DIR = base_dir+"/data/cache/indexes_day_k"
OUTPUT_FILE = base_dir+"/data/processed/training_dataset_v5.csv"

os.makedirs("output", exist_ok=True)

# ---------- 读取交易清单 ----------
trades = pd.read_csv(TRADES_FILE)
rows = []

for _, trade in trades.iterrows():
    symbol = trade["underlying"]
    stock_file = f"{STOCK_DIR}/{symbol}.csv"

    if not os.path.exists(stock_file):
        continue

    stock = pd.read_csv(stock_file)
    # 强制float
    for col in ["Open","High","Low","Close","Volume"]:
        if col in stock.columns:
            stock[col] = pd.to_numeric(stock[col], errors="coerce")
    stock = stock.dropna(subset=["Open","High","Low","Close"])
    if stock.empty:
        continue

    stock["Date"] = pd.to_datetime(stock["Date"])
    open_time = pd.to_datetime(trade["open_time"])
    stock = stock[stock["Date"] <= open_time]
    if len(stock) < 60:
        continue

    # ---------- 技术指标 ----------
    stock["MA5"] = ta.sma(stock["Close"],5)
    stock["MA20"] = ta.sma(stock["Close"],20)
    stock["MA60"] = ta.sma(stock["Close"],60)
    stock["RSI"] = ta.rsi(stock["Close"],14)
    stock["ATR"] = ta.atr(stock["High"], stock["Low"], stock["Close"],14)
    stock["VOL20"] = stock["Close"].pct_change().rolling(20).std()
    stock["MOM10"] = stock["Close"].pct_change(10)

    last = stock.iloc[-1]
    price = last["Close"]
    strike = trade["strike"]
    T = max(trade["dte"]/365, 1e-5)  # 剩余期限，年化
    r = 0.03  # 无风险利率假设

    # ---------- 期权特征 ----------
    option_type = trade["option_type"].lower()
    mid_price = trade["avg_open_price"]  # 或使用市价中值

    try:
        IV = implied_volatility(mid_price, price, strike, T, r, option_type)
        Delta = bs_delta(price, strike, T, r, IV, option_type)
        Gamma = bs_gamma(price, strike, T, r, IV)
    except:
        IV, Delta, Gamma = np.nan, np.nan, np.nan

    # ---------- 市场指数特征 ----------
    SPY_trend = np.nan
    VIX_level = np.nan
    spy_file = f"{INDEX_DIR}/SPY.csv"
    vix_file = f"{INDEX_DIR}/VIX.csv"
    if os.path.exists(spy_file):
        spy = pd.read_csv(spy_file)
        spy["Date"] = pd.to_datetime(spy["Date"])
        for col in ["Open","High","Low","Close","Volume"]:
            if col in spy.columns:
                spy[col] = pd.to_numeric(spy[col], errors="coerce")
        spy = spy[spy["Date"] <= open_time]
        if len(spy.dropna(subset=["Open","Close"]))>0:
            spy_last = spy.dropna(subset=["Open","Close"]).iloc[-1]
            SPY_trend = spy_last["Close"] - spy_last["Open"]
    if os.path.exists(vix_file):
        vix = pd.read_csv(vix_file)
        vix["Date"] = pd.to_datetime(vix["Date"])
        for col in ["Open","High","Low","Close","Volume"]:
            if col in vix.columns:
                vix[col] = pd.to_numeric(vix[col], errors="coerce")
        vix = vix[vix["Date"] <= open_time]
        if len(vix.dropna(subset=["Close"]))>0:
            VIX_level = vix.dropna(subset=["Close"]).iloc[-1]["Close"]

    # ---------- 构建训练行 ----------
    row = {
        "symbol": trade["symbol"],
        "underlying": symbol,
        "option_type": option_type,
        "stock_price": price,
        "strike": strike,
        "moneyness": price/strike,
        "distance_to_strike": (price-strike)/strike,
        "log_moneyness": np.log(price/strike),
        "dte": trade["dte"],
        "holding_minutes": trade["holding_minutes"],
        "MA5": last["MA5"],
        "MA20": last["MA20"],
        "MA60": last["MA60"],
        "trend_strength": last["MA5"]-last["MA20"],
        "RSI": last["RSI"],
        "ATR": last["ATR"],
        "volatility20": last["VOL20"],
        "momentum10": last["MOM10"],
        "SPY_trend": SPY_trend,
        "VIX_level": VIX_level,
        "avg_open_price": trade["avg_open_price"],
        "pnl": trade["pnl"],
        "pnl_pct": trade["pnl_pct"],
        "IV": IV,
        "Delta": Delta,
        "Gamma": Gamma,
        "label": trade["label"],
    }
    rows.append(row)

# ---------- 输出训练数据 ----------
dataset = pd.DataFrame(rows)
dataset.to_csv(OUTPUT_FILE,index=False)
print("training dataset size:",len(dataset))
print("saved to:", OUTPUT_FILE)