# build_training_dataset_pipeline.py
import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime
from math import log, sqrt, exp
from scipy.stats import norm
import yfinance as yf  # 用于下载股票数据，可选


from light_gbm.train_ai import base_dir

# ---------- BS模型计算 ----------
def bs_delta(S,K,T,r,sigma,option_type):
    d1 = (log(S/K)+(r+0.5*sigma**2)*T)/(sigma*sqrt(T))
    if option_type.lower()=="call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1)-1

def bs_gamma(S,K,T,r,sigma):
    d1 = (log(S/K)+(r+0.5*sigma**2)*T)/(sigma*sqrt(T))
    return norm.pdf(d1)/(S*sigma*sqrt(T))

def implied_volatility(price,S,K,T,r,option_type):
    sigma = 0.2
    for i in range(50):
        d1 = (log(S/K)+(r+0.5*sigma**2)*T)/(sigma*sqrt(T))
        d2 = d1-sigma*sqrt(T)
        if option_type.lower()=="call":
            price_calc = S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
        else:
            price_calc = K*exp(-r*T)*norm.cdf(-d2)-S*norm.cdf(-d1)
        vega = S*norm.pdf(d1)*sqrt(T)
        if vega==0: break
        diff = price_calc-price
        if abs(diff)<1e-5: break
        sigma -= diff/vega
        if sigma<=0: sigma=1e-5
    return sigma

# ---------- 配置 ----------

# 文件路径
TRADES_FILE = base_dir+"/data/processed/trades_clean_v2.csv"
STOCK_DIR = base_dir+"/data/cache/stocks_day_k"
INDEX_DIR = base_dir+"/data/cache/indexes_day_k"
OUTPUT_FILE = base_dir+"/data/processed/training_dataset_latest.csv"

os.makedirs("output",exist_ok=True)
os.makedirs(STOCK_DIR,exist_ok=True)
os.makedirs(INDEX_DIR,exist_ok=True)

# ---------- 下载/更新股票行情（可选） ----------
def update_stock_cache(symbol,start="2023-01-01"):
    stock_file = f"{STOCK_DIR}/{symbol}.csv"
    if os.path.exists(stock_file):
        df_old = pd.read_csv(stock_file)
        last_date = pd.to_datetime(df_old["Date"]).max()
        if last_date>=pd.Timestamp.today()-pd.Timedelta(days=1):
            return  # 已经最新
        start_date = (last_date+pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start_date = start
    try:
        df = yf.download(symbol,start=start_date)
        if df.empty: return
        df.reset_index(inplace=True)
        df.rename(columns={"Date":"Date","Adj Close":"Close"},inplace=True)
        if os.path.exists(stock_file):
            df_old = pd.read_csv(stock_file)
            df = pd.concat([df_old,df],ignore_index=True)
        df.to_csv(stock_file,index=False)
    except Exception as e:
        print("failed to update",symbol,e)

# ---------- 更新所有股票 ----------
trades = pd.read_csv(TRADES_FILE)
symbols = trades["underlying"].unique()
for s in symbols:
    update_stock_cache(s)

# ---------- 检查已有训练集 ----------
if os.path.exists(OUTPUT_FILE):
    dataset_old = pd.read_csv(OUTPUT_FILE)
    processed_keys = set(dataset_old["symbol"]+"_"+dataset_old["holding_minutes"].astype(str))
    rows=[]
    append_mode=True
else:
    dataset_old=None
    processed_keys=set()
    rows=[]
    append_mode=False

# ---------- 生成训练行 ----------
for _,trade in trades.iterrows():
    key=trade["symbol"]+"_"+str(trade["holding_minutes"])
    if key in processed_keys: continue

    symbol=trade["underlying"]
    stock_file=f"{STOCK_DIR}/{symbol}.csv"
    if not os.path.exists(stock_file): continue

    stock=pd.read_csv(stock_file)
    for col in ["Open","High","Low","Close","Volume"]:
        if col in stock.columns:
            stock[col]=pd.to_numeric(stock[col],errors="coerce")
    stock=stock.dropna(subset=["Open","High","Low","Close"])
    if stock.empty: continue
    stock["Date"]=pd.to_datetime(stock["Date"])
    open_time=pd.to_datetime(trade["open_time"])
    stock=stock[stock["Date"]<=open_time]
    if len(stock)<60: continue

    # 技术指标
    stock["MA5"]=ta.sma(stock["Close"],5)
    stock["MA20"]=ta.sma(stock["Close"],20)
    stock["MA60"]=ta.sma(stock["Close"],60)
    stock["RSI"]=ta.rsi(stock["Close"],14)
    stock["ATR"]=ta.atr(stock["High"],stock["Low"],stock["Close"],14)
    stock["VOL20"]=stock["Close"].pct_change().rolling(20).std()
    stock["MOM10"]=stock["Close"].pct_change(10)

    last=stock.iloc[-1]
    price=last["Close"]
    strike=trade["strike"]
    T=max(trade["dte"]/365,1e-5)
    r=0.03
    option_type=trade["option_type"].lower()
    mid_price=trade["avg_open_price"]

    try:
        IV=implied_volatility(mid_price,price,strike,T,r,option_type)
        Delta=bs_delta(price,strike,T,r,IV,option_type)
        Gamma=bs_gamma(price,strike,T,r,IV)
    except:
        IV,Delta,Gamma=np.nan,np.nan,np.nan

    # 市场指数
    SPY_trend=np.nan
    VIX_level=np.nan
    spy_file=f"{INDEX_DIR}/SPY.csv"
    vix_file=f"{INDEX_DIR}/VIX.csv"
    if os.path.exists(spy_file):
        spy=pd.read_csv(spy_file)
        spy["Date"]=pd.to_datetime(spy["Date"])
        for col in ["Open","High","Low","Close","Volume"]:
            if col in spy.columns:
                spy[col]=pd.to_numeric(spy[col],errors="coerce")
        spy=spy[spy["Date"]<=open_time]
        if len(spy.dropna(subset=["Open","Close"]))>0:
            spy_last=spy.dropna(subset=["Open","Close"]).iloc[-1]
            SPY_trend=spy_last["Close"]-spy_last["Open"]
    if os.path.exists(vix_file):
        vix=pd.read_csv(vix_file)
        vix["Date"]=pd.to_datetime(vix["Date"])
        for col in ["Open","High","Low","Close","Volume"]:
            if col in vix.columns:
                vix[col]=pd.to_numeric(vix[col],errors="coerce")
        vix=vix[vix["Date"]<=open_time]
        if len(vix.dropna(subset=["Close"]))>0:
            VIX_level=vix.dropna(subset=["Close"]).iloc[-1]["Close"]

    row={
        "symbol":trade["symbol"],
        "underlying":symbol,
        "option_type":option_type,
        "stock_price":price,
        "strike":strike,
        "moneyness":price/strike,
        "distance_to_strike":(price-strike)/strike,
        "log_moneyness":np.log(price/strike),
        "dte":trade["dte"],
        "holding_minutes":trade["holding_minutes"],
        "MA5":last["MA5"],
        "MA20":last["MA20"],
        "MA60":last["MA60"],
        "trend_strength":last["MA5"]-last["MA20"],
        "RSI":last["RSI"],
        "ATR":last["ATR"],
        "volatility20":last["VOL20"],
        "momentum10":last["MOM10"],
        "SPY_trend":SPY_trend,
        "VIX_level":VIX_level,
        "avg_open_price":trade["avg_open_price"],
        "pnl":trade["pnl"],
        "pnl_pct":trade["pnl_pct"],
        "IV":IV,
        "Delta":Delta,
        "Gamma":Gamma,
        "label":trade["label"]
    }
    rows.append(row)

# 合并已有数据
dataset_new=pd.DataFrame(rows)
if append_mode:
    dataset=pd.concat([dataset_old,dataset_new],ignore_index=True)
else:
    dataset=dataset_new

dataset.to_csv(OUTPUT_FILE,index=False)
print("training dataset size:",len(dataset))
print("saved to:",OUTPUT_FILE)