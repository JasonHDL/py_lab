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
def bs_delta(S, K, T, r, sigma, option_type):
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    if option_type.lower() == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1


def bs_gamma(S, K, T, r, sigma):
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return norm.pdf(d1) / (S * sigma * sqrt(T))


def implied_volatility(price, S, K, T, r, option_type):
    sigma = 0.2
    for _ in range(50):
        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
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


# ---------- Tiger 1分钟线 -> 当日(截至下单时刻)日K ----------
def _normalize_tiger_timeline(timeline_obj):
    """把 Tiger timeline 返回结果尽量归一化为 DataFrame。"""
    if timeline_obj is None:
        return pd.DataFrame()

    if isinstance(timeline_obj, pd.DataFrame):
        return timeline_obj.copy()

    if isinstance(timeline_obj, dict):
        if "items" in timeline_obj and isinstance(timeline_obj["items"], list):
            return pd.DataFrame(timeline_obj["items"])
        return pd.DataFrame([timeline_obj])

    if isinstance(timeline_obj, list):
        return pd.DataFrame(timeline_obj)

    if hasattr(timeline_obj, "to_dict"):
        try:
            d = timeline_obj.to_dict()
            if isinstance(d, dict) and "items" in d and isinstance(d["items"], list):
                return pd.DataFrame(d["items"])
            if isinstance(d, dict):
                return pd.DataFrame([d])
        except Exception:
            pass

    if hasattr(timeline_obj, "items"):
        try:
            items = timeline_obj.items
            if isinstance(items, list):
                return pd.DataFrame(items)
        except Exception:
            pass

    return pd.DataFrame()


def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_partial_day_k_from_tiger_minute(quote_client, symbol, date, open_time):
    """
    获取指定股票、指定日期的全天 1分钟 K线数据，并聚合成截至 open_time 的当日日K。
    """
    if quote_client is None:
        return None

    try:
        # 请求历史数据
        timeline = quote_client.get_timeline_history(
            symbols=[symbol],
            date=date.strftime("%Y-%m-%d"),
        )
    except Exception as e:
        print(f"tiger timeline request failed: {symbol} {date.date()} {e}")
        return None

    tl_df = _normalize_tiger_timeline(timeline)
    if tl_df.empty:
        return None

    # 有些返回会把 symbol 分层或混入同一表，优先过滤当前 symbol
    if "symbol" in tl_df.columns:
        tl_df = tl_df[tl_df["symbol"].astype(str) == str(symbol)]

    time_col = _pick_col(tl_df, ["time", "timestamp", "time_point", "trade_time", "datetime"])
    open_col = _pick_col(tl_df, ["open", "Open", "o"])
    high_col = _pick_col(tl_df, ["high", "High", "h"])
    low_col = _pick_col(tl_df, ["low", "Low", "l"])
    close_col = _pick_col(tl_df, ["close", "Close", "c", "latest_price", "price"])
    vol_col = _pick_col(tl_df, ["volume", "Volume", "v"])

    if not all([time_col, open_col, high_col, low_col, close_col]):
        return None

    tl_df = tl_df.copy()
    tl_df[time_col] = pd.to_datetime(tl_df[time_col], errors="coerce")
    for c in [open_col, high_col, low_col, close_col, vol_col]:
        if c:
            tl_df[c] = pd.to_numeric(tl_df[c], errors="coerce")
    tl_df = tl_df.dropna(subset=[time_col, open_col, high_col, low_col, close_col]).sort_values(time_col)
    if tl_df.empty:
        return None

    # 只取到订单时间（包含该分钟）
    cutoff = pd.to_datetime(open_time)
    tl_df = tl_df[tl_df[time_col] <= cutoff]
    if tl_df.empty:
        return None

    day_bar = {
        "Date": cutoff.normalize(),
        "Open": tl_df.iloc[0][open_col],
        "High": tl_df[high_col].max(),
        "Low": tl_df[low_col].min(),
        "Close": tl_df.iloc[-1][close_col],
        "Volume": tl_df[vol_col].sum() if vol_col in tl_df.columns else np.nan,
    }
    return day_bar


# ---------- 配置 ----------
TRADES_FILE = base_dir + "/data/processed/trades_clean_v2.csv"
STOCK_DIR = base_dir + "/data/cache/stocks_day_k"
INDEX_DIR = base_dir + "/data/cache/indexes_day_k"
OUTPUT_FILE = base_dir + "/data/processed/training_dataset_latest.csv"

# 若你已有 tiger quote client，请在外部注入；默认为 None，回退到原日K缓存。
TIGER_QUOTE_CLIENT = None

os.makedirs("output", exist_ok=True)
os.makedirs(STOCK_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)


# ---------- 下载/更新股票行情（可选） ----------
def update_stock_cache(symbol, start="2023-01-01"):
    stock_file = f"{STOCK_DIR}/{symbol}.csv"
    if os.path.exists(stock_file):
        df_old = pd.read_csv(stock_file)
        last_date = pd.to_datetime(df_old["Date"]).max()
        if last_date >= pd.Timestamp.today() - pd.Timedelta(days=1):
            return
        start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        start_date = start

    try:
        df = yf.download(symbol, start=start_date)
        if df.empty:
            return
        df.reset_index(inplace=True)
        df.rename(columns={"Date": "Date", "Adj Close": "Close"}, inplace=True)
        if os.path.exists(stock_file):
            df_old = pd.read_csv(stock_file)
            df = pd.concat([df_old, df], ignore_index=True)
        df.to_csv(stock_file, index=False)
    except Exception as e:
        print("failed to update", symbol, e)


# ---------- 更新所有股票 ----------
trades = pd.read_csv(TRADES_FILE)
symbols = trades["underlying"].unique()
for s in symbols:
    update_stock_cache(s)


# ---------- 检查已有训练集 ----------
if os.path.exists(OUTPUT_FILE):
    dataset_old = pd.read_csv(OUTPUT_FILE)
    processed_keys = set(dataset_old["symbol"] + "_" + dataset_old["holding_minutes"].astype(str))
    rows = []
    append_mode = True
else:
    dataset_old = None
    processed_keys = set()
    rows = []
    append_mode = False


# ---------- 生成训练行 ----------
for _, trade in trades.iterrows():
    key = trade["symbol"] + "_" + str(trade["holding_minutes"])
    if key in processed_keys:
        continue

    symbol = trade["underlying"]
    stock_file = f"{STOCK_DIR}/{symbol}.csv"
    if not os.path.exists(stock_file):
        continue

    stock = pd.read_csv(stock_file)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in stock.columns:
            stock[col] = pd.to_numeric(stock[col], errors="coerce")
    stock = stock.dropna(subset=["Open", "High", "Low", "Close"])
    if stock.empty:
        continue

    stock["Date"] = pd.to_datetime(stock["Date"])
    open_time = pd.to_datetime(trade["open_time"])

    # 用 Tiger 1分钟线重算“订单当日K线（截至订单时间）”
    open_date = open_time.normalize()
    partial_bar = build_partial_day_k_from_tiger_minute(
        quote_client=TIGER_QUOTE_CLIENT,
        symbol=symbol,
        date=open_date,
        open_time=open_time,
    )
    if partial_bar is not None:
        stock = stock[stock["Date"] < open_date]
        stock = pd.concat([stock, pd.DataFrame([partial_bar])], ignore_index=True)

    stock = stock[stock["Date"] <= open_time]
    stock = stock.sort_values("Date")
    if len(stock) < 60:
        continue

    # 技术指标（订单当日由 1分钟线聚合出的“部分日K”参与计算）
    stock["MA5"] = ta.sma(stock["Close"], 5)
    stock["MA20"] = ta.sma(stock["Close"], 20)
    stock["MA60"] = ta.sma(stock["Close"], 60)
    stock["RSI"] = ta.rsi(stock["Close"], 14)
    stock["ATR"] = ta.atr(stock["High"], stock["Low"], stock["Close"], 14)
    stock["VOL20"] = stock["Close"].pct_change().rolling(20).std()
    stock["MOM10"] = stock["Close"].pct_change(10)

    last = stock.iloc[-1]
    price = last["Close"]
    strike = trade["strike"]
    T = max(trade["dte"] / 365, 1e-5)
    r = 0.03
    option_type = trade["option_type"].lower()
    mid_price = trade["avg_open_price"]

    try:
        IV = implied_volatility(mid_price, price, strike, T, r, option_type)
        Delta = bs_delta(price, strike, T, r, IV, option_type)
        Gamma = bs_gamma(price, strike, T, r, IV)
    except Exception:
        IV, Delta, Gamma = np.nan, np.nan, np.nan

    # 市场指数
    SPY_trend = np.nan
    VIX_level = np.nan
    spy_file = f"{INDEX_DIR}/SPY.csv"
    vix_file = f"{INDEX_DIR}/VIX.csv"
    if os.path.exists(spy_file):
        spy = pd.read_csv(spy_file)
        spy["Date"] = pd.to_datetime(spy["Date"])
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in spy.columns:
                spy[col] = pd.to_numeric(spy[col], errors="coerce")
        spy = spy[spy["Date"] <= open_time]
        if len(spy.dropna(subset=["Open", "Close"])) > 0:
            spy_last = spy.dropna(subset=["Open", "Close"]).iloc[-1]
            SPY_trend = spy_last["Close"] - spy_last["Open"]

    if os.path.exists(vix_file):
        vix = pd.read_csv(vix_file)
        vix["Date"] = pd.to_datetime(vix["Date"])
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in vix.columns:
                vix[col] = pd.to_numeric(vix[col], errors="coerce")
        vix = vix[vix["Date"] <= open_time]
        if len(vix.dropna(subset=["Close"])) > 0:
            VIX_level = vix.dropna(subset=["Close"]).iloc[-1]["Close"]

    row = {
        "symbol": trade["symbol"],
        "underlying": symbol,
        "option_type": option_type,
        "stock_price": price,
        "strike": strike,
        "moneyness": price / strike,
        "distance_to_strike": (price - strike) / strike,
        "log_moneyness": np.log(price / strike),
        "dte": trade["dte"],
        "holding_minutes": trade["holding_minutes"],
        "MA5": last["MA5"],
        "MA20": last["MA20"],
        "MA60": last["MA60"],
        "trend_strength": last["MA5"] - last["MA20"],
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


# 合并已有数据
dataset_new = pd.DataFrame(rows)
if append_mode:
    dataset = pd.concat([dataset_old, dataset_new], ignore_index=True)
else:
    dataset = dataset_new

dataset.to_csv(OUTPUT_FILE, index=False)
print("training dataset size:", len(dataset))
print("saved to:", OUTPUT_FILE)
