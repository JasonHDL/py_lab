import pandas as pd
from datetime import datetime
from light_gbm.train_ai import base_dir


INPUT = base_dir+"/data/raw/trader_history.csv"
OUTPUT = base_dir+"/data/processed/trades_clean_v2.csv"

def parse_option_symbol(symbol):

    parts = symbol.split()

    underlying = parts[0]
    expiry = datetime.strptime(parts[1], "%d%b%y")
    strike = float(parts[2])
    option_type = parts[3]

    return underlying, expiry, strike, option_type


df = pd.read_csv(INPUT)

# 修复数值类型
numeric_cols = [
    "Quantity",
    "T. Price",
    "Realized P/L",
    "MTM P/L",
    "Basis",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df["trade_time"] = pd.to_datetime(df["Date/Time"])

df = df[df["Asset Category"] == "Equity and Index Options"]

df = df.sort_values("trade_time")

rows = []

for symbol, g in df.groupby("Symbol"):

    g = g.sort_values("trade_time")

    try:
        underlying, expiry, strike, option_type = parse_option_symbol(symbol)
    except:
        continue

    open_trades = g[g["Code"].str.contains("O", na=False)]
    close_trades = g[g["Code"].str.contains("C", na=False)]

    if len(open_trades) == 0:
        continue

    open_qty = open_trades["Quantity"].sum()

    if open_qty == 0:
        continue

    avg_open_price = (
        (abs(open_trades["Quantity"]) * open_trades["T. Price"]).sum()
        / abs(open_qty)
    )

    open_time = open_trades.iloc[0]["trade_time"]

    close_time = g.iloc[-1]["trade_time"]

    holding_minutes = (close_time - open_time).total_seconds() / 60

    pnl = g["Realized P/L"].sum() + g["MTM P/L"].sum()

    basis = abs(g["Basis"].sum())

    pnl_pct = pnl / basis if basis != 0 else 0

    dte = (expiry - open_time).days

    code_all = "".join(g["Code"].astype(str).values)

    if "Ep" in code_all:
        close_type = "expire"
    elif pnl > 0:
        close_type = "take_profit"
    else:
        close_type = "stop_loss"

    label = 1 if pnl > 0 else 0

    rows.append(
        {
            "symbol": symbol,
            "underlying": underlying,
            "option_type": option_type,
            "strike": strike,
            "expiry": expiry,
            "open_time": open_time,
            "close_time": close_time,
            "holding_minutes": holding_minutes,
            "open_qty": open_qty,
            "avg_open_price": avg_open_price,
            "dte": dte,
            "close_type": close_type,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "label": label,
        }
    )

out = pd.DataFrame(rows)

out.to_csv(OUTPUT, index=False)

print("trades:", len(out))
print("saved:", OUTPUT)