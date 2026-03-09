

import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

base_dir = "E:/hhy/trading_ai"

CACHE_DIR = base_dir+"/data/cache/indexes_day_k"
os.makedirs(CACHE_DIR, exist_ok=True)



indexes = ["SPY", "VIX","QQQ"]

for s in indexes:

    file = f"{CACHE_DIR}/{s}.csv"

    try:

        if not os.path.exists(file):

            print("FULL DOWNLOAD:", s)

            df = yf.download(
                s,
                start="2015-01-01",
                progress=False
            )

            df.reset_index(inplace=True)

            df.to_csv(file, index=False)

        else:

            old = pd.read_csv(file)

            old["Date"] = pd.to_datetime(old["Date"])

            last_date = old["Date"].max()

            start = last_date + timedelta(days=1)

            if start.date() >= datetime.today().date():

                print("UP TO DATE:", s)
                continue

            print("UPDATE:", s, "from", start.date())

            new = yf.download(
                s,
                start=start.strftime("%Y-%m-%d"),
                progress=False
            )

            if len(new) == 0:
                continue

            new.reset_index(inplace=True)

            merged = pd.concat([old, new])

            merged = merged.drop_duplicates(subset=["Date"])

            merged = merged.sort_values("Date")

            merged.to_csv(file, index=False)

    except Exception as e:

        print("FAILED:", s, e)

