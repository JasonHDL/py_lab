目标是构建一个 机构级特征体系，但仍然能在 16GB CPU机器上运行，并适合 LightGBM。

设计原则：

📊 特征分 5 大类

1 股票趋势特征
2 股票波动率特征
3 期权定价特征
4 期权市场结构特征
5 市场宏观特征

最终模型通常 60~120个特征。

一、完整 Feature 列表（推荐）
1 股票趋势特征
stock_price
ma5_ratio
ma20_ratio
ma60_ratio

momentum_5
momentum_20

rsi_14
macd

distance_from_high20
distance_from_low20
2 股票波动率特征
realized_vol_10
realized_vol_20
realized_vol_60

atr_14

volatility_regime
3 期权定价特征（核心）
moneyness
log_moneyness
dte
sqrt_dte

iv
iv_rank
iv_percentile

iv_vs_realized_vol

iv_skew
term_structure
4 期权市场结构
open_interest
oi_change

volume
volume_oi_ratio

put_call_ratio

gamma_exposure
5 市场宏观
SPY_trend
SPY_volatility
VIX
VIX_term_structure



===========================================================================
很好，这一步我们直接把 完整可运行的 Feature Engineering + LightGBM 训练框架代码写出来。
目标是：把你的期权订单数据 → 转换成机器学习特征 → 训练 LightGBM 模型预测机会评分。

整个流程分为 6 个模块：

1 数据结构
2 读取数据
3 股票市场特征
4 期权市场特征
5 生成训练样本
6 LightGBM训练

全部都可以在 16GB CPU机器运行

一、项目目录结构

建议做成这样：

option_model/

    data/
        orders.csv
        stock_1min/
            AAPL.csv
            TSLA.csv

        option_chain/
            AAPL.csv

    features/
        feature_builder.py

    model/
        train_model.py

    utils/
        indicators.py

    main.py

   orders.csv

字段

symbol
trade_time
expiry
strike
put_call
quantity
underlying_price
iv
pnl
