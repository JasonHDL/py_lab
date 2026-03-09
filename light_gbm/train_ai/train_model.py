# train_lightgbm.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve
import joblib
from light_gbm.train_ai import base_dir

# 文件路径
DATA_FILE = base_dir+"/data/processed/training_dataset_latest.csv"
MODEL_FILE = base_dir+"/data/lightgbm_model.pkl"

# ---------- 读取训练集 ----------
data = pd.read_csv(DATA_FILE)

# ---------- 特征 & 标签 ----------
# 去掉泄露特征 pnl / pnl_pct / holding_minutes
drop_cols = ["symbol","underlying","option_type","label","pnl","pnl_pct","holding_minutes"]
X = data.drop(columns=drop_cols)
y = data["label"]

# 缺失值填充
X.fillna(0, inplace=True)

# ---------- 切分训练/验证集 ----------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- LightGBM 数据集 ----------
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

# ---------- 参数 ----------
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
    "is_unbalance": True  # 自动平衡正负样本
}

callbacks = [
    lgb.early_stopping(stopping_rounds=50, first_metric_only=True),
    lgb.log_evaluation(50)
]

# ---------- 训练 ----------
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_val],
    valid_names=["train","val"],
    callbacks=callbacks
)

# ---------- 验证 ----------
y_pred_prob = gbm.predict(X_val, num_iteration=gbm.best_iteration)

# 自动选择最佳阈值 (maximize F1)
precision, recall, thresholds = precision_recall_curve(y_val, y_pred_prob)
f1_scores = 2*precision*recall/(precision+recall+1e-10)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
y_pred = (y_pred_prob >= best_threshold).astype(int)

acc = accuracy_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_pred_prob)
print(f"Validation Accuracy: {acc:.4f}, AUC: {auc:.4f}, Best Threshold: {best_threshold:.4f}")

# ---------- 特征重要性 ----------
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": gbm.feature_importance()
}).sort_values(by="importance", ascending=False)
print("Top 20 features:\n", importance.head(20))

# ---------- 保存模型 ----------
joblib.dump(gbm, MODEL_FILE)
print("LightGBM model saved to:", MODEL_FILE)