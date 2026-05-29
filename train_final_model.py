import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

from config_loader import CATEGORICAL_FEATURES, load_and_split_data

PARAMS_FILE = Path("best_params.json")
MODEL_FILE = Path("final_lgbm_model.joblib")
RANDOM_STATE = 42
VALIDATION_SIZE = 0.2


def select_best_f1_threshold(y_true, y_prob):
    """Select the operating threshold on a validation set only."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    if len(thresholds) == 0:
        return 0.5, np.nan

    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


print("加载数据用于最终模型训练...")
X_train, X_test, y_train, y_test = load_and_split_data()
_ = X_test, y_test  # The outer test split must remain untouched during training.
print("\n")

print("从训练集内部划分验证集，用于 early stopping 和 operating point selection ...")
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_train,
)
print(f"内部训练子集: {X_fit.shape}")
print(f"内部验证子集: {X_val.shape}")
print("\n")

try:
    with PARAMS_FILE.open("r", encoding="utf-8") as file:
        best_params = json.load(file)
    print(f"成功从 '{PARAMS_FILE}' 加载最优参数。")
except FileNotFoundError:
    print(f"错误：未找到参数文件 '{PARAMS_FILE}'。请先运行 'tune.py'。")
    raise SystemExit(1)

print("--- 训练 early-stopping 模型（仅使用训练集内部验证集） ---")
base_params = {
    "objective": "binary",
    "metric": "auc",
    "is_unbalance": True,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    **best_params,
}

early_stop_model = lgb.LGBMClassifier(**base_params)
early_stop_model.fit(
    X_fit,
    y_fit,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(100, verbose=True)],
    categorical_feature=CATEGORICAL_FEATURES,
)

best_iteration = early_stop_model.best_iteration_
if best_iteration is None or int(best_iteration) <= 0:
    best_iteration = int(base_params["n_estimators"])
else:
    best_iteration = int(best_iteration)

val_pred_probs = early_stop_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_pred_probs)
val_pr_auc = average_precision_score(y_val, val_pred_probs)
operating_threshold, best_val_f1 = select_best_f1_threshold(y_val, val_pred_probs)

print("--- 内部验证集结果 ---")
print(f"Best iteration from internal validation: {best_iteration}")
print(f"Validation ROC-AUC: {val_auc:.4f}")
print(f"Validation PR-AUC: {val_pr_auc:.4f}")
print(f"Operating threshold (max F1 on validation PR curve): {operating_threshold:.4f}")
print(f"Validation F1 at selected threshold: {best_val_f1:.4f}")
print("\n")

print("--- 在完整训练集上重训最终模型（不再使用测试集） ---")
final_params = dict(base_params)
final_params["n_estimators"] = best_iteration
final_model = lgb.LGBMClassifier(**final_params)
final_model.fit(
    X_train,
    y_train,
    categorical_feature=CATEGORICAL_FEATURES,
)

artifact = {
    "model": final_model,
    "operating_threshold": operating_threshold,
    "threshold_selection_metric": "max_f1_on_internal_validation_pr_curve",
    "best_iteration": best_iteration,
    "validation_auc": float(val_auc),
    "validation_pr_auc": float(val_pr_auc),
    "validation_f1": float(best_val_f1),
    "validation_size": VALIDATION_SIZE,
    "random_state": RANDOM_STATE,
}

joblib.dump(artifact, MODEL_FILE)
print(f"最终模型与 operating point 已保存到 '{MODEL_FILE}'")
