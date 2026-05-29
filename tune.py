import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
import json
from config_loader import load_and_split_data, CATEGORICAL_FEATURES

# --- 1. 定义 Optuna 的“目标函数” ---
# (此函数在 tune.py 内部定义，因为它只在这里使用)

# 加载数据（我们只需要训练集来进行交叉验证）
print("加载数据用于调优...")
X_train, _, y_train, _ = load_and_split_data()
print("\n")


def objective(trial):
    """
    Optuna 的目标函数，使用 StratifiedKFold 进行交叉验证。
    """

    # --- A. 定义参数的搜索空间 ---
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 2000, step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
    }

    # --- B. 定义固定参数 ---
    fixed_params = {
        'objective': 'binary',
        'metric': 'auc',
        'is_unbalance': True,
        'random_state': 42,
        'n_jobs': 48,
        'verbose': -1,  # 关闭 LightGBM 的日志
    }

    # --- C. 定义交叉验证策略 ---
    N_SPLITS = 5
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    scores = []

    # 重置索引以便 .iloc[idx] 能正确工作
    X_train_reset = X_train.reset_index(drop=True)
    y_train_reset = y_train.reset_index(drop=True)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_reset, y_train_reset)):
        X_train_fold = X_train_reset.iloc[train_idx]
        y_train_fold = y_train_reset.iloc[train_idx]
        X_val_fold = X_train_reset.iloc[val_idx]
        y_val_fold = y_train_reset.iloc[val_idx]

        model = lgb.LGBMClassifier(**fixed_params, **params)

        model.fit(
            X_train_fold,
            y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(100, verbose=False)],
            categorical_feature=CATEGORICAL_FEATURES
        )

        preds_val = model.predict_proba(X_val_fold)[:, 1]
        auc_score = roc_auc_score(y_val_fold, preds_val)
        scores.append(auc_score)

    return np.mean(scores)


# --- 2. 创建并运行 Optuna "Study" ---
if __name__ == "__main__":
    print("--- 2. 开始 Optuna 超参数寻优 ---")
    study = optuna.create_study(direction='maximize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    N_TRIALS = 100

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("--- 寻优完成 ---")
    print(f"共尝试了 {N_TRIALS} 次。")
    print(f"最佳平均 AUC (5-Fold CV): {study.best_value:.4f}")
    print("找到的最佳参数:")
    print(study.best_params)
    print("\n")

    # --- 3. 保存最佳参数 ---
    PARAMS_FILE = 'best_params.json'
    with open(PARAMS_FILE, 'w') as f:
        json.dump(study.best_params, f, indent=4)

    print(f"最佳参数已保存到 '{PARAMS_FILE}'")