import os
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from config_loader import load_and_split_data

SAVE_DIR = Path("result")
SAVE_DIR.mkdir(exist_ok=True)
MODEL_FILE = Path("final_lgbm_model.joblib")
BOOTSTRAP_ROUNDS = 1000


def bootstrap_ci(y, p, metric_fn, n_boot=BOOTSTRAP_ROUNDS, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(metric_fn(y[idx], p[idx]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(np.mean(stats)), float(lo), float(hi)


def load_model_artifact(model_path: Path):
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        operating_threshold = float(artifact.get("operating_threshold", 0.5))
        threshold_source = artifact.get(
            "threshold_selection_metric",
            "saved_with_model_artifact",
        )
        metadata = {
            "best_iteration": artifact.get("best_iteration"),
            "validation_auc": artifact.get("validation_auc"),
            "validation_pr_auc": artifact.get("validation_pr_auc"),
            "validation_f1": artifact.get("validation_f1"),
            "validation_size": artifact.get("validation_size"),
            "threshold_source": threshold_source,
        }
        return model, operating_threshold, metadata

    # Backward compatibility for legacy artifacts saved as a bare model.
    return artifact, 0.5, {"threshold_source": "legacy_model_default_0.5"}


def make_groups_from_X(X_full: pd.DataFrame) -> pd.DataFrame:
    gdf = pd.DataFrame(index=X_full.index)
    if "age" in X_full.columns:
        age = pd.to_numeric(X_full["age"], errors="coerce")
        gdf["age_group"] = pd.cut(
            age,
            bins=[-np.inf, 44, 54, 64, 74, np.inf],
            labels=["<=44", "45-54", "55-64", "65-74", "75+"],
        )

    if "education_years" in X_full.columns:
        edu = pd.to_numeric(X_full["education_years"], errors="coerce")

        def edu_bin(x):
            if pd.isna(x):
                return np.nan
            if x <= 0:
                return "0"
            if x <= 6:
                return "1-6"
            if x <= 9:
                return "7-9"
            if x <= 12:
                return "10-12"
            return "13+"

        gdf["edu_group"] = edu.apply(edu_bin).astype("category")

    if "gender" in X_full.columns:
        gdf["gender_label"] = X_full["gender"].map({1: "Male", 2: "Female"})

    if "residence" in X_full.columns:
        gdf["residence_label"] = X_full["residence"].map({1: "Urban", 2: "Rural"})
    return gdf


def circadian_percent_by_group(
    X_full: pd.DataFrame,
    shap_values_class_1: np.ndarray,
    group_series: pd.Series,
    circadian_feature: str = "circadian_rhythm_score",
) -> pd.DataFrame:
    feature_names = list(X_full.columns)
    circadian_index = feature_names.index(circadian_feature)
    rows = []

    valid_groups = group_series.dropna()
    for group_name, idx in valid_groups.groupby(valid_groups).groups.items():
        idx = list(idx)
        sv_group = shap_values_class_1[X_full.index.isin(idx), :]
        mean_abs = np.mean(np.abs(sv_group), axis=0)
        total = float(np.sum(mean_abs))
        pct = float(mean_abs[circadian_index] / total * 100.0) if total > 0 else np.nan
        rows.append({"group": str(group_name), "circ_percentage": pct})

    return pd.DataFrame(rows)


print("加载数据用于评估...")
X_train, X_test, y_train, y_test = load_and_split_data()
print("\n")

try:
    model, operating_threshold, artifact_metadata = load_model_artifact(MODEL_FILE)
    print(f"成功从 '{MODEL_FILE}' 加载模型。")
    print(
        f"使用固定 operating threshold: {operating_threshold:.4f} "
        f"({artifact_metadata['threshold_source']})"
    )
    if artifact_metadata.get("best_iteration") is not None:
        print(f"best_iteration: {artifact_metadata['best_iteration']}")
except FileNotFoundError:
    print(f"错误：未找到模型文件 '{MODEL_FILE}'。请先运行 'train_final_model.py'。")
    raise SystemExit(1)

print("--- 3. 在测试集上评估最终模型 ---")
y_pred_probs = model.predict_proba(X_test)[:, 1]
pr_auc = average_precision_score(y_test, y_pred_probs)
brier = brier_score_loss(y_test, y_pred_probs)
auc_score = roc_auc_score(y_test, y_pred_probs)

print(f"测试集 PR-AUC (Average Precision): {pr_auc:.4f}")
print(f"测试集 Brier score: {brier:.4f}")
print(f"测试集 AUC (ROC-AUC): {auc_score:.4f}")

y_pred_new = (y_pred_probs >= operating_threshold).astype(int)

print(f"\n--- 分类报告 (Fixed Threshold = {operating_threshold:.4f}) ---")
print(
    classification_report(
        y_test,
        y_pred_new,
        target_names=["Non-Depressed (0)", "Depressed (1)"],
        digits=3,
    )
)

print("\n--- 正在保存混淆矩阵 ---")
cm = confusion_matrix(y_test, y_pred_new)
fig, ax = plt.subplots(figsize=(8, 6))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-Depressed", "Depressed"],
)
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")
plt.title(f"Confusion Matrix on Test Set (Threshold = {operating_threshold:.4f})")
plt.savefig(SAVE_DIR / "confusion_matrix.png", dpi=300)
plt.close()

print("--- 4. 正在保存特征重要性图 ---")
lgb.plot_importance(
    model,
    max_num_features=30,
    importance_type="gain",
    figsize=(10, 8),
    title="Feature Importance (Gain - Final Model)",
)
plt.tight_layout()
plt.savefig(SAVE_DIR / "feature_importance_gain.png", dpi=300)
plt.close()

try:
    print("\n" + "=" * 50)
    print("--- 5. 开始 SHAP 可解释性分析 ---")

    explainer = shap.TreeExplainer(model, X_train)
    X_full = pd.concat([X_train, X_test], axis=0)
    shap_values_output = explainer.shap_values(X_full)

    if isinstance(shap_values_output, list):
        shap_values_class_1 = shap_values_output[1]
    else:
        shap_values_class_1 = shap_values_output

    print("--- 5.5. 正在保存 SHAP 摘要图 ---")
    plt.figure()
    shap.summary_plot(
        shap_values_class_1,
        X_full,
        plot_type="dot",
        max_display=20,
        show=False,
    )
    plt.title("SHAP Summary Plot")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "shap_summary_plot.png", dpi=300)
    plt.close()

    print("--- 5.6. 正在保存所有特征的 SHAP 依赖图 ---")
    dep_dir = SAVE_DIR / "dependence_plots"
    dep_dir.mkdir(exist_ok=True)

    for feature_name in X_full.columns:
        try:
            plt.figure()
            shap.dependence_plot(
                feature_name,
                shap_values_class_1,
                X_full,
                interaction_index=None,
                show=False,
            )
            plt.title(f"SHAP Dependence: {feature_name}")
            plt.savefig(dep_dir / f"shap_dep_{feature_name}.png", dpi=300)
            plt.close("all")
        except Exception as exc:
            print(f"绘制 '{feature_name}' 时出错: {exc}")
            plt.close("all")

    print("--- 5.7. 正在生成 CRS 分组贡献分析 ---")
    gdf = make_groups_from_X(X_full)
    panel_candidates = [
        ("age_group", "Age group"),
        ("edu_group", "Education (years)"),
        ("gender_label", "Sex"),
        ("residence_label", "Residence"),
    ]
    panel = [(dim, title) for dim, title in panel_candidates if dim in gdf.columns]

    if panel:
        n_panels = len(panel)
        n_cols = 2
        n_rows = int(np.ceil(n_panels / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(12, 4 * n_rows),
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).flatten()

        for ax, (dim, title) in zip(axes, panel):
            dfp = circadian_percent_by_group(X_full, shap_values_class_1, gdf[dim])
            ax.bar(dfp["group"], dfp["circ_percentage"])
            ax.set_title(title)
            ax.set_ylabel("CRS SHAP Importance (%)")

        for ax in axes[len(panel):]:
            ax.axis("off")

        plt.suptitle("Circadian Rhythm Score Contribution (%)", fontsize=14)
        plt.savefig(SAVE_DIR / "crs_shap_percent_panels.png", dpi=300)
        plt.close()
    else:
        print("未找到可用于 CRS 分组贡献分析的分组变量，跳过该图。")

    print("--- 正在保存 SHAP 特征重要性百分比图 ---")
    mean_abs_shap = np.mean(np.abs(shap_values_class_1), axis=0)
    importance_df = pd.DataFrame(
        {"Feature": X_full.columns, "Mean_Abs_SHAP": mean_abs_shap}
    )
    importance_df["Percentage"] = (
        importance_df["Mean_Abs_SHAP"] / importance_df["Mean_Abs_SHAP"].sum()
    ) * 100
    importance_df = importance_df.sort_values(by="Percentage", ascending=True).tail(20)

    plt.figure(figsize=(10, 12))
    bars = plt.barh(importance_df["Feature"], importance_df["Percentage"], color="#1f77b4")
    plt.xlabel("SHAP Importance Percentage (%)")
    plt.title("Top 20 Features by SHAP Importance (%)")
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.2, bar.get_y() + bar.get_height() / 2, f"{width:.2f}%", va="center")
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "shap_importance_bar_chart.png", dpi=300)
    plt.close()

except Exception as exc:
    print(f"SHAP 分析失败: {exc}")

auc_m, auc_lo, auc_hi = bootstrap_ci(y_test.to_numpy(), y_pred_probs, roc_auc_score)
prauc_m, prauc_lo, prauc_hi = bootstrap_ci(y_test.to_numpy(), y_pred_probs, average_precision_score)
brier_m, brier_lo, brier_hi = bootstrap_ci(y_test.to_numpy(), y_pred_probs, brier_score_loss)

print(f"AUC 95% CI: [{auc_lo:.4f}, {auc_hi:.4f}]")
print(f"PR-AUC 95% CI: [{prauc_lo:.4f}, {prauc_hi:.4f}]")
print(f"Brier 95% CI: [{brier_lo:.4f}, {brier_hi:.4f}]")

threshold_note = (
    f"Threshold = {operating_threshold:.4f}\n"
    f"Threshold source = {artifact_metadata['threshold_source']}\n"
    f"Validation AUC = {artifact_metadata.get('validation_auc')}\n"
    f"Validation PR-AUC = {artifact_metadata.get('validation_pr_auc')}\n"
    f"Validation F1 = {artifact_metadata.get('validation_f1')}\n"
)
(SAVE_DIR / "operating_point.txt").write_text(threshold_note, encoding="utf-8")

print(f"\n所有图像已成功保存至 {os.path.abspath(SAVE_DIR)}")
print("--- 评估任务完成 ---")
