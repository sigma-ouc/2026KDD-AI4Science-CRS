# config_loader.py
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from feature_engineering_activity import add_activity_volume_features, ENGINEERED_ACTIVITY_FEATURES


# --- 1. 变量定义 ---

# 目标变量
TARGET_COLUMN = 'is_depressed'

# 核心预测变量 (昼夜节律特征)
CORE_PREDICTORS = [
    'sleep_quality_score',
    'night_sleep_duration_hr',
    'nap_duration_min',
    'activity_regularity',
    'total_activity_level_met',
    'activity_intensity',
    'primary_activity_purpose'
]

# 协变量
COVARIATES = [
    'age',
    'gender',
    'residence',
    'marry',
    'chronic_disease_count',
    'pain',
    'weighted_social_score',
    'smoking_status',
    'smoking_intensity',
    'drinking_status',
    'high_freq_drinker',
    'total_alcohol_g_per_month',
    'work_life_status_code'
]

NEW_FEATURES = [
    'circadian_rhythm_score',
]
# If needed for a dedicated activity-volume experiment, temporarily restore:
# + ENGINEERED_ACTIVITY_FEATURES

# 所有用于建模的特征
ALL_FEATURES = (
                CORE_PREDICTORS
                + COVARIATES
                + NEW_FEATURES
                )

# 分类特征
CATEGORICAL_FEATURES = [
    'activity_intensity',
    'primary_activity_purpose',
    'gender',
    'residence',
    'marry',
    'smoking_status',
    'drinking_status',
    'high_freq_drinker',
    'work_life_status_code'
]

# --- 2. 共享数据加载函数 ---

FILE_PATH = 'AAA_Age_Cleaned.csv'
# FILE_PATH = 'CHARLS_Final_With_WorkStatus_Encoded.csv'

TEST_SPLIT_SIZE = 0.3
RANDOM_SEED = 42  # 确保数据划分一致


def load_and_split_data():
    """
    加载、预处理并划分数据。
    返回: X_train, X_test, y_train, y_test
    """
    try:
        print(f"--- 1. 正在从 '{FILE_PATH}' 加载数据 ---")
        df = pd.read_csv(FILE_PATH)

        # 体力活动特征工程：DA053/DA054/DA055 分箱映射 -> 分钟/周 -> MET-min/week
        # scheme="midpoint"：用区间中点；scheme="lower_bound"：保守下界（建议做敏感性分析）
        df = add_activity_volume_features(df, scheme="midpoint")

        columns_to_load = [TARGET_COLUMN] + ALL_FEATURES
        df = df[columns_to_load]

        print(f"成功加载 {len(df)} 行数据。")

        print(f"--- 2. 正在转换 {len(CATEGORICAL_FEATURES)} 个分类特征 ---")
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype('category')
            else:
                print(f"警告：预期的分类特征 '{col}' 未在文件中找到。")

        print("--- 3. 正在分离特征 (X) 和目标 (y) ---")
        X = df[ALL_FEATURES]
        y = df[TARGET_COLUMN]

        print("目标变量 (is_depression) 分布情况:")
        print(y.value_counts(normalize=True))

        print(f"--- 4. 正在划分 训练集 ({1 - TEST_SPLIT_SIZE:.0%}) 和 测试集 ({TEST_SPLIT_SIZE:.0%}) ---")
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SPLIT_SIZE,
            random_state=RANDOM_SEED,
            stratify=y  # 确保训练集和测试集中的抑郁比例一致
        )

        print("--- 划分完成 ---")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test  shape: {X_test.shape}")

        return X_train, X_test, y_train, y_test

    except FileNotFoundError:
        print(f"错误：文件 '{FILE_PATH}' 未找到。请确保文件在正确的路径下。", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"错误：数据中缺少必要的列。 {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"发生了一个意外错误: {e}", file=sys.stderr)
        sys.exit(1)
