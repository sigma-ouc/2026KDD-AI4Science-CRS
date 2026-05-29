# 2026KDD-AI4Science-CRS

This repository is a script-level snapshot exported from the local workspace. It contains the core configuration, tuning, training, and evaluation entry points used for the CRS-based depression prediction pipeline.

## Included files

- `config_loader.py`
  - Centralizes dataset loading, train/test split, feature lists, and categorical column definitions.
  - Expects the cleaned CSV at the repository root.

- `tune.py`
  - Runs Optuna hyperparameter search for LightGBM on the training split only.
  - Writes the tuned parameter file `best_params.json`.

- `train_final_model.py`
  - Trains the final LightGBM model using `best_params.json`.
  - Uses an internal validation split from the training set for early stopping and operating-point selection.
  - Saves `final_lgbm_model.joblib` as a model artifact dictionary containing:
    - the fitted model,
    - the fixed operating threshold,
    - the selected `best_iteration`,
    - validation metrics used to choose the threshold.

- `predict_evaluate.py`
  - Loads `final_lgbm_model.joblib`.
  - Evaluates the final model on the held-out test split once.
  - Uses the fixed threshold saved during training; it does **not** optimize the threshold on the test set.
  - Exports plots and evaluation artifacts to `result/`.

## Recommended workflow

1. Create an environment with the required packages:
   - `lightgbm=4.6.0`
   - `optuna=4.5.0`
   - `pandas=2.3.1`
   - `numpy=1.26.4`
   - `scikit-learn=1.6.1`
   - `matplotlib=3.9.4`
   - `shap=0.49.1`
   - `joblib=1.5.1`

2. Put the cleaned CSV file in the repository root.

3. Tune hyperparameters:

   ```bash
   python tune.py
   ```

4. Train the final model and select the operating point on an internal validation split:

   ```bash
   python train_final_model.py
   ```

5. Evaluate on the held-out test split:

   ```bash
   python predict_evaluate.py
   ```

## Notes

- `config_loader.py` currently enables:
  - `CORE_PREDICTORS`
  - `COVARIATES`
  - `NEW_FEATURES = ['circadian_rhythm_score']`

- Engineered activity features are currently disabled in `config_loader.py` by default. If an activity-volume experiment requires them, re-enable them explicitly in `NEW_FEATURES`.

- `predict_evaluate.py` assumes that `final_lgbm_model.joblib` was produced by `train_final_model.py`. If you load a legacy bare-model artifact instead, the script falls back to a threshold of `0.5`.
