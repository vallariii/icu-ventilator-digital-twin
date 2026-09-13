"""Train XGBoost RUL regressor on CMAPSS FD001 features.

PDF Section 7.4 hyperparameters:
    n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8
Target: RMSE on RUL; < 10% of total useful life (125) = RMSE < 12.5
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
import joblib

ROOT = Path(__file__).resolve().parent.parent
FEAT_PATH = ROOT / 'data' / 'processed' / 'cmapss_FD001_features.parquet'
MODEL_DIR = ROOT / 'models'
MODEL_DIR.mkdir(exist_ok=True)


def main():
    print(f'Loading features from {FEAT_PATH}')
    df = pd.read_parquet(FEAT_PATH)
    print(f'Dataset: {df.shape}')

    y = df['RUL'].astype(float)
    groups = df['unit'].astype(int)
    X = df.drop(columns=['unit', 'cycle', 'RUL'])

    # Drop constant / all-NaN columns
    nunique = X.nunique(dropna=False)
    const_cols = nunique[nunique <= 1].index.tolist()
    if const_cols:
        print(f'Dropping {len(const_cols)} constant feature cols')
        X = X.drop(columns=const_cols)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    print(f'Feature matrix after cleanup: {X.shape}')

    # Unit-level split so no engine leaks across train/val
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(splitter.split(X, y, groups))
    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
    print(f'Train: {len(X_tr)} (units={groups.iloc[train_idx].nunique()})   '
          f'Val: {len(X_va)} (units={groups.iloc[val_idx].nunique()})')

    model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective='reg:squarederror', tree_method='hist',
        early_stopping_rounds=20, n_jobs=-1, random_state=42,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    preds = model.predict(X_va)
    rmse = float(np.sqrt(mean_squared_error(y_va, preds)))
    mae = float(mean_absolute_error(y_va, preds))
    r2 = float(r2_score(y_va, preds))
    print(f'RMSE={rmse:.3f}  MAE={mae:.3f}  R²={r2:.3f}')
    print(f'Target: RMSE < 12.5 (10% of 125)   -> {"PASS" if rmse < 12.5 else "BELOW TARGET"}')

    # Top-20 features by gain
    imp = (pd.Series(model.feature_importances_, index=X.columns)
           .sort_values(ascending=False).head(20))
    print('\nTop-20 features:')
    print(imp.to_string())

    # Save model and metadata
    joblib.dump(model, MODEL_DIR / 'xgb_rul_fd001.joblib')
    meta = {
        'dataset': 'CMAPSS FD001',
        'n_train': int(len(X_tr)), 'n_val': int(len(X_va)),
        'n_features': int(X.shape[1]),
        'rmse': rmse, 'mae': mae, 'r2': r2,
        'best_iteration': int(model.best_iteration),
        'dropped_const_cols': const_cols,
        'feature_cols': X.columns.tolist(),
    }
    with open(MODEL_DIR / 'xgb_rul_fd001_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\nSaved -> {MODEL_DIR}/xgb_rul_fd001.joblib')


if __name__ == '__main__':
    main()
