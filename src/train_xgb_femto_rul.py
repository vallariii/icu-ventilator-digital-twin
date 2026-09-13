"""XGBoost RUL regressor on FEMTO bearing features (PDF Section 7.4).

Features already extracted by femto_features.py. Split by BEARING (not by
row) so no bearing leaks between train and val — this is the honest way to
evaluate RUL generalisation to unseen equipment.

Target: RUL_s (seconds). Normalise to [0, 1] per PDF guidance, then
inverse-scale back to seconds at inference.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / 'data' / 'processed' / 'femto_learning_set_features.parquet'
MODELS = ROOT / 'models'
MODELS.mkdir(exist_ok=True)

# Hold out 2 bearings (one per operating condition family) for validation
HOLDOUT = ['Bearing1_2', 'Bearing2_2']


def main():
    df = pd.read_parquet(FEAT)
    print(f'FEMTO features: {df.shape}')
    print(f'Bearings: {sorted(df["bearing"].unique())}')

    # Predict fraction of life remaining per bearing — the standard FEMTO target.
    # This normalises away the different bearing lifetimes so the model learns
    # "how worn is this bearing" rather than "how many seconds until failure
    # on a specific fixture."
    df['bearing_life_s'] = df.groupby('bearing')['RUL_s'].transform('max')
    df['RUL_frac'] = df['RUL_s'] / df['bearing_life_s']
    y = df['RUL_frac'].to_numpy()
    train_mask = ~df['bearing'].isin(HOLDOUT)
    print(f'Target: RUL_frac in [0,1]   (0 = failure now, 1 = fresh bearing)')

    drop_cols = ['bearing', 'snapshot_idx', 'elapsed_s', 'RUL_s',
                 'RUL_frac', 'bearing_life_s']
    X = df.drop(columns=drop_cols)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    print(f'Feature cols: {X.shape[1]}')

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_va, y_va = X[~train_mask], y[~train_mask]
    print(f'Train: {len(X_tr)} rows (bearings: {sorted(df.loc[train_mask,"bearing"].unique())})')
    print(f'Val:   {len(X_va)} rows (bearings: {HOLDOUT})')

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
    print(f'\nValidation (RUL as fraction of bearing life):')
    print(f'  RMSE={rmse:.4f}')
    print(f'  MAE ={mae:.4f}')
    print(f'  R²  ={r2:.3f}')
    print(f'  Target RMSE < 0.10   -> {"PASS" if rmse < 0.10 else "BELOW TARGET"}')

    # Translate to per-bearing seconds for interpretability
    val_df = df[~train_mask].copy()
    val_df['pred_frac'] = preds
    val_df['pred_rul_s'] = val_df['pred_frac'] * val_df['bearing_life_s']
    per_bearing = val_df.groupby('bearing').apply(
        lambda g: np.sqrt(mean_squared_error(g['RUL_s'], g['pred_rul_s']))
    )
    print('\nPer-bearing RMSE (seconds):')
    for b, v in per_bearing.items():
        print(f'  {b}: {v:.0f} s ({v/60:.1f} min)')

    # Top features
    imp = (pd.Series(model.feature_importances_, index=X.columns)
           .sort_values(ascending=False).head(15))
    print('\nTop-15 features:')
    print(imp.to_string())

    joblib.dump({'model': model, 'features': X.columns.tolist(),
                 'target': 'RUL_frac'},
                MODELS / 'xgb_rul_femto.joblib')
    with open(MODELS / 'xgb_rul_femto_meta.json', 'w') as f:
        json.dump({
            'target': 'RUL as fraction of bearing lifetime',
            'train_bearings': sorted(df.loc[train_mask,'bearing'].unique().tolist()),
            'val_bearings': HOLDOUT,
            'n_train': int(len(X_tr)), 'n_val': int(len(X_va)),
            'n_features': int(X.shape[1]),
            'rmse_frac': rmse, 'mae_frac': mae, 'r2': r2,
            'per_bearing_rmse_s': {k: float(v) for k, v in per_bearing.items()},
            'best_iteration': int(model.best_iteration),
        }, f, indent=2)
    print(f'\nSaved -> {MODELS}/xgb_rul_femto.joblib')


if __name__ == '__main__':
    main()
