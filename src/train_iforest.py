"""Train Isolation Forest anomaly detector on CMAPSS FD001 features.

PDF Section 7.2:
  - n_estimators=200, contamination=0.05, max_samples='auto'
  - Train on 'baseline' (normal) data — here: earliest cycles of each engine
  - Thresholds: score < -0.1 review, < -0.3 critical

CMAPSS has no explicit labels, so we treat high-RUL (early life) windows as
healthy baseline and score everything. Expect anomaly score to decline as
RUL decreases (i.e., engine degrades).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

ROOT = Path(__file__).resolve().parent.parent
FEAT_PATH = ROOT / 'data' / 'processed' / 'cmapss_FD001_features.parquet'
MODEL_DIR = ROOT / 'models'
MODEL_DIR.mkdir(exist_ok=True)

HEALTHY_RUL_MIN = 100  # windows with RUL >= 100 treated as healthy baseline


def main():
    df = pd.read_parquet(FEAT_PATH)
    print(f'Dataset: {df.shape}')

    y_rul = df['RUL'].astype(float)
    X = df.drop(columns=['unit', 'cycle', 'RUL'])

    # Drop constants, clean infs
    const = X.nunique(dropna=False) <= 1
    X = X.loc[:, ~const].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    print(f'Features after cleanup: {X.shape[1]}')

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Baseline = healthy windows
    healthy_mask = (y_rul >= HEALTHY_RUL_MIN).to_numpy()
    print(f'Healthy baseline windows (RUL>={HEALTHY_RUL_MIN}): {healthy_mask.sum()} '
          f'({100 * healthy_mask.mean():.1f}%)')

    iso = IsolationForest(
        n_estimators=200, contamination=0.05,
        max_samples='auto', random_state=42, n_jobs=-1,
    )
    iso.fit(X_scaled[healthy_mask])

    scores = iso.score_samples(X_scaled)  # higher = more normal
    df_out = df[['unit', 'cycle', 'RUL']].copy()
    df_out['anomaly_score'] = scores

    # Sanity check: mean score by RUL bucket — should drop as engine degrades
    buckets = pd.cut(df_out['RUL'], bins=[0, 20, 50, 80, 125],
                     labels=['0-20', '20-50', '50-80', '80-125'])
    by_bucket = df_out.groupby(buckets, observed=False)['anomaly_score'].agg(['mean', 'std', 'count'])
    print('\nAnomaly score by RUL bucket (lower = more anomalous):')
    print(by_bucket.to_string())

    # Empirical thresholds from healthy distribution (sklearn score convention
    # differs from PDF; use percentiles of healthy baseline).
    healthy_scores = scores[healthy_mask]
    thr_review = float(np.percentile(healthy_scores, 5))
    thr_critical = float(np.percentile(healthy_scores, 1))
    n_review = int((scores < thr_review).sum())
    n_critical = int((scores < thr_critical).sum())
    print(f'\nEmpirical thresholds:')
    print(f'  review   (5th pct healthy) = {thr_review:.4f}  -> {n_review} flags ({100*n_review/len(scores):.1f}%)')
    print(f'  critical (1st pct healthy) = {thr_critical:.4f}  -> {n_critical} flags ({100*n_critical/len(scores):.1f}%)')
    # Fraction of near-failure windows (RUL<=20) that trip critical
    near_fail = df_out['RUL'] <= 20
    hit_rate = float((scores[near_fail] < thr_critical).mean())
    print(f'  critical hit rate on RUL<=20 windows: {100*hit_rate:.1f}%')

    # Persist
    joblib.dump({'model': iso, 'scaler': scaler, 'features': X.columns.tolist()},
                MODEL_DIR / 'iforest_fd001.joblib')
    df_out.to_parquet(MODEL_DIR / 'iforest_fd001_scores.parquet', index=False)
    meta = {
        'dataset': 'CMAPSS FD001',
        'n_features': int(X.shape[1]),
        'healthy_rul_min': HEALTHY_RUL_MIN,
        'n_healthy_train': int(healthy_mask.sum()),
        'n_total_scored': int(len(scores)),
        'n_review': n_review, 'n_critical': n_critical,
        'threshold_review': thr_review,
        'threshold_critical': thr_critical,
        'critical_hit_rate_rul_le_20': hit_rate,
        'score_by_rul_bucket': by_bucket.reset_index().to_dict(orient='records'),
    }
    with open(MODEL_DIR / 'iforest_fd001_meta.json', 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    print(f'\nSaved -> {MODEL_DIR}/iforest_fd001.joblib')


if __name__ == '__main__':
    main()
