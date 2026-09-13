"""LSTM Autoencoder for multivariate time-series anomaly detection (PDF 7.3).

Train on CMAPSS FD001 — reconstruct healthy sensor sequences. At inference,
per-window reconstruction MAE is the anomaly score.

Architecture per PDF:
    LSTM(128) -> LSTM(64) -> RepeatVector(T) -> LSTM(64) -> LSTM(128) -> Dense(n_features)

Input: (batch, 50 timesteps, n_features) — 50 cycles per window.
Training data: only windows where RUL >= 100 (healthy).
Threshold: 95th percentile of training MAE + 2 * std.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler
import joblib

ROOT = Path(__file__).resolve().parent.parent
CMAPSS = ROOT / 'data' / 'processed' / 'cmapss_FD001_train.parquet'
MODELS = ROOT / 'models'
MODELS.mkdir(exist_ok=True)

TIMESTEPS = 50
HEALTHY_RUL_MIN = 100
SENSOR_COLS = [f's{i}' for i in range(1, 22)]


def make_windows_generic(df, cols, timesteps):
    """Slide a window of length `timesteps` across each unit's cycles."""
    X, meta = [], []
    for unit_id, grp in df.groupby('unit', sort=True):
        grp = grp.sort_values('cycle').reset_index(drop=True)
        sigs = grp[cols].to_numpy(dtype=np.float32)
        n = len(grp)
        if n < timesteps:
            continue
        for start in range(n - timesteps + 1):
            X.append(sigs[start:start + timesteps])
            row = grp.iloc[start + timesteps - 1]
            meta.append({'unit': int(unit_id),
                         'end_cycle': int(row['cycle']),
                         'end_RUL': float(row['RUL'])})
    return np.stack(X), pd.DataFrame(meta)


def build_model(n_features: int, timesteps: int = TIMESTEPS):
    inp = layers.Input(shape=(timesteps, n_features))
    x = layers.LSTM(128, return_sequences=True)(inp)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.RepeatVector(timesteps)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(128, return_sequences=True)(x)
    out = layers.TimeDistributed(layers.Dense(n_features))(x)
    m = models.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
    return m


def main():
    print(f'Loading {CMAPSS}')
    df = pd.read_parquet(CMAPSS)
    print(f'CMAPSS FD001 train: {df.shape}')

    # Drop constant sensor cols (they confuse the scaler)
    nunique = df[SENSOR_COLS].nunique()
    const_sensors = nunique[nunique <= 1].index.tolist()
    use_cols = [c for c in SENSOR_COLS if c not in const_sensors]
    print(f'Using {len(use_cols)}/{len(SENSOR_COLS)} sensors (dropped {const_sensors})')

    # Scale on the whole training set (healthy + degraded) so stats are stable
    scaler = StandardScaler()
    df[use_cols] = scaler.fit_transform(df[use_cols])

    # Build windows over the non-constant sensors only
    X_all, meta_all = make_windows_generic(df, use_cols, TIMESTEPS)
    print(f'All windows: {X_all.shape}')

    healthy = meta_all['end_RUL'] >= HEALTHY_RUL_MIN
    X_train = X_all[healthy.values]
    X_rest = X_all[~healthy.values]
    print(f'Healthy train: {X_train.shape}    Rest (eval): {X_rest.shape}')

    # Train/val split from healthy
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X_train))
    split = int(0.9 * len(idx))
    X_tr, X_va = X_train[idx[:split]], X_train[idx[split:]]

    model = build_model(n_features=len(use_cols))
    model.summary(print_fn=lambda s: print(f'  {s}'))

    es = callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss')
    hist = model.fit(X_tr, X_tr, validation_data=(X_va, X_va),
                     epochs=50, batch_size=64, callbacks=[es], verbose=2)

    # Threshold from training reconstruction error
    recon_tr = model.predict(X_tr, batch_size=128, verbose=0)
    mae_tr = np.mean(np.abs(X_tr - recon_tr), axis=(1, 2))
    threshold = float(np.percentile(mae_tr, 95) + 2 * mae_tr.std())
    print(f'\nTraining MAE: mean={mae_tr.mean():.4f}  p95={np.percentile(mae_tr,95):.4f}  std={mae_tr.std():.4f}')
    print(f'Anomaly threshold (p95 + 2σ): {threshold:.4f}')

    # Score everything
    recon_all = model.predict(X_all, batch_size=128, verbose=0)
    mae_all = np.mean(np.abs(X_all - recon_all), axis=(1, 2))
    meta_all['recon_mae'] = mae_all
    meta_all['is_anomaly'] = mae_all > threshold

    # Bucket sanity
    buckets = pd.cut(meta_all['end_RUL'], bins=[0, 20, 50, 80, 125],
                     labels=['0-20', '20-50', '50-80', '80-125'])
    print('\nMAE by RUL bucket (higher = more anomalous):')
    print(meta_all.groupby(buckets, observed=False)['recon_mae'].agg(['mean', 'std', 'count']).to_string())
    near_fail = meta_all['end_RUL'] <= 20
    hit = float(meta_all.loc[near_fail, 'is_anomaly'].mean())
    print(f'Anomaly hit rate on RUL<=20 windows: {100*hit:.1f}%')

    # Persist
    model.save(MODELS / 'lstm_ae_fd001.keras')
    joblib.dump({'scaler': scaler, 'sensor_cols': use_cols,
                 'timesteps': TIMESTEPS, 'threshold': threshold},
                MODELS / 'lstm_ae_fd001_preproc.joblib')
    meta_all.to_parquet(MODELS / 'lstm_ae_fd001_scores.parquet', index=False)
    with open(MODELS / 'lstm_ae_fd001_meta.json', 'w') as f:
        json.dump({
            'n_features': len(use_cols), 'timesteps': TIMESTEPS,
            'n_healthy_train': int(healthy.sum()),
            'threshold': threshold,
            'hit_rate_rul_le_20': hit,
            'epochs_trained': len(hist.history['loss']),
            'final_val_loss': float(hist.history['val_loss'][-1]),
        }, f, indent=2)
    print(f'\nSaved -> {MODELS}/lstm_ae_fd001.keras')


if __name__ == '__main__':
    main()
