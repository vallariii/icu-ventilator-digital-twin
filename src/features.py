"""Feature engineering for ICU Ventilator Digital Twin.

Two entry points:
    extract_window_features(window: np.ndarray, fs: float) -> dict
        For live sensor windows (vibration, pressure, current, etc.).
        window shape: (n_samples, n_channels) or (n_samples,) for 1D.

    extract_cmapss_features(df, window: int, stride: int) -> pd.DataFrame
        Rolling-window stats across the 21 CMAPSS sensor channels per unit.
        Produces one feature row per window with the unit's RUL at window end.

Per PDF Section 7.1: rolling stats (mean/std/min/max/RMS), FFT bands
(0-10, 10-50, 50-200 Hz), dX/dt, kurtosis, skew.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import rfft, rfftfreq

FFT_BANDS = [(0, 10), (10, 50), (50, 200)]


def _band_power(signal: np.ndarray, fs: float) -> dict:
    """Return energy in each FFT band (rotation / mechanical / bearing)."""
    n = len(signal)
    if n < 4:
        return {f'fft_{lo}_{hi}': 0.0 for lo, hi in FFT_BANDS}
    spec = np.abs(rfft(signal - signal.mean()))
    freqs = rfftfreq(n, d=1.0 / fs)
    out = {}
    for lo, hi in FFT_BANDS:
        mask = (freqs >= lo) & (freqs < hi)
        out[f'fft_{lo}_{hi}'] = float(spec[mask].sum()) if mask.any() else 0.0
    return out


def _channel_features(sig: np.ndarray, fs: float, prefix: str = '') -> dict:
    """Basic time + frequency features for one 1-D signal."""
    if sig.size == 0:
        return {}
    rms = float(np.sqrt(np.mean(sig ** 2)))
    feats = {
        f'{prefix}mean': float(sig.mean()),
        f'{prefix}std': float(sig.std()),
        f'{prefix}min': float(sig.min()),
        f'{prefix}max': float(sig.max()),
        f'{prefix}rms': rms,
        f'{prefix}kurt': float(stats.kurtosis(sig)),
        f'{prefix}skew': float(stats.skew(sig)),
        f'{prefix}ptp': float(np.ptp(sig)),
    }
    # rate of change (dX/dt approximation)
    if sig.size > 1:
        dx = np.diff(sig) * fs
        feats[f'{prefix}dxdt_rms'] = float(np.sqrt(np.mean(dx ** 2)))
        feats[f'{prefix}dxdt_max'] = float(np.max(np.abs(dx)))
    # FFT bands
    for k, v in _band_power(sig, fs).items():
        feats[f'{prefix}{k}'] = v
    return feats


def extract_window_features(window, fs: float, channel_names: list[str] | None = None) -> dict:
    """Extract features from a single window.

    window: 1-D array, or 2-D array shape (n_samples, n_channels).
    fs: sampling frequency in Hz.
    """
    arr = np.asarray(window, dtype=float)
    if arr.ndim == 1:
        return _channel_features(arr, fs, prefix='')
    if channel_names is None:
        channel_names = [f'ch{i}' for i in range(arr.shape[1])]
    feats = {}
    for i, name in enumerate(channel_names):
        feats.update(_channel_features(arr[:, i], fs, prefix=f'{name}_'))
    # cross-sensor correlation (upper triangle)
    if arr.shape[1] >= 2:
        corr = np.corrcoef(arr.T)
        for i in range(arr.shape[1]):
            for j in range(i + 1, arr.shape[1]):
                feats[f'corr_{channel_names[i]}_{channel_names[j]}'] = float(corr[i, j])
    return feats


# --- CMAPSS-specific windowing -----------------------------------------------

SENSOR_COLS = [f's{i}' for i in range(1, 22)]


def extract_cmapss_features(df: pd.DataFrame, window: int = 30, stride: int = 1) -> pd.DataFrame:
    """Rolling window features per unit for CMAPSS.

    window: # of cycles per window (CMAPSS is 1 row per cycle, so fs=1 "per cycle").
    stride: step between window starts.
    """
    rows = []
    for unit_id, grp in df.groupby('unit', sort=True):
        grp = grp.sort_values('cycle').reset_index(drop=True)
        signals = grp[SENSOR_COLS].to_numpy()
        n = len(grp)
        if n < window:
            continue
        for start in range(0, n - window + 1, stride):
            end = start + window
            win = signals[start:end]
            feats = extract_window_features(win, fs=1.0, channel_names=SENSOR_COLS)
            feats['unit'] = int(unit_id)
            feats['cycle'] = int(grp.loc[end - 1, 'cycle'])
            feats['RUL'] = float(grp.loc[end - 1, 'RUL'])
            rows.append(feats)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from cmapss_loader import load_cmapss

    print('Loading CMAPSS FD001 ...')
    train, _, _ = load_cmapss('FD001')
    print(f'Train: {len(train)} rows, {train["unit"].nunique()} units')

    print('Extracting rolling-window features (window=30, stride=1) ...')
    feats = extract_cmapss_features(train, window=30, stride=1)
    print(f'Feature matrix: {feats.shape}')
    print(f'Columns (first 10): {list(feats.columns[:10])}')

    out = Path(__file__).resolve().parent.parent / 'data' / 'processed' / 'cmapss_FD001_features.parquet'
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out, index=False)
    print(f'Saved -> {out}')
