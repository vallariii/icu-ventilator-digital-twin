"""Extract vibration features from FEMTO snapshots using features.py.

One feature row per snapshot (0.1s window, 2560 samples @ 25.6 kHz).
Target variable: RUL_s (remaining seconds until failure).
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import numpy as np
import pandas as pd

from features import extract_window_features
from femto_loader import (list_bearings, load_snapshot,
                          SAMPLE_RATE_HZ, SNAPSHOT_INTERVAL_S)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'data' / 'processed'


def process_bearing(bearing_dir: Path) -> pd.DataFrame:
    acc_files = sorted(bearing_dir.glob('acc_*.csv'))
    n = len(acc_files)
    rows = []
    for i, f in enumerate(acc_files):
        sig = load_snapshot(f)  # (2560, 2)
        feats = extract_window_features(sig, fs=SAMPLE_RATE_HZ,
                                        channel_names=['h', 'v'])
        feats['bearing'] = bearing_dir.name
        feats['snapshot_idx'] = i
        feats['elapsed_s'] = i * SNAPSHOT_INTERVAL_S
        feats['RUL_s'] = (n - 1 - i) * SNAPSHOT_INTERVAL_S
        rows.append(feats)
    df = pd.DataFrame(rows)
    print(f'  {bearing_dir.name}: {len(df)} snapshots, {df.shape[1]} features')
    return df


def main(split: str = 'Learning_set', n_workers: int = None):
    bearings = list_bearings(split)
    print(f'{split}: {len(bearings)} bearings')
    n_workers = n_workers or max(1, (os.cpu_count() or 2) - 1)
    print(f'Using {n_workers} parallel workers')

    frames = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(process_bearing, b): b for b in bearings}
        for fut in as_completed(futures):
            frames.append(fut.result())

    full = pd.concat(frames, ignore_index=True)
    full = full.sort_values(['bearing', 'snapshot_idx']).reset_index(drop=True)

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f'femto_{split.lower()}_features.parquet'
    full.to_parquet(out_path, index=False)
    print(f'\nSaved {full.shape} -> {out_path}')
    print(f'Bearings: {full["bearing"].unique().tolist()}')
    print(f'RUL_s range: [{full["RUL_s"].min():.0f}, {full["RUL_s"].max():.0f}]')


if __name__ == '__main__':
    main('Learning_set')
