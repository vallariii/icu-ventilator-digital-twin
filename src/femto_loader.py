"""Load FEMTO/PRONOSTIA bearing dataset and compute RUL per snapshot.

Each bearing has N snapshots taken every 10s. Each acc_XXXXX.csv has 2560 rows
sampled at 25.6 kHz (0.1 s window) with columns:
    hour, minute, second, microsecond, accel_horizontal, accel_vertical

RUL convention: for each snapshot i out of N_total, RUL = (N_total - i) * 10s

Learning_set bearings (full run-to-failure — use these for training):
    Bearing1_1, 1_2, 2_1, 2_2, 3_1, 3_2
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'data' / 'raw' / 'femto' / 'phm-ieee-2012-data-challenge-dataset-master'
OUT = ROOT / 'data' / 'processed'
SNAPSHOT_INTERVAL_S = 10.0  # snapshots taken every 10 seconds
SAMPLE_RATE_HZ = 25600.0

COLS = ['hour', 'minute', 'second', 'microsec', 'acc_h', 'acc_v']


def list_bearings(split: str = 'Learning_set') -> list[Path]:
    base = RAW / split
    return sorted([p for p in base.iterdir() if p.is_dir()])


def load_snapshot(csv_path: Path) -> np.ndarray:
    """Read one acc_XXXXX.csv; return (n_samples, 2) array of (h, v) accel."""
    df = pd.read_csv(csv_path, header=None, names=COLS)
    return df[['acc_h', 'acc_v']].to_numpy(dtype=np.float32)


def load_bearing_snapshots(bearing_dir: Path):
    """Return list of (snapshot_idx, (n,2) array) for all acc files in order."""
    acc_files = sorted(bearing_dir.glob('acc_*.csv'))
    return [(i, load_snapshot(f)) for i, f in enumerate(acc_files)]


def snapshot_index(bearing_dir: Path) -> pd.DataFrame:
    """Return DataFrame of snapshots for a bearing with RUL labels (no raw signal)."""
    acc_files = sorted(bearing_dir.glob('acc_*.csv'))
    n = len(acc_files)
    idx = np.arange(n)
    return pd.DataFrame({
        'bearing': bearing_dir.name,
        'snapshot_idx': idx,
        'file': [f.name for f in acc_files],
        'elapsed_s': idx * SNAPSHOT_INTERVAL_S,
        'total_s': n * SNAPSHOT_INTERVAL_S,
        'RUL_s': (n - 1 - idx) * SNAPSHOT_INTERVAL_S,
    })


if __name__ == '__main__':
    summary = []
    for b in list_bearings('Learning_set'):
        files = list(b.glob('acc_*.csv'))
        summary.append({
            'bearing': b.name,
            'n_snapshots': len(files),
            'duration_min': round(len(files) * SNAPSHOT_INTERVAL_S / 60, 1),
        })
    df = pd.DataFrame(summary)
    print('Learning set bearings:')
    print(df.to_string(index=False))

    # Sanity check on one snapshot
    first = list_bearings('Learning_set')[0]
    first_csv = sorted(first.glob('acc_*.csv'))[0]
    sig = load_snapshot(first_csv)
    print(f'\nSample snapshot shape: {sig.shape}')
    print(f'accel_h range: [{sig[:,0].min():.2f}, {sig[:,0].max():.2f}] g')
    print(f'accel_v range: [{sig[:,1].min():.2f}, {sig[:,1].max():.2f}] g')

    # Save snapshot index for all learning bearings
    all_idx = pd.concat([snapshot_index(b) for b in list_bearings('Learning_set')])
    OUT.mkdir(parents=True, exist_ok=True)
    all_idx.to_parquet(OUT / 'femto_snapshot_index.parquet', index=False)
    print(f'\nSaved snapshot index ({len(all_idx)} rows) -> {OUT / "femto_snapshot_index.parquet"}')
