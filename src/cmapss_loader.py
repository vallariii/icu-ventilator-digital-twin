"""Load CMAPSS turbofan dataset and compute RUL labels.

Usage:
    from cmapss_loader import load_cmapss
    train, test, rul_truth = load_cmapss('FD001')
    # train columns: unit, cycle, os1..os3, s1..s21, RUL

Then saves processed parquet files to data/processed/cmapss_<subset>.parquet.
"""
from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / 'data' / 'raw' / 'cmapss'
OUT_DIR = Path(__file__).resolve().parent.parent / 'data' / 'processed'

COLUMNS = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r'\s+', header=None, names=COLUMNS, engine='python')


def load_cmapss(subset: str = 'FD001'):
    """Load one subset and return (train_df, test_df, rul_truth_series)."""
    train = _read(RAW_DIR / f'train_{subset}.txt')
    test = _read(RAW_DIR / f'test_{subset}.txt')
    rul_truth = pd.read_csv(RAW_DIR / f'RUL_{subset}.txt', header=None, names=['RUL']).squeeze('columns')

    # Training RUL: for each unit, RUL = max(cycle) - cycle
    max_cycle = train.groupby('unit')['cycle'].transform('max')
    train['RUL'] = max_cycle - train['cycle']

    # Test RUL: truth file gives RUL AT the last observed cycle per unit.
    # So RUL at any earlier cycle = truth + (max_cycle_in_test - cycle)
    max_test_cycle = test.groupby('unit')['cycle'].transform('max')
    rul_map = {i + 1: v for i, v in enumerate(rul_truth.values)}
    test['RUL'] = test['unit'].map(rul_map) + (max_test_cycle - test['cycle'])

    return train, test, rul_truth


def save_all_subsets(clip_rul: int = 125):
    """Process all 4 subsets and save as parquet.

    clip_rul: common practice — RUL > 125 clipped, since early life gives no
    degradation signal. Set to None to disable.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for subset in ['FD001', 'FD002', 'FD003', 'FD004']:
        train, test, _ = load_cmapss(subset)
        if clip_rul is not None:
            train['RUL'] = train['RUL'].clip(upper=clip_rul)
            test['RUL'] = test['RUL'].clip(upper=clip_rul)
        train_path = OUT_DIR / f'cmapss_{subset}_train.parquet'
        test_path = OUT_DIR / f'cmapss_{subset}_test.parquet'
        train.to_parquet(train_path, index=False)
        test.to_parquet(test_path, index=False)
        summary.append({
            'subset': subset,
            'train_rows': len(train), 'train_units': train['unit'].nunique(),
            'test_rows': len(test), 'test_units': test['unit'].nunique(),
            'rul_min': int(train['RUL'].min()), 'rul_max': int(train['RUL'].max()),
        })
    return pd.DataFrame(summary)


if __name__ == '__main__':
    df = save_all_subsets()
    print(df.to_string(index=False))
    print(f'\nSaved to {OUT_DIR}')
