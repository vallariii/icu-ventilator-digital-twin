"""Fault state classifier on UCI AI4I 2020 dataset (PDF Section 7.5).

AI4I columns → map to the PDF's fault taxonomy:
    Normal (no failure),
    TWF  -> Tool Wear Failure            ~ Bearing Wear
    HDF  -> Heat Dissipation Failure     ~ Cooling Failure
    PWF  -> Power Failure                ~ Motor Overload
    OSF  -> Overstrain Failure           ~ Circuit Leak (overstrain proxy)
    RNF  -> Random Failure               ~ Sensor Fault (rare, random)

Two models per PDF:
    SVM with RBF kernel (C=10, gamma='scale')  — Normal vs. Fault (binary)
    Random Forest (n_estimators=300)            — 6-class fault classification
SMOTE oversampling for imbalance. F1-score per class target > 0.85.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)
from imblearn.over_sampling import SMOTE
import joblib

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / 'data' / 'raw' / 'femto' / 'ai4i2020.csv'
MODELS = ROOT / 'models'
MODELS.mkdir(exist_ok=True)

FAULT_FLAGS = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
FAULT_NAMES = {
    'TWF': 'BearingWear',
    'HDF': 'CoolingFailure',
    'PWF': 'MotorOverload',
    'OSF': 'CircuitLeak',
    'RNF': 'SensorFault',
}


def derive_label(row):
    """Pick the single most informative fault label per row (multi-label → single)."""
    flags = [f for f in FAULT_FLAGS if row[f] == 1]
    if not flags:
        return 'Normal'
    # If multiple flags, prefer the rarest failure mode (most informative)
    priority = {'RNF': 0, 'PWF': 1, 'OSF': 2, 'HDF': 3, 'TWF': 4}
    return FAULT_NAMES[sorted(flags, key=lambda f: priority[f])[0]]


def main():
    df = pd.read_csv(CSV)
    df.columns = [c.strip() for c in df.columns]
    # Clean BOM in first column name if present
    first = df.columns[0]
    if first.startswith('\ufeff'):
        df = df.rename(columns={first: first.lstrip('\ufeff')})
    print(f'AI4I: {df.shape}  columns: {list(df.columns)}')

    df['fault'] = df.apply(derive_label, axis=1)
    print('\nClass distribution:')
    print(df['fault'].value_counts().to_string())

    # Features: numerical sensors + one-hot of product Type
    num_cols = ['Air temperature [K]', 'Process temperature [K]',
                'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    X = df[num_cols].copy()
    X = pd.concat([X, pd.get_dummies(df['Type'], prefix='type').astype(int)], axis=1)
    y_multi = df['fault'].to_numpy()
    y_bin = (df['fault'] != 'Normal').astype(int).to_numpy()

    X_tr, X_te, y_m_tr, y_m_te, y_b_tr, y_b_te = train_test_split(
        X, y_multi, y_bin, test_size=0.2, stratify=y_multi, random_state=42
    )
    print(f'\nTrain: {len(X_tr)}   Test: {len(X_te)}')

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # --- Binary SVM (Normal vs Fault) ---
    print('\n=== Binary SVM (Normal vs Fault) ===')
    sm_bin = SMOTE(random_state=42, k_neighbors=3)
    X_tr_bin, y_tr_bin = sm_bin.fit_resample(X_tr_s, y_b_tr)
    print(f'After SMOTE: {len(X_tr_bin)} rows  '
          f'(pos={(y_tr_bin==1).sum()}, neg={(y_tr_bin==0).sum()})')
    svm = SVC(C=10, gamma='scale', kernel='rbf', probability=True, random_state=42)
    svm.fit(X_tr_bin, y_tr_bin)
    y_pred_bin = svm.predict(X_te_s)
    print(classification_report(y_b_te, y_pred_bin, target_names=['Normal', 'Fault'], digits=3))
    bin_f1 = f1_score(y_b_te, y_pred_bin)
    bin_acc = accuracy_score(y_b_te, y_pred_bin)

    # --- Multi-class Random Forest ---
    print('\n=== Multi-class Random Forest ===')
    # SMOTE struggles when any class has < k_neighbors+1 samples; find smallest
    from collections import Counter
    ctr = Counter(y_m_tr)
    min_k = max(1, min(ctr.values()) - 1)
    k_nb = min(3, min_k)
    print(f'Class counts in train: {dict(ctr)}  (using k_neighbors={k_nb})')
    sm_multi = SMOTE(random_state=42, k_neighbors=k_nb)
    X_tr_mul, y_tr_mul = sm_multi.fit_resample(X_tr_s, y_m_tr)
    print(f'After SMOTE: {len(X_tr_mul)} rows')

    rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                                class_weight='balanced', n_jobs=-1, random_state=42)
    rf.fit(X_tr_mul, y_tr_mul)
    y_pred_mul = rf.predict(X_te_s)
    print(classification_report(y_m_te, y_pred_mul, digits=3))
    multi_f1 = f1_score(y_m_te, y_pred_mul, average='macro')
    multi_acc = accuracy_score(y_m_te, y_pred_mul)
    labels = sorted(df['fault'].unique())
    cm = confusion_matrix(y_m_te, y_pred_mul, labels=labels)
    print(f'\nConfusion matrix (rows=true, cols=pred, labels={labels}):')
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())

    # Persist
    joblib.dump({'model': svm, 'scaler': scaler, 'features': X.columns.tolist()},
                MODELS / 'svm_binary_ai4i.joblib')
    joblib.dump({'model': rf, 'scaler': scaler, 'features': X.columns.tolist(),
                 'classes': rf.classes_.tolist()},
                MODELS / 'rf_multiclass_ai4i.joblib')
    with open(MODELS / 'fault_classifier_meta.json', 'w') as f:
        json.dump({
            'binary_f1': bin_f1, 'binary_accuracy': bin_acc,
            'multi_macro_f1': multi_f1, 'multi_accuracy': multi_acc,
            'classes': labels,
            'confusion_matrix': cm.tolist(),
        }, f, indent=2)
    print(f'\nSaved -> {MODELS}/svm_binary_ai4i.joblib and rf_multiclass_ai4i.joblib')


if __name__ == '__main__':
    main()
