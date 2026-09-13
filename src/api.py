"""Flask inference API (PDF Section 7.6).

Endpoints:
    GET  /health              -> simple liveness probe
    POST /predict/anomaly     -> Isolation Forest anomaly score (CMAPSS-trained)
    POST /predict/rul_cmapss  -> XGBoost RUL (cycles, CMAPSS-trained)
    POST /predict/rul_femto   -> XGBoost RUL (fraction, FEMTO-trained)
    POST /predict/fault       -> RandomForest multi-class + SVM binary (AI4I)

Request bodies: JSON {"features": {featureName: value, ...}}
Missing features are filled with 0 (with warning in response).

Run:   python src/api.py     (dev server on :5000)
Test:  curl -s localhost:5000/health
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify

try:
    import tensorflow as tf
    HAS_TF = True
except Exception:
    HAS_TF = False

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'models'

app = Flask(__name__)

# ---- Load all models at startup -----------------------------------------

def _load(name):
    p = MODELS / name
    return joblib.load(p) if p.exists() else None


print('Loading models...')
IFOREST = _load('iforest_fd001.joblib')
XGB_CMAPSS = _load('xgb_rul_fd001.joblib')
XGB_FEMTO = _load('xgb_rul_femto.joblib')
SVM_BIN = _load('svm_binary_ai4i.joblib')
RF_MULTI = _load('rf_multiclass_ai4i.joblib')

LSTM_AE = None
LSTM_PRE = None
if HAS_TF and (MODELS / 'lstm_ae_fd001.keras').exists():
    LSTM_AE = tf.keras.models.load_model(MODELS / 'lstm_ae_fd001.keras', compile=False)
    LSTM_PRE = joblib.load(MODELS / 'lstm_ae_fd001_preproc.joblib')

loaded = {
    'iforest_fd001': IFOREST is not None,
    'xgb_rul_cmapss': XGB_CMAPSS is not None,
    'xgb_rul_femto': XGB_FEMTO is not None,
    'svm_binary_ai4i': SVM_BIN is not None,
    'rf_multiclass_ai4i': RF_MULTI is not None,
    'lstm_ae_fd001': LSTM_AE is not None,
}
print('Loaded:', loaded)


# ---- Helpers ------------------------------------------------------------

def _vectorise(feat_dict: dict, feature_list: list):
    """Build a 1-row dataframe with columns in the order the model expects."""
    missing = [c for c in feature_list if c not in feat_dict]
    row = {c: float(feat_dict.get(c, 0.0)) for c in feature_list}
    df = pd.DataFrame([row])
    return df, missing


# ---- Routes -------------------------------------------------------------

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'models_loaded': loaded})


@app.post('/predict/anomaly')
def predict_anomaly():
    body = request.get_json(force=True) or {}
    feats = body.get('features', {})
    if IFOREST is None:
        return jsonify({'error': 'iforest not loaded'}), 503
    bundle = IFOREST
    X, missing = _vectorise(feats, bundle['features'])
    X_s = bundle['scaler'].transform(X)
    score = float(bundle['model'].score_samples(X_s)[0])
    return jsonify({
        'anomaly_score': score,
        'interpretation': 'lower is more anomalous',
        'missing_features': missing,
    })


@app.post('/predict/rul_cmapss')
def predict_rul_cmapss():
    body = request.get_json(force=True) or {}
    feats = body.get('features', {})
    if XGB_CMAPSS is None:
        return jsonify({'error': 'xgb_cmapss not loaded'}), 503
    # XGB_CMAPSS is the bare model — feature list lives in meta JSON
    import json
    with open(MODELS / 'xgb_rul_fd001_meta.json') as f:
        meta = json.load(f)
    X, missing = _vectorise(feats, meta['feature_cols'])
    rul = float(XGB_CMAPSS.predict(X)[0])
    return jsonify({
        'rul_cycles': rul,
        'missing_features': missing,
    })


@app.post('/predict/rul_femto')
def predict_rul_femto():
    body = request.get_json(force=True) or {}
    feats = body.get('features', {})
    if XGB_FEMTO is None:
        return jsonify({'error': 'xgb_femto not loaded'}), 503
    bundle = XGB_FEMTO
    X, missing = _vectorise(feats, bundle['features'])
    frac = float(bundle['model'].predict(X)[0])
    return jsonify({
        'rul_fraction': frac,
        'interpretation': '0 = failure now, 1 = fresh bearing',
        'missing_features': missing,
    })


@app.post('/predict/fault')
def predict_fault():
    body = request.get_json(force=True) or {}
    feats = body.get('features', {})
    if RF_MULTI is None or SVM_BIN is None:
        return jsonify({'error': 'fault classifiers not loaded'}), 503
    rf_bundle, svm_bundle = RF_MULTI, SVM_BIN
    X, missing = _vectorise(feats, rf_bundle['features'])
    X_s = rf_bundle['scaler'].transform(X)

    rf_probs = rf_bundle['model'].predict_proba(X_s)[0]
    rf_class = rf_bundle['classes'][int(np.argmax(rf_probs))]
    svm_prob_fault = float(svm_bundle['model'].predict_proba(X_s)[0][1])

    return jsonify({
        'binary_fault_prob': svm_prob_fault,
        'fault_class': rf_class,
        'class_probabilities': dict(zip(rf_bundle['classes'], [float(p) for p in rf_probs])),
        'missing_features': missing,
    })


@app.post('/predict/anomaly_lstm')
def predict_anomaly_lstm():
    body = request.get_json(force=True) or {}
    window = body.get('window')  # expect list shape (timesteps, n_features)
    if LSTM_AE is None:
        return jsonify({'error': 'lstm_ae not loaded'}), 503
    arr = np.asarray(window, dtype=np.float32)
    pre = LSTM_PRE
    if arr.ndim != 2 or arr.shape[0] != pre['timesteps'] or arr.shape[1] != len(pre['sensor_cols']):
        return jsonify({
            'error': f'window must be shape ({pre["timesteps"]}, {len(pre["sensor_cols"])}) ',
            'got_shape': list(arr.shape),
        }), 400
    arr_s = pre['scaler'].transform(arr)[None, ...]
    recon = LSTM_AE(arr_s, training=False).numpy()
    mae = float(np.mean(np.abs(arr_s - recon)))
    return jsonify({
        'recon_mae': mae,
        'threshold': pre['threshold'],
        'is_anomaly': mae > pre['threshold'],
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
