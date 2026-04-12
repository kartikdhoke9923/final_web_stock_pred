"""
app.py — Flask API for Render deployment
Loads saved LSTM model + scalers, serves predictions via REST API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import json
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# ── Load model + scalers on startup ──────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, 'model')

print("[STARTUP] Loading model and scalers...")

model          = load_model(os.path.join(MODEL_DIR, 'lstm_model.h5'), compile=False)
feature_scaler = pickle.load(open(os.path.join(MODEL_DIR, 'feature_scaler.pkl'), 'rb'))
target_scaler  = pickle.load(open(os.path.join(MODEL_DIR, 'target_scaler.pkl'), 'rb'))
last_sequence  = np.load(os.path.join(MODEL_DIR, 'last_sequence.npy'))

with open(os.path.join(MODEL_DIR, 'columns.json')) as f:
    cols = json.load(f)

FEATURE_COLS = cols['feature_cols']
TARGET_COLS  = cols['target_cols']
WINDOW       = last_sequence.shape[0]

print(f"[STARTUP] Model loaded | window={WINDOW} | features={len(FEATURE_COLS)}")


# ── Recursive prediction ──────────────────────────────────────
def predict_future(sequence, days=2):
    current_seq        = sequence.copy()
    predictions_scaled = []

    c_idx = FEATURE_COLS.index('Close')
    o_idx = FEATURE_COLS.index('Open')
    h_idx = FEATURE_COLS.index('High')
    l_idx = FEATURE_COLS.index('Low')

    for _ in range(days):
        pred = model.predict(
            current_seq.reshape(1, *current_seq.shape), verbose=0
        )[0]
        predictions_scaled.append(pred)

        new_row        = current_seq[-1].copy()
        new_row[o_idx] = pred[0]
        new_row[h_idx] = pred[1]
        new_row[l_idx] = pred[2]
        new_row[c_idx] = pred[3]

        wc = np.append(current_seq[1:, c_idx], pred[3])
        new_row[FEATURE_COLS.index('MA_5')]  = wc[-5:].mean()
        new_row[FEATURE_COLS.index('MA_10')] = wc[-10:].mean() if len(wc)>=10 else wc.mean()
        new_row[FEATURE_COLS.index('MA_20')] = wc.mean()

        current_seq = np.vstack([current_seq[1:], new_row])

    return target_scaler.inverse_transform(np.array(predictions_scaled))


# ── ROUTES ────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'Stock LSTM API',
        'status':  'running',
        'routes': {
            'GET  /health':  'health check',
            'POST /predict': 'get OHLC forecast'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'loaded'})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Body: { "days": 2 }
    Returns OHLC forecast per day with confidence level.
    """
    try:
        body = request.get_json(force=True) or {}
        days = int(body.get('days', 2))
        days = max(1, min(days, 5))

        preds = predict_future(last_sequence, days)

        CONF  = ['HIGH', 'MEDIUM', 'LOW', 'POOR', 'POOR']
        today = pd.Timestamp.today()
        bdays = pd.bdate_range(start=today + pd.Timedelta(days=1), periods=days)

        prev_close = float(target_scaler.inverse_transform(
            last_sequence[-1:, :len(TARGET_COLS)]
        )[0][TARGET_COLS.index('Close')])

        forecast = []
        prev = prev_close
        for i, row in enumerate(preds):
            chg = round((float(row[3]) - prev) / prev * 100, 4)
            forecast.append({
                'day':        i + 1,
                'date':       bdays[i].strftime('%Y-%m-%d'),
                'open':       round(float(row[0]), 4),
                'high':       round(float(row[1]), 4),
                'low':        round(float(row[2]), 4),
                'close':      round(float(row[3]), 4),
                'change_pct': chg,
                'confidence': CONF[min(i, len(CONF)-1)]
            })
            prev = float(row[3])

        return jsonify({
            'status':   'ok',
            'model':    'LSTM',
            'days':     days,
            'forecast': forecast
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/predict', methods=['OPTIONS'])
def predict_options():
    return '', 204


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port)
