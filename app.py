"""
app.py — Minimal Flask API
Loads saved model once, delegates all logic to pipeline.py
"""

import os, pickle, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model

from pipeline import FEATURE_COLS, TARGET_COLS, recursive_predict

app = Flask(__name__)
CORS(app, origins="*", allow_headers=["Content-Type"], methods=["GET","POST","OPTIONS"])

@app.after_request
def add_cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return r

# ── Load once at startup ──────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, 'model')

print("[STARTUP] Loading model...")
model = load_model(os.path.join(MODEL_DIR, 'lstm_model.h5'), compile=False)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    feature_scaler = pickle.load(open(os.path.join(MODEL_DIR,'feature_scaler.pkl'),'rb'))
    target_scaler  = pickle.load(open(os.path.join(MODEL_DIR,'target_scaler.pkl'),'rb'))

last_sequence = np.load(os.path.join(MODEL_DIR,'last_sequence.npy'))
WINDOW        = last_sequence.shape[0]

print(f"[STARTUP] Ready | window={WINDOW} | features={len(FEATURE_COLS)}")

# ── Run one warmup prediction to pre-cache TF graph ──────────
try:
    dummy = last_sequence.astype(np.float32).copy()
    model.predict(dummy.reshape(1,*dummy.shape), verbose=0)
    print("[STARTUP] Warmup prediction done — predict route will be fast")
except Exception as e:
    print(f"[STARTUP] Warmup failed (non-fatal): {e}")


# ── ROUTES ────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    return jsonify({'service':'Stock LSTM API','status':'running'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status':'ok','model':'loaded'})

@app.route('/predict', methods=['POST','OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        body = request.get_json(force=True) or {}
        days = max(1, min(int(body.get('days', 2)), 5))

        preds = recursive_predict(model, last_sequence, target_scaler, days)

        CONF  = ['HIGH','MEDIUM','LOW','POOR','POOR']
        bdays = pd.bdate_range(
            start=pd.Timestamp.today() + pd.Timedelta(days=1),
            periods=days
        )
        prev_close = float(target_scaler.inverse_transform(
            last_sequence[-1:, :len(TARGET_COLS)]
        )[0][TARGET_COLS.index('Close')])

        forecast, prev = [], prev_close
        for i, row in enumerate(preds):
            chg = round((float(row[3])-prev)/prev*100, 4)
            forecast.append({
                'day':i+1, 'date':bdays[i].strftime('%Y-%m-%d'),
                'open':round(float(row[0]),4), 'high':round(float(row[1]),4),
                'low':round(float(row[2]),4),  'close':round(float(row[3]),4),
                'change_pct':chg, 'confidence':CONF[min(i,4)]
            })
            prev = float(row[3])

        return jsonify({'status':'ok','model':'LSTM','days':days,'forecast':forecast})

    except Exception as e:
        return jsonify({'status':'error','message':str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',5050)))
