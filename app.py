"""
app.py — Cached prediction version
Runs prediction at startup, serves cached result instantly
Avoids Render's 30s HTTP timeout
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

# ── Pre-compute predictions for all day options at startup ────
print("[STARTUP] Pre-computing predictions...")
CACHE = {}
CONF  = ['HIGH','MEDIUM','LOW','POOR','POOR']

for days in [1, 2, 3, 4, 5]:
    try:
        preds = recursive_predict(model, last_sequence, target_scaler, days)
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
        CACHE[days] = forecast
        print(f"[STARTUP] Cached {days}-day forecast ✓")
    except Exception as e:
        print(f"[STARTUP] Failed to cache {days}-day: {e}")

print(f"[STARTUP] Ready — {len(CACHE)} forecasts cached")


@app.route('/', methods=['GET'])
def index():
    return jsonify({'service':'Stock LSTM API','status':'running'})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status':'ok','model':'loaded','cached_days':list(CACHE.keys())})

@app.route('/predict', methods=['POST','OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        body = request.get_json(force=True) or {}
        days = max(1, min(int(body.get('days', 2)), 5))

        if days in CACHE:
            return jsonify({
                'status':'ok','model':'LSTM',
                'days':days,'forecast':CACHE[days]
            })
        else:
            return jsonify({'status':'error',
                'message':f'No cached forecast for {days} days'}), 500

    except Exception as e:
        return jsonify({'status':'error','message':str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',5050)))
