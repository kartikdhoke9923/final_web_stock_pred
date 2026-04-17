import os, pickle, json, warnings, threading, time
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from pipeline import FEATURE_COLS, TARGET_COLS, recursive_predict, create_features

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

# ── Load model + scalers once ─────────────────────────────────
print("[STARTUP] Loading model...")
model = load_model(os.path.join(MODEL_DIR, 'lstm_model.h5'), compile=False)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    feature_scaler = pickle.load(open(os.path.join(MODEL_DIR,'feature_scaler.pkl'),'rb'))
    target_scaler  = pickle.load(open(os.path.join(MODEL_DIR,'target_scaler.pkl'),'rb'))

fallback_sequence = np.load(os.path.join(MODEL_DIR,'last_sequence.npy'))
WINDOW = fallback_sequence.shape[0]

with open(os.path.join(MODEL_DIR,'columns.json')) as f:
    cols = json.load(f)

print(f"[STARTUP] Model loaded | window={WINDOW} | features={len(FEATURE_COLS)}")

# ── Global state ──────────────────────────────────────────────
CACHE          = {}          # {days: forecast_list}
LAST_UPDATED   = None        # timestamp of last Yahoo fetch
LAST_CLOSE     = None        # latest close price
DATA_SOURCE    = 'fallback'  # 'yahoo' or 'fallback'


# ── Fetch latest data from Yahoo Finance ─────────────────────
def fetch_latest_data(ticker='VOW3.DE', period='6mo'):
    """
    Downloads last 6 months of VW data from Yahoo Finance.
    Builds feature sequence and returns last WINDOW rows scaled.
    Falls back to saved sequence if Yahoo fails.
    """
    global DATA_SOURCE, LAST_CLOSE

    try:
        print(f"[YAHOO] Fetching {ticker} data...")
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)

        if df.empty or len(df) < WINDOW + 30:
            raise ValueError(f"Not enough data: {len(df)} rows")

        # Flatten multi-index columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()
        df.columns = [str(c) for c in df.columns]
        df = df.rename(columns={'index':'Date','Price':'Close'} if 'Price' in df.columns else {})

        # Ensure correct columns
        for col in ['Open','High','Low','Close','Volume']:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # Feature engineering
        df = create_features(df)

        # Scale using saved scalers
        df_scaled = df.copy()
        df_scaled[FEATURE_COLS] = feature_scaler.transform(df_scaled[FEATURE_COLS])

        # Get last WINDOW rows as sequence
        sequence   = df_scaled[FEATURE_COLS].values[-WINDOW:]
        last_close = float(df['Close'].iloc[-1])
        last_date  = df['Date'].iloc[-1]

        LAST_CLOSE  = last_close
        DATA_SOURCE = 'yahoo'

        print(f"[YAHOO] ✓ {ticker} | Last close: {last_close:.2f} | Date: {last_date.date()}")
        return sequence, last_close, last_date

    except Exception as e:
        print(f"[YAHOO] Failed: {e} — using fallback sequence")
        DATA_SOURCE = 'fallback'

        # Use saved last_sequence
        prev_close = float(target_scaler.inverse_transform(
            fallback_sequence[-1:, :len(TARGET_COLS)]
        )[0][TARGET_COLS.index('Close')])
        LAST_CLOSE = prev_close
        return fallback_sequence.copy(), prev_close, pd.Timestamp.today()


# ── Build forecast from sequence ─────────────────────────────
def build_forecast(sequence, last_close, last_date):
    global CACHE
    CONF  = ['HIGH','MEDIUM','LOW','POOR','POOR']
    new_cache = {}

    for days in [1, 2, 3, 4, 5]:
        try:
            preds = recursive_predict(model, sequence, target_scaler, days)
            bdays = pd.bdate_range(
                start=last_date + pd.Timedelta(days=1),
                periods=days
            )
            forecast, prev = [], last_close
            for i, row in enumerate(preds):
                chg = round((float(row[3])-prev)/prev*100, 4)
                forecast.append({
                    'day':i+1,
                    'date':bdays[i].strftime('%Y-%m-%d'),
                    'open':round(float(row[0]),4),
                    'high':round(float(row[1]),4),
                    'low':round(float(row[2]),4),
                    'close':round(float(row[3]),4),
                    'change_pct':chg,
                    'confidence':CONF[min(i,4)]
                })
                prev = float(row[3])
            new_cache[days] = forecast
            print(f"[CACHE] {days}-day forecast cached ✓")
        except Exception as e:
            print(f"[CACHE] {days}-day failed: {e}")

    CACHE = new_cache
    print(f"[CACHE] All forecasts updated | source={DATA_SOURCE} | close={last_close:.2f}")


# ── Daily refresh loop (runs in background thread) ────────────
def daily_refresh():
    global LAST_UPDATED
    while True:
        try:
            sequence, last_close, last_date = fetch_latest_data()
            build_forecast(sequence, last_close, last_date)
            LAST_UPDATED = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')
        except Exception as e:
            print(f"[REFRESH] Error: {e}")

        # Refresh every 6 hours
        time.sleep(24 * 60 * 60)


# ── Initial load at startup ───────────────────────────────────
print("[STARTUP] Running initial data fetch...")
try:
    seq, lc, ld = fetch_latest_data()
    build_forecast(seq, lc, ld)
    LAST_UPDATED = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')
    print("[STARTUP] ✓ Initial forecast ready")
except Exception as e:
    print(f"[STARTUP] Initial fetch failed: {e}")

# Start background refresh thread
refresh_thread = threading.Thread(target=daily_refresh, daemon=True)
refresh_thread.start()
print("[STARTUP] Background refresh thread started")


# ── ROUTES ────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service':      'Stock LSTM API — Auto-updating',
        'status':       'running',
        'data_source':  DATA_SOURCE,
        'last_updated': LAST_UPDATED,
        'last_close':   LAST_CLOSE
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':       'ok',
        'model':        'loaded',
        'data_source':  DATA_SOURCE,
        'last_updated': LAST_UPDATED,
        'last_close':   LAST_CLOSE,
        'cached_days':  list(CACHE.keys())
    })


@app.route('/predict', methods=['POST','OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        body = request.get_json(force=True) or {}
        days = max(1, min(int(body.get('days', 2)), 5))

        if not CACHE:
            return jsonify({'status':'error',
                'message':'Forecast not ready yet — try again in 30 seconds'}), 503

        if days not in CACHE:
            return jsonify({'status':'error',
                'message':f'No forecast for {days} days'}), 500

        return jsonify({
            'status':       'ok',
            'model':        'LSTM',
            'data_source':  DATA_SOURCE,
            'last_updated': LAST_UPDATED,
            'last_close':   LAST_CLOSE,
            'days':         days,
            'forecast':     CACHE[days]
        })

    except Exception as e:
        return jsonify({'status':'error','message':str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',5050)))
