"""
app.py — Dynamic Stock LSTM API v3
Upload ANY stock CSV → trains fresh LSTM on it → saves model → predicts
Cached: same stock uploaded again skips retraining
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle, json, os, hashlib, threading, uuid, warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)
CORS(app)

BASE      = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE, 'model_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

WINDOW       = 20
FEATURE_COLS = [
    'Open','High','Low','Close','Volume',
    'Return_1','Return_3','Return_5',
    'MA_5','MA_10','MA_20','EMA_9','EMA_21',
    'Volatility_5','Volatility_10',
    'HL_Spread','OC_Spread',
    'Momentum_3','Momentum_5',
    'Volume_MA5','Volume_Ratio',
    'RSI','Gap'
]
TARGET_COLS = ['Open','High','Low','Close']

# job tracker
jobs = {}

print("[STARTUP] Dynamic LSTM API v3 ready")


def clean_data(rows):
    df = pd.DataFrame(rows)
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
    df = df.sort_values('Date').reset_index(drop=True)
    for col in ['Open','High','Low','Close','Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna().drop_duplicates(subset='Date').reset_index(drop=True)


def create_features(df):
    df = df.copy()
    df['Return_1']     = df['Close'].pct_change(1)
    df['Return_3']     = df['Close'].pct_change(3)
    df['Return_5']     = df['Close'].pct_change(5)
    df['MA_5']         = df['Close'].rolling(5).mean()
    df['MA_10']        = df['Close'].rolling(10).mean()
    df['MA_20']        = df['Close'].rolling(20).mean()
    df['EMA_9']        = df['Close'].ewm(span=9,  adjust=False).mean()
    df['EMA_21']       = df['Close'].ewm(span=21, adjust=False).mean()
    df['Volatility_5'] = df['Close'].rolling(5).std()
    df['Volatility_10']= df['Close'].rolling(10).std()
    df['HL_Spread']    = df['High'] - df['Low']
    df['OC_Spread']    = df['Close'] - df['Open']
    df['Momentum_3']   = df['Close'] - df['Close'].shift(3)
    df['Momentum_5']   = df['Close'] - df['Close'].shift(5)
    df['Volume_MA5']   = df['Volume'].rolling(5).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA5'] + 1e-9)
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-9)
    return df.dropna().reset_index(drop=True)


def create_sequences(feat, tgt):
    X, y = [], []
    for i in range(len(feat) - WINDOW):
        X.append(feat[i:i+WINDOW])
        y.append(tgt[i+WINDOW])
    return np.array(X), np.array(y)


def build_model(shape):
    m = Sequential([
        Input(shape=shape),
        LSTM(64, return_sequences=True), Dropout(0.2),
        LSTM(32), Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(4)
    ])
    m.compile(optimizer='adam', loss='mse')
    return m


def predict_future(model, seq, t_scaler, days):
    cur = seq.copy()
    psc = []
    ci  = FEATURE_COLS.index('Close')
    oi  = FEATURE_COLS.index('Open')
    hi  = FEATURE_COLS.index('High')
    li  = FEATURE_COLS.index('Low')
    for _ in range(days):
        p = model.predict(cur.reshape(1,*cur.shape), verbose=0)[0]
        psc.append(p)
        nr = cur[-1].copy()
        nr[oi]=p[0]; nr[hi]=p[1]; nr[li]=p[2]; nr[ci]=p[3]
        wc = np.append(cur[1:,ci], p[3])
        nr[FEATURE_COLS.index('MA_5')]  = wc[-5:].mean()
        nr[FEATURE_COLS.index('MA_10')] = wc[-10:].mean() if len(wc)>=10 else wc.mean()
        nr[FEATURE_COLS.index('MA_20')] = wc.mean()
        cur = np.vstack([cur[1:], nr])
    return t_scaler.inverse_transform(np.array(psc))


def stock_id(rows):
    key = f"{rows[0]['Date']}_{rows[-1]['Date']}_{len(rows)}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def load_cache(sid):
    paths = [f'{sid}_model.h5',f'{sid}_fs.pkl',f'{sid}_ts.pkl',f'{sid}_seq.npy',f'{sid}_meta.json']
    if all(os.path.exists(os.path.join(CACHE_DIR,p)) for p in paths):
        m  = load_model(os.path.join(CACHE_DIR,f'{sid}_model.h5'), compile=False)
        fs = pickle.load(open(os.path.join(CACHE_DIR,f'{sid}_fs.pkl'),'rb'))
        ts = pickle.load(open(os.path.join(CACHE_DIR,f'{sid}_ts.pkl'),'rb'))
        sq = np.load(os.path.join(CACHE_DIR,f'{sid}_seq.npy'))
        with open(os.path.join(CACHE_DIR,f'{sid}_meta.json')) as f: meta=json.load(f)
        return m,fs,ts,sq,meta
    return None


def save_cache(sid, model, fs, ts, seq, meta):
    model.save(os.path.join(CACHE_DIR,f'{sid}_model.h5'))
    pickle.dump(fs, open(os.path.join(CACHE_DIR,f'{sid}_fs.pkl'),'wb'))
    pickle.dump(ts, open(os.path.join(CACHE_DIR,f'{sid}_ts.pkl'),'wb'))
    np.save(os.path.join(CACHE_DIR,f'{sid}_seq.npy'), seq)
    with open(os.path.join(CACHE_DIR,f'{sid}_meta.json'),'w') as f: json.dump(meta,f)


def format_forecast(future, last_close, last_date, days):
    CONF  = ['HIGH','MEDIUM','LOW','POOR','POOR']
    bdays = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=days)
    out   = []
    prev  = last_close
    for i, row in enumerate(future):
        chg = round((float(row[3])-prev)/prev*100, 4)
        out.append({
            'day':i+1, 'date':bdays[i].strftime('%Y-%m-%d'),
            'open':round(float(row[0]),4), 'high':round(float(row[1]),4),
            'low':round(float(row[2]),4),  'close':round(float(row[3]),4),
            'change_pct':chg, 'confidence':CONF[min(i,4)]
        })
        prev = float(row[3])
    return out


# ── BACKGROUND TRAINING ───────────────────────────────────────
def run_pipeline(job_id, rows, days, stock_name):
    def upd(s, p): jobs[job_id]['status']=s; jobs[job_id]['progress']=p
    try:
        sid = stock_id(rows)
        upd('cleaning data', 10)
        df = clean_data(rows)
        if len(df) < 60: raise ValueError(f"Need 60+ rows. Got {len(df)}.")

        upd('engineering features', 20)
        df = create_features(df)

        upd('scaling', 30)
        fs = MinMaxScaler(); ts = MinMaxScaler()
        dfs = df.copy()
        dfs[FEATURE_COLS] = fs.fit_transform(dfs[FEATURE_COLS])
        dfs[TARGET_COLS]  = ts.fit_transform(dfs[TARGET_COLS])

        upd('creating sequences', 40)
        X, y = create_sequences(dfs[FEATURE_COLS].values, dfs[TARGET_COLS].values)

        upd('training LSTM (25 epochs)', 50)
        split = int(len(X)*0.8)
        model = build_model((X.shape[1], X.shape[2]))
        es    = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
        model.fit(X[:split], y[:split],
                  validation_data=(X[split:], y[split:]),
                  epochs=25, batch_size=32, callbacks=[es], verbose=0)

        upd('evaluating', 80)
        preds   = ts.inverse_transform(model.predict(X[split:], verbose=0))
        actuals = ts.inverse_transform(y[split:])
        ci      = TARGET_COLS.index('Close')
        mae     = round(mean_absolute_error(actuals[:,ci], preds[:,ci]), 4)
        pct_err = round(float(np.abs((preds[:,ci]-actuals[:,ci])/actuals[:,ci]).mean()*100), 2)

        upd('predicting', 90)
        last_seq   = dfs[FEATURE_COLS].values[-WINDOW:]
        last_close = float(df['Close'].iloc[-1])
        last_date  = df['Date'].iloc[-1]
        future     = predict_future(model, last_seq, ts, days)

        meta = {'stock_name':stock_name,'rows':len(rows),
                'last_close':last_close,'last_date':str(last_date.date()),
                'mae':mae,'pct_err':pct_err}
        save_cache(sid, model, fs, ts, last_seq, meta)

        upd('done', 100)
        jobs[job_id]['result'] = {
            'status':'ok', 'stock':stock_name, 'rows_used':len(df),
            'mae':mae, 'pct_error':f'{pct_err}%',
            'last_close':last_close, 'last_date':str(last_date.date()),
            'forecast': format_forecast(future, last_close, last_date, days)
        }
    except Exception as e:
        jobs[job_id]['status']='error'; jobs[job_id]['error']=str(e)
        print(f"[ERROR] {job_id}: {e}")


# ── ROUTES ────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    cached = len([f for f in os.listdir(CACHE_DIR) if f.endswith('_meta.json')])
    return jsonify({'status':'ok','models_cached':cached})


@app.route('/train', methods=['POST'])
def train():
    """
    Body: {"days":2, "stock_name":"NVIDIA", "rows":[...]}
    If cached → returns instantly
    If new    → returns job_id, poll /status/<job_id>
    """
    try:
        body       = request.get_json(force=True) or {}
        days       = max(1, min(int(body.get('days',2)), 5))
        rows       = body.get('rows', [])
        stock_name = body.get('stock_name', 'Stock')

        if len(rows) < 60:
            return jsonify({'status':'error',
                'message':f'Need 60+ rows. Got {len(rows)}.'}), 400

        sid = stock_id(rows)

        # Cache hit — instant response
        cached = load_cache(sid)
        if cached:
            m,fs,ts,seq,meta = cached
            last_close = meta['last_close']
            last_date  = pd.Timestamp(meta['last_date'])
            future     = predict_future(m, seq, ts, days)
            return jsonify({
                'status':'ok','cached':True,'stock':stock_name,
                'rows_used':len(rows),'mae':meta.get('mae'),
                'pct_error':f"{meta.get('pct_err')}%",
                'last_close':last_close,'last_date':meta['last_date'],
                'forecast':format_forecast(future, last_close, last_date, days)
            })

        # New stock — train in background
        job_id = str(uuid.uuid4())[:8]
        jobs[job_id] = {'status':'queued','progress':0,'result':None,'error':None}
        threading.Thread(target=run_pipeline,
                         args=(job_id,rows,days,stock_name), daemon=True).start()

        return jsonify({
            'status':'training_started','job_id':job_id,
            'message':f'Training LSTM on {len(rows)} rows of {stock_name}',
            'poll':f'/status/{job_id}','estimate':'2-4 minutes'
        }), 202

    except Exception as e:
        return jsonify({'status':'error','message':str(e)}), 500


@app.route('/status/<job_id>', methods=['GET'])
def status(job_id):
    if job_id not in jobs:
        return jsonify({'status':'error','message':'Job not found'}), 404
    job = jobs[job_id]
    if job['status'] == 'done':   return jsonify(job['result'])
    if job['status'] == 'error':  return jsonify({'status':'error','message':job['error']}), 500
    return jsonify({'status':'training','step':job['status'],'progress':job['progress']})


@app.route('/train',            methods=['OPTIONS'])
@app.route('/status/<job_id>', methods=['OPTIONS'])
def options(job_id=None): return '', 204


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT',5050)))
