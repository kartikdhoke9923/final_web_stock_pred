"""
pipeline.py — Feature engineering + prediction logic
Imported by app.py — keeps backend lean and separated
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

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


def recursive_predict(model, sequence, target_scaler, days=2):
    current_seq        = sequence.astype(np.float32).copy()
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
