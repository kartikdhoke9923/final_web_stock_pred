# 📈 VW Stock Forecast — LSTM Prediction Pipeline

> Predicts Volkswagen stock **Open · High · Low · Close** for the next 1–2 business days using a deep learning LSTM model, served via a live REST API and an interactive frontend.

**Live Demo → [kartikworks.co.in/tools/stock-forecast.html](https://kartikworks.co.in/tools/stock-forecast.html)**  


---

## Preview

![Stock Forecast Tool](https://img.shields.io/badge/Status-Live-brightgreen) ![Python](https://img.shields.io/badge/Python-3.11-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange) ![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey) ![Render](https://img.shields.io/badge/Deployed-Render-purple)

---

## What This Project Does

- Takes **15 years of Volkswagen (VOW.DE) OHLCV** historical data
- Engineers **23 technical indicators** — RSI, EMA, momentum, overnight gap, volume ratio
- Trains an **LSTM neural network** with 4 simultaneous outputs (Open, High, Low, Close)
- **Recursively predicts** next 1–2 business days with confidence scoring
- Serves predictions via a **Flask REST API** deployed permanently on Render
- **Interactive frontend** in pure HTML/CSS/JS with Chart.js visualization

---

## Project Structure

```
final_web_stock_pred/
│
├── app.py                    # Flask REST API — loads model, serves predictions
├── requirements.txt          # Python dependencies
├── render.yaml               # Render deployment configuration
│
├── model/
│   ├── lstm_model.h5         # Trained LSTM model (saved without optimizer)
│   ├── feature_scaler.pkl    # MinMaxScaler for 23 input features
│   ├── target_scaler.pkl     # MinMaxScaler for OHLC targets
│   ├── last_sequence.npy     # Last 20-day input sequence for recursive prediction
│   └── columns.json          # Feature and target column configuration
│
└── stock-forecast.html       # Full frontend UI (HTML + CSS + JS)
```

---

## Model Architecture

```
Input: (20 days × 23 features)
         ↓
LSTM Layer 1 — 128 units, return_sequences=True
         ↓
Dropout — 0.2
         ↓
LSTM Layer 2 — 64 units
         ↓
Dropout — 0.2
         ↓
Dense — 32 units, ReLU
         ↓
Output — 4 units (Open, High, Low, Close)
```

| Parameter | Value |
|-----------|-------|
| Lookback window | 20 days |
| Input features | 23 |
| Output | 4 (OHLC) |
| Optimizer | Adam |
| Loss | MSE |
| Train/Test split | 80/20 chronological |
| Early stopping | patience=10 on val_loss |

---

## Feature Engineering (23 Features)

| Category | Features |
|----------|----------|
| Price | Open, High, Low, Close, Volume |
| Returns | Return_1, Return_3, Return_5 |
| Moving Averages | MA_5, MA_10, MA_20, EMA_9, EMA_21 |
| Volatility | Volatility_5, Volatility_10 |
| Spread | HL_Spread, OC_Spread |
| Momentum | Momentum_3, Momentum_5 |
| Volume | Volume_MA5, Volume_Ratio |
| Indicators | RSI_14, Gap (overnight) |

---


### GET /health
```json
{
  "model": "loaded",
  "status": "ok"
}
```

### POST /predict
**Request:**
```json
{
  "days": 2
}
```

**Response:**
```json
{
  "status": "ok",
  "model": "LSTM",
  "days": 2,
  "forecast": [
    {
      "day": 1,
      "date": "2026-04-14",
      "open": 127.70,
      "high": 132.83,
      "low": 131.56,
      "close": 132.07,
      "change_pct": 3.46,
      "confidence": "HIGH"
    },
    {
      "day": 2,
      "date": "2026-04-15",
      "open": 132.08,
      "high": 132.76,
      "low": 131.36,
      "close": 131.92,
      "change_pct": -0.11,
      "confidence": "MEDIUM"
    }
  ]
}
```

---

## 📊 Confidence Levels

| Day | Confidence | Reason |
|-----|-----------|--------|
| Day 1 | HIGH | Single-step prediction, minimal error compounding |
| Day 2 | MEDIUM | One recursive step, slight error accumulation |
| Day 3 | LOW | Multiple recursive steps, increasing uncertainty |
| Day 4–5 | POOR | Directional estimate only |
| Day 6+ | DISABLED | Signal degrades to noise |

---

## Run Locally

**1 — Clone the repo**
```bash
git clone https://github.com/kartikdhoke9923/final_web_stock_pred.git
cd final_web_stock_pred
```

**2 — Install dependencies**
```bash
pip install -r requirements.txt
```


**3 — Run the API**
```bash
python app.py
```

**4 — Test it**
```bash
curl -X POST http://localhost:5050/predict \
  -H "Content-Type: application/json" \
  -d '{"days": 2}'
```

**5 — Open the frontend**  
Open `stock-forecast.html` in your browser → update `LSTM_API_URL` to `http://localhost:5050`

---

## Retrain the Model

1. Open `stock_pipeline_colab_v2.py` in Google Colab
2. Upload your updated CSV file
3. Run all cells in order
4. Run `save_model.py` cell to export model files
5. Replace files in `model/` folder
6. Push to GitHub — Render auto-redeploys

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | TensorFlow 2.19 · Keras · LSTM |
| Data Processing | Pandas · NumPy · Scikit-learn |
| API | Python · Flask · Flask-CORS · Gunicorn |
| Frontend | HTML · CSS · JavaScript · Chart.js |
| Deployment | Render (API) · Lovable (Portfolio) |
| Version Control | Git · GitHub |

---

## Disclaimer

This project is built for **educational and portfolio purposes only**.  
LSTM stock predictions carry inherent uncertainty — especially beyond Day 1.  
**Do not use this for actual trading decisions** without validating MAE/RMSE metrics against your acceptable error range.

Day 1 predictions carry the most signal. Beyond Day 2, treat output as directional estimates only.

---

## Author

**Kartik R. Dhoke** — Data Analyst & ML Enthusiast  
[kartikworks.co.in](https://kartikworks.co.in)  
[LinkedIn](https://linkedin.com/in/kartik-dhoke)  
[GitHub](https://github.com/kartikdhoke9923)

---

*Built with Python · TensorFlow · Flask · Deployed on Render*
