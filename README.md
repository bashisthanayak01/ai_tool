# Crypto AI Analytics Platform

A real-time crypto trading intelligence system with AI scoring, whale detection, RL optimization, and a live Streamlit dashboard.

## Features

- 📡 **Live Analytics** — Scans 90 top USDT pairs every 5 minutes
- 🏆 **Top Opportunities** — AI-ranked coins with smart Entry/SL/TP levels
- 📊 **Backtesting** — 90-day historical simulation with adaptive strategy
- 🧠 **AI Learning** — Self-learning engine + RL optimizer (daily)
- 🎯 **Strategy Intelligence** — Auto-optimization via grid search
- 🐋 **Whale Intelligence** — Large capital movement detection (5 Binance sources)

## AI Formula

```
base_score  = Technical(70%) + News(30%)
whale_adj   = base_score + (whale_score_norm × WHALE_WEIGHT)
final_score = whale_adj × Regime_Multiplier × RL_Weight
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure credentials
```bash
cp config/settings.template.py config/settings.py
# Edit config/settings.py and add your MongoDB Atlas URI
```

### 3. Run the scheduler (keeps data live)
```bash
python scheduler.py
```

### 4. Open the dashboard
```bash
python -m streamlit run dashboard.py
```
Then open http://localhost:8501

## Project Structure

```
crypto_ai_tool/
├── ai/                    # Core AI engines
│   ├── whale_tracker.py   # Whale intelligence (5 Binance sources)
│   ├── smart_levels.py    # AI-driven Entry/SL/TP (ATR-based)
│   ├── market_regime.py   # BULL/BEAR/SIDEWAYS detection
│   ├── rl_optimizer.py    # Reinforcement Learning optimizer
│   └── probability_engine.py
├── services/              # Data pipeline
│   ├── market_pipeline.py # Main 90-coin scan orchestrator
│   ├── ai_score.py        # Final score formula
│   ├── indicator_engine.py# RSI/EMA/MACD/ATR/S&R
│   └── news_service.py    # News sentiment (3-tier cache)
├── backtesting/           # Historical simulation
├── optimization/          # Strategy auto-optimization
├── database/              # MongoDB client
├── config/
│   ├── settings.template.py  # Copy this → settings.py
│   └── settings.py           # YOUR credentials (not in git)
├── scheduler.py           # APScheduler — runs all jobs
├── dashboard.py           # Streamlit 6-tab UI
└── learning_engine.py     # Self-learning weight adaptation
```

## Data Sources (100% Free)
- **Binance Public API** — klines, aggTrades, order book, depth
- **CryptoPanic / CoinGecko** — news sentiment
- **MongoDB Atlas** — free tier cloud database

## How It Runs Automatically

| Job | Frequency |
|---|---|
| Market scan (90 coins) | Every 5 min |
| Whale intelligence scan | Every 10 min |
| Regime snapshot | Every 1 hour |
| RL Learning | Every 24 hours |
| Self-Learning | Every 7 days |
| Strategy Optimizer | Every 7 days |

> **Note:** All jobs run while `python scheduler.py` is active.

## ⚠️ Disclaimer
This tool is for educational purposes only. It does not provide financial advice. Always do your own research before trading.
