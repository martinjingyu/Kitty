# Kitty — Quantitative Trading Signal Bot

A personal quantitative trading agent that builds alternative financial bars (Volume / Dollar / Runs / Imbalance), trains directional signal models, and delivers real-time trade alerts to Discord.

## Architecture

```
Polygon.io (1-min bars)
        │
        ▼
┌───────────────────┐
│  Bar Construction │  Volume / Dollar / Runs / Imbalance bars
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Feature Engineer  │  Price structure · Momentum · Volatility · Cross-bar density
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Triple-Barrier   │  Labels: +1 / -1 / 0  (profit target / stop loss / timeout)
│     Labeling      │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Random Forest    │  Weekly model (1-week horizon)
│     Models        │  Monthly model (1-month horizon)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Signal Engine    │  Real-time dollar bar accumulator + predict_proba
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│   Discord Bot     │  Signal alerts + hourly status
└───────────────────┘
```

## Tickers

| Ticker | Weekly Model | Monthly Model |
|--------|-------------|---------------|
| SPY    | thr=0.63    | thr=0.65      |
| SNDK   | thr=0.63    | —             |
| CRWV   | thr=0.60    | thr=0.63      |

## Signal Message Format

```
📈 CRWV  LONG  [1 周]

入场价   $185.40
目标价   $189.52  (+2.22%)
止损价   $181.28  (-2.22%)

预计涨幅  2.22%  (波动率 2.22%/bar)
持仓周期  1 周  · 建议 05/06 前平仓
置信度    63.4%
```

## Setup

### 1. Clone & install dependencies

```bash
git clone <repo>
cd Kitty
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```
DISCORD_TOKEN=your_discord_bot_token
SCHEDULE_CHANNEL_ID=your_channel_id
POLYGON_API_KEY=your_polygon_api_key
```

**Discord setup:**
- Create a bot at [discord.com/developers/applications](https://discord.com/developers/applications)
- Enable **Message Content Intent** under Bot settings
- Invite the bot with `bot` scope + `Send Messages` / `Read Messages` permissions

**Polygon.io:** Free account at [polygon.io](https://polygon.io) — provides 2 years of 1-minute historical data.

### 3. Build data & train models

Run the full pipeline for all tickers (fetches data, builds bars, trains models):

```bash
bash start.sh pipeline
```

Or for a single ticker:

```bash
python3 scripts/run_pipeline.py --tickers SPY
python3 scripts/run_pipeline.py --tickers SNDK CRWV
```

### 4. Start the bot

```bash
bash start.sh bot
```

## Project Structure

```
Kitty/
├── bot.py                      # Discord bot entry point
├── requirements.txt
├── .env.example
├── start.sh                    # Startup script
│
├── data/
│   ├── collector.py            # Polygon.io data fetching + caching
│   ├── features.py             # Feature engineering
│   ├── labels.py               # Triple-barrier labeling
│   └── bars/
│       ├── volume.py           # Volume bars
│       ├── dollar.py           # Dollar bars
│       ├── runs.py             # Dollar runs bars
│       └── imbalance.py        # Dollar imbalance bars
│
├── models/
│   ├── cv.py                   # Purged K-Fold cross-validation
│   └── saved/                  # Trained model .pkl files
│
├── scripts/
│   ├── run_pipeline.py         # End-to-end: fetch → bars → train
│   ├── build_bars.py           # Bar construction only
│   ├── build_and_train.py      # Feature engineering + model training
│   └── train_model.py          # Legacy single-model training script
│
├── signals/
│   ├── engine.py               # Real-time bar accumulator + signal generation
│   └── formatter.py            # Discord message formatting
│
└── data/
    ├── raw/                    # Cached Polygon.io minute data (.parquet)
    ├── bars/processed/         # Built bars (.parquet)
    └── dataset/                # Labeled ML datasets (.parquet)
```

## Bot Commands

| Command    | Description                                      |
|------------|--------------------------------------------------|
| `!ping`    | Check bot latency                                |
| `!status`  | Show last signal time per ticker                 |
| `!signal`  | Force-poll all tickers now (for testing)         |

## Bar Types

| Bar | Closes when... | Captures |
|-----|---------------|---------|
| **Volume** | Cumulative volume ≥ threshold | Even information sampling |
| **Dollar** | Cumulative dollar value ≥ threshold | More stable across price levels |
| **Runs** | One-sided dollar flow ≥ threshold | Directional momentum |
| **Imbalance** | Net signed dollar flow ≥ threshold | Order flow imbalance |

## Model Design

- **Labels:** Triple-barrier — price hits profit target (+1), stop loss (-1), or times out (0)
- **Barriers:** `h × rolling_vol` — adaptive to market volatility
- **Features:** 28 features — bar structure, momentum (50/100/400 bar windows), volatility regime, cross-bar density
- **Validation:** Purged K-Fold (no train/test leakage from overlapping label windows)
- **Signal filter:** Only trade when `predict_proba > threshold` (~8–20% of bars)

## Notes

- Polygon.io free tier provides data with ~15-minute delay. Signals are labeled accordingly.
- Monthly model for SNDK is omitted due to insufficient training data (~13 months of history).
- Models should be retrained periodically as new data accumulates (`bash start.sh pipeline`).
