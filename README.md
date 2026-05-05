# Kitty — Quantitative Trading Signal Bot

A personal quantitative trading agent that builds alternative financial bars (Volume / Dollar / Runs / Imbalance), trains directional signal models across three holding-period regimes, and delivers real-time trade alerts and position monitoring to Discord.

## Architecture

```
Polygon.io (1-min bars)
        │
        ▼
┌───────────────────┐
│  Bar Construction │  Volume / Dollar / Runs / Imbalance bars
│ (rolling thresh.) │  Threshold adapts to trailing 20-day volume
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Feature Engineer  │  Bar structure · Momentum · Volatility · Cross-bar density
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────────────┐
│  Regime-Specific Labeling             │
│  Intraday  exit-on-first-touch        │  binary  LONG / SHORT
│  Weekly    asymmetric momentum filter │  binary  LONG / SHORT
│  Monthly   price-area integral        │  3-class LONG / SHORT / CONDOR
└────────┬──────────────────────────────┘
         │
         ▼
┌───────────────────┐
│  XGBoost Models   │  One pooled model per regime (all tickers combined)
│  (multi-ticker)   │  Class-balanced sample weights + Purged K-Fold CV
└────────┬──────────┘
         │
         ▼
┌────────────────────────────┐
│  Signal Engine             │  Real-time dollar bar accumulator
│                            │  Direction-change dedup (no repeat signals)
│                            │  Per-position TP / SL alarm checks (每分钟)
└────────┬───────────────────┘
         │
         ▼
┌───────────────────┐
│   Discord Bot     │  信号推送 · 手动开平仓记录 · TP/SL 预警
└───────────────────┘
```

## Tickers & Models

All six tickers share a single pooled model per regime:

| Regime | Tickers | CV Accuracy | Threshold | Precision | Mean ret/trade |
|--------|---------|-------------|-----------|-----------|----------------|
| Intraday | SPY, AAPL, CRWV, NVDA, SNDK, TSLA | 51.8% ± 1.4% | 0.65 | 71.8% | 0.061%/trade |
| Weekly   | SPY, AAPL, CRWV, NVDA, SNDK, TSLA | 54.9% ± 1.6% | 0.50 | 60.4% | 0.052%/trade |
| Monthly  | SPY, AAPL, CRWV, NVDA, SNDK, TSLA | 40.9% ± 7.0% | 0.54 | 52.5% | 4.97%/trade  |

## Discord Message Formats

### Entry signal
```
📈 CRWV  LONG  [日内]
   日内方向

入场价   $185.40
目标价   $187.51  (+1.14%)
止损价   $183.29  (-1.14%)
盈亏比   1.09x  (波动率 2.27%/bar)

持仓周期  日内  · 建议 05/07 前平仓
置信度    65.2%
```

### TP alarm (price + model annotation)
```
🎯 CRWV  止盈预警  [日内]

目标价     $187.51  已触达
当前价     $187.63
持仓盈亏   +1.20%  (持仓 100%)

📈 模型仍看多 (68.3%)，可考虑部分止盈继续持仓

!close CRWV intraday <价格> [50%] [HH:MM]
```

### SL alarm (model-driven)
```
⚠️ CRWV  止损预警  [日内]
   模型空信号置信度偏高，建议减仓或止损

持仓方向   LONG  入场价 $185.40
当前盈亏   -0.43%
模型概率   多 28.1%  空 71.9%
反方向置信  71.9%
```

### Open / close confirm
```
📈 CRWV  开仓记录  [日内]

方向       LONG
入场价     $185.40
止盈目标   $187.51
时间       2026-05-02 14:30 UTC
```
```
✅ CRWV  平仓记录  [日内]

方向       LONG
入场价     $185.40  @ 05/02 14:30 UTC
出场价     $187.51  @ 05/02 15:45 UTC
盈亏       +1.14%  ·  已平 50%，剩余 50%
持仓时长   1h 15m
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
- Invite with `bot` scope + `Send Messages` / `Read Messages` permissions

**Polygon.io:** Free account at [polygon.io](https://polygon.io) — provides up to 2 years of 1-minute historical data.

### 3. Build data & train models

```bash
bash start.sh pipeline
# or
python3 scripts/run_pipeline.py
```

After updating bar-construction logic, use `--force` to rebuild from raw data:

```bash
python3 scripts/run_pipeline.py --force
```

Train only selected regimes:

```bash
python3 scripts/run_pipeline.py --regimes intraday
python3 scripts/run_pipeline.py --tickers SPY AAPL NVDA --regimes intraday
python3 scripts/build_and_train.py --tickers SPY AAPL NVDA --regimes intraday weekly
```

### 4. Tune intraday boundaries

Before retraining intraday models, sweep asymmetric upside / downside
barriers per ticker:

```bash
python3 scripts/sweep_intraday_boundaries.py
```

Useful custom run:

```bash
python3 scripts/sweep_intraday_boundaries.py \
  --tickers SPY AAPL NVDA TSLA \
  --h-up 0.6,0.8,1.0,1.2 \
  --h-down 0.6,0.8,1.0,1.2 \
  --min-up 0.01,0.0125,0.015,0.02 \
  --min-down 0.008,0.01,0.0125,0.015 \
  --min-balance 0.55 \
  --max-short-share 0.65 \
  --top-n 5
```

Outputs:

```text
models/saved/intraday_boundary_sweep.csv
models/saved/intraday_boundary_sweep_top.csv
```

The sweep does **not** train a model. It labels historical dollar bars with:

```python
up_dist = max(h_up * vol, min_up)
down_dist = max(h_down * vol, min_down)
```

Then ranks settings by directional coverage, LONG/SHORT balance, sample count,
and target/stop ratio. Use this to pick ticker-specific intraday boundary
settings before editing the training config.

Training reads the top sweep output automatically when the file exists:

```text
models/saved/intraday_boundary_sweep_top.csv
```

To train with a different sweep result:

```bash
python3 scripts/build_and_train.py \
  --intraday-boundaries models/saved/my_intraday_boundary_top.csv
```

To ignore per-ticker intraday boundaries and use the symmetric default:

```bash
python3 scripts/build_and_train.py --intraday-boundaries none
```

### 5. Start the bot

```bash
bash start.sh bot
```

The bot only polls during **US market hours (Mon–Fri, 4:00 AM – 8:00 PM ET)**. It is idle on weekends and outside those hours.

## Bot Commands

### Monitoring

| Command | Description |
|---------|-------------|
| `!ping` | Check bot latency |
| `!status` | Show last signal time per ticker |
| `!signal` | Force-poll now; shows current predict_proba if no threshold crossed |
| `!analyze [TICKER]` | Full model breakdown — all regimes, probabilities, TP/SL levels |

### Position journal

| Command | Example | Description |
|---------|---------|-------------|
| `!open TICKER REGIME DIR PRICE [HH:MM]` | `!open SPY intraday LONG 580.50` | Record opening a position; auto-calculates TP target |
| `!close TICKER REGIME PRICE [PCT%] [HH:MM]` | `!close SPY intraday 583.20 50%` | Record close (full or partial); computes P&L |
| `!positions` | — | List all open positions with current hold time |

**Partial close example:**
```
!close SPY intraday 583.20 50%   → closes half, remainder stays monitored
!close SPY intraday 585.00       → closes remaining half
```

## TP / SL Alarm Logic

Every minute, for each open position:

**Stop-loss (model-driven)**
- Calls `model.predict_proba()` for the position's ticker/regime
- If `P(against_direction) > threshold` → sends ⚠️ SL alarm with current P&L + model probabilities
- Resets automatically if the model normalises; will re-alarm if it flips again

**Take-profit (price-driven + model annotation)**
- Compares latest bar's high/low against the auto-calculated target (`h_target × vol × entry`)
- When target is hit → sends 🎯 TP alarm
- Simultaneously queries current model signal:
  - Still bullish/bearish → "可考虑部分止盈继续持仓"
  - Signal neutral → "建议止盈"
- After a partial close, `tp_alerted` resets so the remaining position can trigger again

## Project Structure

```
Kitty/
├── bot.py                      # Discord bot — signal loop, position journal commands
├── requirements.txt
├── .env.example
├── start.sh
│
├── data/
│   ├── collector.py            # Polygon.io data fetching + caching
│   ├── features.py             # Feature engineering (28 features per bar)
│   ├── labels.py               # Regime-specific labeling functions
│   │                           #   label_intraday()        — exit-on-first-touch
│   │                           #   label_weekly_reversal() — asymmetric barriers
│   │                           #   label_monthly_area()    — price-area integral
│   └── bars/
│       ├── utils.py            # Tick rule + bar aggregation
│       ├── volume.py           # Volume bars    (rolling threshold)
│       ├── dollar.py           # Dollar bars    (rolling threshold)
│       ├── runs.py             # Dollar runs bars (rolling threshold)
│       └── imbalance.py        # Dollar imbalance bars (rolling threshold)
│
├── models/
│   ├── cv.py                   # Purged K-Fold cross-validation
│   └── saved/                  # Trained model .pkl files
│       ├── multi_xgb_intraday.pkl
│       ├── multi_xgb_weekly.pkl
│       └── multi_xgb_monthly.pkl
│
├── scripts/
│   ├── run_pipeline.py         # End-to-end: fetch → bars → train  [--force]
│   ├── build_bars.py           # Bar construction only              [--force]
│   ├── build_and_train.py      # Feature engineering + model training
│   └── sweep_intraday_boundaries.py
│                               # Per-ticker intraday boundary sweep
│
├── signals/
│   ├── engine.py               # Dollar bar accumulator · signal generation
│   │                           #   Direction-change dedup · get_proba() for alarms
│   └── formatter.py            # Discord message formatting (signals, alarms, journal)
│
└── data/
    ├── raw/                    # Cached Polygon.io 1-min data (.parquet)
    ├── bars/processed/         # Built bars (.parquet)
    └── dataset/                # Labeled ML datasets (.parquet)
```

## Model Design

### Labels

**Intraday** — exit-on-first-touch triple-barrier (binary):

| Label | Trigger |
|-------|---------|
| **+1 LONG**  | Price hits upper barrier first |
| **−1 SHORT** | Price hits lower barrier first |

Current training uses a symmetric barrier:

```python
barrier = max(h * vol, min_target_return)
upper = entry * (1 + barrier)
lower = entry * (1 - barrier)
```

If `models/saved/intraday_boundary_sweep_top.csv` exists, training reads the
first row per ticker and switches intraday labels to asymmetric barriers:

```python
up_dist = max(h_up * vol, min_up)
down_dist = max(h_down * vol, min_down)
upper = entry * (1 + up_dist)
lower = entry * (1 - down_dist)
```

**Weekly** — asymmetric momentum-filtered barriers (binary):

Samples only reversal candidates filtered by prior vol-adjusted momentum `mom_std = log_ret / (vol × √window)`:
- `mom_std < −1.0` → dip buy: wide profit barrier / tight stop
- `mom_std > +1.0` → top sell: tight stop / wide profit barrier
- Intermediate zone → skipped

**Monthly** — price-area integral (3-class):

```
area      = Σ (close[t+i] − close[t]) / close[t]   for i in [1, max_hold]
area_norm = area / (vol × max_hold)
```

| Label | Threshold | Meaning |
|-------|-----------|---------|
| **+1 LONG**   | area_norm ≥ +5.0 | Price consistently above entry |
| **−1 SHORT**  | area_norm ≤ −5.0 | Price consistently below entry |
| **0 CONDOR**  | \|area_norm\| ≤ 2.0 | Price oscillated near entry — iron condor candidate |
| skip | intermediate zone | Ambiguous, not sampled |

### Regimes

| Regime | max_hold | Labeling | Feature windows |
|--------|----------|----------|-----------------|
| **intraday** | 20 bars (~1 day)    | exit-on-first-touch, binary  | [20, 50, 100, 400] |
| **weekly**   | 100 bars (~1 week)  | asymmetric barrier, binary   | [50, 100, 400]     |
| **monthly**  | 400 bars (~1 month) | price-area integral, 3-class | [100, 200, 400]    |

### Training

- **Model:** XGBoost (`multi:softprob` for 3-class, `binary:logistic` for binary)
- **Pooled:** all tickers trained together; absolute-scale features z-scored per ticker
- **Split:** first 70% train, last 30% test (time-ordered, no shuffle)
- **Validation:** Purged K-Fold (5 splits, embargo = `max_hold` bars) to prevent label-window leakage
- **Sample weights:** `(1 / avg_concurrent_events) × (1 / class_freq)` — removes overlap bias and balances minority classes

### Signal Generation

- A signal fires when `predict_proba(LONG or SHORT) > threshold` **and** the direction has changed from the previous signal for that ticker/regime
- CONDOR predictions suppress directional signals but appear in `!signal` / `!analyze` scans as 🦅
- All CPU-bound inference runs in a thread-pool executor to keep the Discord event loop free

## Notes

- Polygon.io free tier has ~15-minute data delay.
- Monthly CONDOR signals are shown in scans but not forwarded as position alerts — they require a different position structure (iron condor).
- Retrain periodically: `python3 scripts/run_pipeline.py --tickers AAPL MSFT NVDA GOOGL AMZN META TSLA WMT MS JPM BE PLTR SPY IBM IWM MU SNDK CRWV NBIS INTC AMD ORCL COIN MSTR --force`
