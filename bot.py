import os
import sys
import logging
import datetime
from pathlib import Path

import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from signals.engine    import SignalEngine, SIGNAL_CONFIG
from signals.formatter import format_signal, format_status

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

CHANNEL_ID = int(os.getenv("SCHEDULE_CHANNEL_ID", "0"))

# ── state ─────────────────────────────────────────────────────────────────────
engine = None
last_signal_time: dict = {}   # {ticker: datetime}
last_status_sent = None

WINDOW_START = datetime.time(8, 30)
WINDOW_END   = datetime.time(23, 30)
STATUS_INTERVAL_HOURS = 1


# ── helpers ───────────────────────────────────────────────────────────────────

def _in_window() -> bool:
    now = datetime.datetime.now().time()
    return WINDOW_START <= now <= WINDOW_END


async def _send(content: str):
    channel = bot.get_channel(CHANNEL_ID)
    if channel:
        await channel.send(content)
    else:
        log.warning(f"Channel {CHANNEL_ID} not found")


# ── tasks ─────────────────────────────────────────────────────────────────────

@tasks.loop(minutes=1)
async def signal_loop():
    if not _in_window():
        return

    global last_status_sent

    # poll for new bars → signals
    try:
        signals = engine.poll()
    except Exception as e:
        log.error(f"Engine poll error: {e}")
        return

    for sig in signals:
        msg = format_signal(sig)
        await _send(msg)
        last_signal_time[sig.ticker] = datetime.datetime.now(datetime.timezone.utc)
        log.info(f"Signal sent: {sig.ticker} {sig.regime} {sig.direction} @ {sig.entry_price}")

    # hourly status if no signals recently
    now = datetime.datetime.now(datetime.timezone.utc)
    need_status = (
        last_status_sent is None
        or (now - last_status_sent).total_seconds() >= STATUS_INTERVAL_HOURS * 3600
    )
    any_recent = any(
        (now - t).total_seconds() < STATUS_INTERVAL_HOURS * 3600
        for t in last_signal_time.values()
    )
    if need_status and not any_recent:
        msg = format_status(last_signal_time, list(SIGNAL_CONFIG.keys()))
        await _send(msg)
        last_status_sent = now


# ── events ────────────────────────────────────────────────────────────────────

@bot.event
async def on_ready():
    global engine
    log.info(f"Logged in as {bot.user} (ID: {bot.user.id})")
    log.info(f"Connected to {len(bot.guilds)} guild(s)")

    log.info("Loading signal engine ...")
    engine = SignalEngine()
    log.info("Signal engine ready")

    if CHANNEL_ID:
        signal_loop.start()
        log.info(f"Signal loop started — polling every 1 min, window {WINDOW_START}–{WINDOW_END}")


@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return
    log.info(f"[#{message.channel}] {message.author}: {message.content}")
    await bot.process_commands(message)


# ── commands ──────────────────────────────────────────────────────────────────

@bot.command(name="ping")
async def ping(ctx: commands.Context):
    await ctx.send(f"Pong! Latency: {round(bot.latency * 1000)}ms")


@bot.command(name="status")
async def status(ctx: commands.Context):
    """Manually trigger a status update."""
    msg = format_status(last_signal_time, list(SIGNAL_CONFIG.keys()))
    await ctx.send(msg)


@bot.command(name="signal")
async def force_signal(ctx: commands.Context):
    """Force-poll all tickers right now (for testing)."""
    import requests as _req
    from signals.engine import API_KEY, BASE_URL, SIGNAL_CONFIG
    from datetime import timezone, timedelta

    now     = datetime.datetime.now(datetime.timezone.utc)
    from_ms = int((now - datetime.timedelta(minutes=15)).timestamp() * 1000)
    to_ms   = int(now.timestamp() * 1000)

    # show what data Polygon actually has right now
    lines = [f"**Polling** `{now.strftime('%H:%M:%S')} UTC`", ""]
    for ticker in SIGNAL_CONFIG:
        url  = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/minute/{from_ms}/{to_ms}"
        resp = _req.get(url, params={"sort":"asc","limit":5,"apiKey":API_KEY}, timeout=8)
        data = resp.json()
        bars = data.get("results", [])
        if bars:
            last_ts = datetime.datetime.fromtimestamp(
                bars[-1]["t"] / 1000, tz=datetime.timezone.utc
            ).strftime("%H:%M")
            lines.append(f"`{ticker}` — {len(bars)} bar(s), last @ {last_ts} UTC  "
                         f"close=${bars[-1]['c']:.2f}")
        else:
            lines.append(f"`{ticker}` — 市场已收盘 / 无最新数据")

    await ctx.send("\n".join(lines))

    # now actually poll for signals
    try:
        signals = engine.poll()
    except Exception as e:
        await ctx.send(f"Engine error: {e}")
        return

    if signals:
        for sig in signals:
            await ctx.send(format_signal(sig))
    else:
        await ctx.send("概率未达阈值，暂无信号。（市场收盘后不产生新 bar）")


@bot.command(name="analyze")
async def analyze(ctx: commands.Context, ticker: str = None):
    """
    Run a full model analysis using the latest bar from saved parquet data.
    Shows predict_proba for every model regardless of threshold.

    Usage:
      !analyze          — analyze all tickers
      !analyze SPY      — analyze one ticker
    """
    import pickle
    import pandas as pd
    import numpy as np
    from collections import deque
    from signals.engine import (
        SIGNAL_CONFIG, MODEL_DIR, BARS_DIR, HISTORY_LEN,
        _compute_features, _rolling_vol,
    )

    tickers = [ticker.upper()] if ticker else list(SIGNAL_CONFIG.keys())
    await ctx.send(f"分析中，请稍候...")

    for t in tickers:
        bar_path = BARS_DIR / f"{t}_dollar_bars.parquet"
        if not bar_path.exists():
            await ctx.send(f"`{t}` — 找不到 bar 数据，请先运行 pipeline")
            continue

        df      = pd.read_parquet(bar_path)
        history = deque(df.tail(HISTORY_LEN).to_dict("records"), maxlen=HISTORY_LEN)
        last    = df.iloc[-1]
        bar_ts  = pd.Timestamp(last["timestamp"])
        price   = last["close"]

        lines = [
            f"**── {t} 分析报告 ──**",
            f"最新 bar: `{bar_ts.strftime('%Y-%m-%d %H:%M')} UTC`  "
            f"收盘价 `${price:,.2f}`",
            "",
        ]

        for regime, thr, h, vol_lb, max_hold in SIGNAL_CONFIG.get(t, []):
            model_path = MODEL_DIR / f"{t}_rf_{regime}.pkl"
            if not model_path.exists():
                lines.append(f"`{regime}` — 模型文件不存在")
                continue

            with open(model_path, "rb") as f:
                obj = pickle.load(f)

            cfg       = obj["config"]
            model     = obj["model"]
            feat_cols = obj["features"]

            feats = _compute_features(history, feat_cols, cfg)
            if feats is None:
                lines.append(f"`{regime}` — 历史数据不足，无法计算特征")
                continue

            proba_long  = float(model.predict_proba(feats)[0, 1])
            proba_short = 1 - proba_long
            vol         = _rolling_vol(history, vol_lb)

            target = price * (1 + h * vol)
            stop   = price * (1 - h * vol)

            # direction indicator
            if proba_long > thr:
                verdict = "📈 **LONG 信号**"
                conf    = proba_long
            elif proba_short > thr:
                verdict = "📉 **SHORT 信号**"
                conf    = proba_short
            else:
                verdict = "⬜ 未达阈值"
                conf    = max(proba_long, proba_short)

            regime_cn = "1 周" if regime == "weekly" else "1 个月"
            days      = max_hold // 20

            lines += [
                f"**{regime_cn}模型**  (阈值 {thr})",
                f"  P(多) `{proba_long:.1%}`  P(空) `{proba_short:.1%}`  → {verdict}",
                f"  置信度 `{conf:.1%}`  |  波动率 `{vol*100:.2f}%/bar`",
                f"  目标价 `${target:,.2f}` (+{(target/price-1)*100:.2f}%)  "
                f"止损价 `${stop:,.2f}` ({(stop/price-1)*100:.2f}%)",
                f"  持仓周期 {regime_cn} (~{days} 交易日)",
                "",
            ]

        await ctx.send("\n".join(lines))


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_TOKEN not set")
    bot.run(token)
