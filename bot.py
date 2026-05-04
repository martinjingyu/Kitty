import asyncio
import os
import sys
import logging
import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from signals.engine    import SignalEngine, SIGNAL_CONFIG
from signals.formatter import (format_signal, format_status, format_scan,
                                format_open_confirm, format_close_confirm,
                                format_sl_alarm, format_tp_alarm)

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

# Manual position journal: {f"{ticker}_{regime}": dict}
open_positions: dict = {}

_ET = ZoneInfo("America/New_York")
MARKET_OPEN    = datetime.time(4, 0)   # 4:00 AM ET — pre-market start
MARKET_CLOSE   = datetime.time(20, 0)  # 8:00 PM ET — after-hours end
REGULAR_CLOSE  = datetime.time(16, 0)  # 4:00 PM ET — regular session close
STATUS_INTERVAL_HOURS = 1


# ── helpers ───────────────────────────────────────────────────────────────────

def _in_window() -> bool:
    now_et = datetime.datetime.now(_ET)
    if now_et.weekday() >= 5:
        return False
    return MARKET_OPEN <= now_et.time() <= MARKET_CLOSE

def _in_regular_session() -> bool:
    """True only during regular trading hours (9:30 AM – 4:00 PM ET, weekdays)."""
    now_et = datetime.datetime.now(_ET)
    if now_et.weekday() >= 5:
        return False
    return datetime.time(9, 30) <= now_et.time() <= REGULAR_CLOSE


def _check_alarms() -> list[str]:
    """
    Sync function (run in executor).
    Returns list of formatted alarm messages for open positions.

    SL: P(against_direction) > threshold for that regime
    TP: latest bar price crossed the target level
    """
    if not engine or not open_positions:
        return []

    msgs = []
    for key, pos in list(open_positions.items()):
        ticker    = pos["ticker"]
        regime    = pos["regime"]
        direction = pos["direction"]

        proba = engine.get_proba(ticker, regime)
        if proba is None:
            continue
        proba_long, proba_short, _ = proba

        # Threshold for this regime
        regime_entry = next((e for e in SIGNAL_CONFIG.get(ticker, []) if e[0] == regime), None)
        thr = regime_entry[1] if regime_entry else 0.50  # thr_long as proxy

        proba_against = proba_short if direction == "LONG" else proba_long

        # Update last seen price for SL alarm P&L display
        hist = engine.histories.get(ticker)
        if hist:
            open_positions[key]["last_price"] = hist[-1]["close"]

        # ── SL: model confidence against position exceeds threshold ──────────
        if proba_against > thr:
            if not pos.get("sl_alerted"):
                msgs.append(format_sl_alarm(pos, proba_against, proba_long, proba_short))
                open_positions[key]["sl_alerted"] = True
        else:
            if pos.get("sl_alerted"):
                open_positions[key]["sl_alerted"] = False  # model normalised — reset

        # ── TP: price crossed target ──────────────────────────────────────────
        if pos.get("tp_alerted") or pos.get("target") is None or not hist:
            continue
        last_bar = hist[-1]
        hit = (direction == "LONG"  and last_bar["high"] >= pos["target"]) or \
              (direction == "SHORT" and last_bar["low"]  <= pos["target"])
        if hit:
            msgs.append(format_tp_alarm(pos, last_bar["close"], proba_long, proba_short))
            open_positions[key]["tp_alerted"] = True

    return msgs


async def _send(content: str):
    channel = bot.get_channel(CHANNEL_ID)
    if channel:
        await channel.send(content)
    else:
        log.warning(f"Channel {CHANNEL_ID} not found")


# ── tasks ─────────────────────────────────────────────────────────────────────

@tasks.loop(minutes=5)
async def signal_loop():
    if not _in_regular_session():
        return

    global last_status_sent

    # run blocking CPU work in a thread so the event loop stays free
    try:
        loop    = asyncio.get_event_loop()
        signals = await loop.run_in_executor(None, engine.poll)
    except Exception as e:
        log.error(f"Engine poll error: {e}")
        return

    for sig in signals:
        msg = format_signal(sig)
        await _send(msg)
        last_signal_time[sig.ticker] = datetime.datetime.now(datetime.timezone.utc)
        log.info(f"Signal: {sig.ticker} {sig.regime} {sig.direction} @ {sig.entry_price}")

    # Check TP/SL alarms for open positions
    if open_positions:
        alarms = await loop.run_in_executor(None, _check_alarms)
        for alarm_msg in alarms:
            await _send(alarm_msg)

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


@tasks.loop(minutes=30)
async def scan_loop():
    if not _in_regular_session():
        return
    try:
        loop  = asyncio.get_event_loop()
        scans = await loop.run_in_executor(None, engine.scan)
    except Exception as e:
        log.error(f"Scan error: {e}")
        return
    if scans:
        await _send(format_scan(scans))


# ── events ────────────────────────────────────────────────────────────────────

@bot.event
async def on_ready():
    global engine
    log.info(f"Logged in as {bot.user} (ID: {bot.user.id})")
    log.info(f"Connected to {len(bot.guilds)} guild(s)")

    log.info("Loading signal engine ...")
    engine = SignalEngine()
    log.info("Signal engine ready — running warmup ...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, engine.warmup)
    log.info("Warmup complete")

    if CHANNEL_ID:
        signal_loop.start()
        scan_loop.start()
        log.info(f"Signal loop started — polling every 1 min, window {MARKET_OPEN}–{MARKET_CLOSE} ET (weekdays only)")
        log.info("Scan loop started — market snapshot every 30 min")


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


@bot.command(name="debug")
async def debug(ctx: commands.Context):
    """Show engine internals: accumulator state, last_ts, history tail."""
    if not engine:
        await ctx.send("Engine not ready")
        return

    now = datetime.datetime.now(datetime.timezone.utc)
    lines = [f"**── Engine Debug  {now.strftime('%H:%M:%S')} UTC ──**", ""]

    for ticker in list(engine.tickers)[:3]:   # first 3 tickers to avoid message limit
        acc      = engine.accumulators[ticker]
        last_ts  = engine.last_ts.get(ticker)
        hist_len = len(engine.histories[ticker])
        hist_last = engine.histories[ticker][-1] if engine.histories[ticker] else None

        ts_str  = (datetime.datetime.fromtimestamp(last_ts / 1000, tz=datetime.timezone.utc)
                   .strftime("%H:%M:%S") if last_ts else "None")
        bar_str = (datetime.datetime.utcfromtimestamp(
                       hist_last["timestamp"].timestamp()
                       if hasattr(hist_last["timestamp"], "timestamp")
                       else hist_last["timestamp"] / 1000
                   ).strftime("%m/%d %H:%M") if hist_last else "—")

        lines += [
            f"**{ticker}**",
            f"  last_ts fetched : `{ts_str} UTC`",
            f"  acc cum_dollar  : `${acc._cum_dollar:,.0f}`",
            f"  acc threshold   : `${acc.threshold:,.0f}`",
            f"  acc bucket bars : `{len(acc._bucket)}`",
            f"  history bars    : `{hist_len}`  (latest {bar_str} UTC)",
            "",
        ]

    await ctx.send("\n".join(lines))


@bot.command(name="status")
async def status(ctx: commands.Context):
    """Manually trigger a status update."""
    msg = format_status(last_signal_time, list(SIGNAL_CONFIG.keys()))
    await ctx.send(msg)


@bot.command(name="open")
async def cmd_open(ctx: commands.Context, ticker: str, regime: str,
                   direction: str, price: float, time_str: str = None):
    """
    记录手动开仓。
    用法: !open <TICKER> <regime> <LONG|SHORT> <价格> [HH:MM]
    示例: !open SPY intraday LONG 580.50
          !open NVDA weekly SHORT 118.00 14:30
    """
    ticker    = ticker.upper()
    direction = direction.upper()
    if direction not in ("LONG", "SHORT"):
        await ctx.send("方向必须是 `LONG` 或 `SHORT`")
        return

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    if time_str:
        try:
            h, m = map(int, time_str.split(":"))
            now_utc = now_utc.replace(hour=h, minute=m, second=0, microsecond=0)
        except ValueError:
            await ctx.send("时间格式错误，请用 `HH:MM`，例如 `14:30`")
            return

    key = f"{ticker}_{regime}"
    if key in open_positions:
        existing = open_positions[key]
        await ctx.send(
            f"⚠️ `{ticker} {regime}` 已有未平仓记录  "
            f"({existing['direction']} @ ${existing['price']:,.2f})，"
            f"请先 `!close` 平仓"
        )
        return

    # Auto-calculate TP target from model config + current vol
    target = None
    try:
        from signals.engine import _rolling_vol
        regime_entry = next((e for e in SIGNAL_CONFIG.get(ticker, []) if e[0] == regime), None)
        if regime_entry and engine and engine.histories.get(ticker):
            _, _, _, h, vol_lb, _ = regime_entry
            vol      = _rolling_vol(engine.histories[ticker], vol_lb)
            cfg      = engine.models.get(ticker, {}).get(regime, {})
            h_target = cfg.get("h_target", h)
            target   = price * (1 + h_target * vol) if direction == "LONG" else price * (1 - h_target * vol)
            target   = round(target, 2)
    except Exception:
        pass

    open_positions[key] = {
        "ticker":        ticker,
        "regime":        regime,
        "direction":     direction,
        "entry_price":   price,
        "time":          now_utc,
        "target":        target,
        "remaining_pct": 1.0,
        "last_price":    price,
        "tp_alerted":    False,
        "sl_alerted":    False,
    }
    await ctx.send(format_open_confirm(ticker, regime, direction, price, now_utc, target))
    log.info(f"Manual open: {key} {direction} @ {price}  target={target}")


@bot.command(name="close")
async def cmd_close(ctx: commands.Context, ticker: str, regime: str,
                    price: float, pct_or_time: str = None, time_str: str = None):
    """
    记录手动平仓（支持部分平仓），计算盈亏。
    用法: !close <TICKER> <regime> <出场价> [50%] [HH:MM]
    示例: !close SPY intraday 583.20
          !close SPY intraday 583.20 50%
          !close NVDA weekly 115.50 50% 16:00
    """
    ticker = ticker.upper()
    key    = f"{ticker}_{regime}"
    pos    = open_positions.get(key)
    if pos is None:
        await ctx.send(f"⚠️ 找不到 `{ticker} {regime}` 的开仓记录")
        return

    # Parse optional pct_or_time: could be "50%" or "16:00"
    close_frac = 1.0
    if pct_or_time:
        if pct_or_time.endswith("%"):
            try:
                close_frac = float(pct_or_time.rstrip("%")) / 100
            except ValueError:
                await ctx.send("百分比格式错误，例如 `50%`")
                return
        else:
            time_str = pct_or_time  # it was actually a time string

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    if time_str:
        try:
            h, m = map(int, time_str.split(":"))
            now_utc = now_utc.replace(hour=h, minute=m, second=0, microsecond=0)
        except ValueError:
            await ctx.send("时间格式错误，请用 `HH:MM`，例如 `16:00`")
            return

    remaining_before = pos["remaining_pct"]
    close_pct        = min(close_frac, remaining_before)
    remaining_after  = round(remaining_before - close_pct, 4)

    msg = format_close_confirm(
        ticker, regime, pos["direction"],
        pos["entry_price"], price,
        pos["time"], now_utc,
        close_pct=close_pct, remaining_pct=remaining_after,
    )

    if remaining_after <= 0:
        del open_positions[key]
        log.info(f"Full close: {key} {pos['direction']} {pos['entry_price']} → {price}")
    else:
        open_positions[key]["remaining_pct"] = remaining_after
        open_positions[key]["tp_alerted"]    = False  # reset so TP can re-trigger on remainder
        log.info(f"Partial close: {key} {close_pct:.0%} closed, {remaining_after:.0%} remaining")

    await ctx.send(msg)


@bot.command(name="positions")
async def cmd_positions(ctx: commands.Context):
    """显示当前所有未平仓记录。"""
    if not open_positions:
        await ctx.send("当前没有未平仓记录")
        return

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    lines   = ["**── 持仓记录 ──**", ""]
    for pos in open_positions.values():
        regime_cn = {"intraday": "日内", "weekly": "1 周", "monthly": "1 个月"}.get(pos["regime"], pos["regime"])
        held_min  = int((now_utc - pos["time"]).total_seconds() / 60)
        held_str  = f"{held_min // 60}h {held_min % 60:02d}m" if held_min >= 60 else f"{held_min}m"
        icon      = "📈" if pos["direction"] == "LONG" else "📉"
        lines.append(
            f"{icon} **{pos['ticker']}** [{regime_cn}]  "
            f"`{pos['direction']}` @ `${pos['entry_price']:,.2f}`  · 已持仓 {held_str}"
        )
    await ctx.send("\n".join(lines))


@bot.command(name="signal")
async def force_signal(ctx: commands.Context):
    """Force-poll all tickers right now (for testing)."""
    import yfinance as yf
    from signals.engine import SIGNAL_CONFIG

    now  = datetime.datetime.now(datetime.timezone.utc)
    diag = [f"**Polling** `{now.strftime('%H:%M:%S')} UTC`", ""]

    for ticker in SIGNAL_CONFIG:
        try:
            df = yf.Ticker(ticker).history(period="1d", interval="1m")
            if df.empty:
                diag.append(f"`{ticker}` — 无数据")
            else:
                last = df.iloc[-1]
                ts   = df.index[-1].tz_convert("UTC").strftime("%H:%M")
                diag.append(f"`{ticker}` — last @ {ts} UTC  close=`${last['Close']:.2f}`")
        except Exception as e:
            diag.append(f"`{ticker}` — 抓取失败: {e}")

    await ctx.send("\n".join(diag))

    # now actually poll for signals (in executor — XGBoost is CPU-blocking)
    loop = asyncio.get_event_loop()
    try:
        signals = await loop.run_in_executor(None, engine.poll)
    except Exception as e:
        await ctx.send(f"Engine error: {e}")
        return

    if signals:
        for sig in signals:
            await ctx.send(format_signal(sig))
    else:
        scans = await loop.run_in_executor(None, engine.scan)
        if scans:
            await ctx.send(format_scan(scans))
        else:
            await ctx.send("暂无预测数据（历史 bar 不足）")


@bot.command(name="analyze")
async def analyze(ctx: commands.Context, ticker: str = None):
    """
    Run a full model analysis using the latest bar from the live engine history.
    Shows predict_proba for every model regardless of threshold.

    Usage:
      !analyze          — analyze all tickers
      !analyze SPY      — analyze one ticker
    """
    import pickle
    import pandas as pd
    import numpy as np
    from signals.engine import (
        SIGNAL_CONFIG, MODEL_DIR, BARS_DIR,
        _compute_features, _rolling_vol,
    )

    tickers = [ticker.upper()] if ticker else list(SIGNAL_CONFIG.keys())
    await ctx.send("分析中，请稍候...")

    def _analyze_ticker(t: str) -> str:
        import pandas as pd
        from collections import deque
        from signals.engine import HISTORY_LEN

        # Prefer live engine history (updated by poll); fall back to parquet if engine not ready
        history = engine.histories.get(t) if engine else None
        if not history:
            bar_path = BARS_DIR / f"{t}_dollar_bars.parquet"
            if not bar_path.exists():
                return f"`{t}` — 找不到 bar 数据，请先运行 pipeline"
            df      = pd.read_parquet(bar_path)
            history = deque(df.tail(HISTORY_LEN).to_dict("records"), maxlen=HISTORY_LEN)

        last   = history[-1]
        bar_ts = pd.Timestamp(last["timestamp"])
        price  = last["close"]

        lines = [
            f"**── {t} 分析报告 ──**",
            f"最新 bar: `{bar_ts.strftime('%Y-%m-%d %H:%M')} UTC`  收盘价 `${price:,.2f}`",
            "",
        ]

        for regime, thr_long, thr_short, h, vol_lb, max_hold in SIGNAL_CONFIG.get(t, []):
            model_path = MODEL_DIR / f"multi_xgb_{regime}.pkl"
            if not model_path.exists():
                model_path = MODEL_DIR / f"{t}_xgb_{regime}.pkl"
            if not model_path.exists():
                model_path = MODEL_DIR / f"{t}_rf_{regime}.pkl"
            if not model_path.exists():
                lines.append(f"`{regime}` — 模型文件不存在")
                continue

            with open(model_path, "rb") as f:
                obj = pickle.load(f)

            cfg          = obj["config"]
            model        = obj["model"]
            feat_cols    = obj["features"]
            zscore_stats = obj.get("zscore_stats", {}).get(t, {})

            feats = _compute_features(history, feat_cols, cfg, zscore_stats)
            if feats is None:
                lines.append(f"`{regime}` — 历史数据不足，无法计算特征")
                continue

            proba_arr    = model.predict_proba(feats)[0]
            proba_long   = float(proba_arr[-1])
            proba_short  = float(proba_arr[0])
            proba_condor = float(proba_arr[1]) if len(proba_arr) > 2 else 0.0
            vol          = _rolling_vol(history, vol_lb)

            h_target = cfg.get("h_target", h)
            h_stop   = cfg.get("h_stop",   h)
            long_target  = price * (1 + h_target * vol)
            long_stop    = price * (1 - h_stop   * vol)
            short_target = price * (1 - h_target * vol)
            short_stop   = price * (1 + h_stop   * vol)

            if proba_long > thr_long:
                verdict = "📈 **LONG 信号**"
                conf    = proba_long
                t_price, s_price = long_target, long_stop
            elif proba_short > thr_short:
                verdict = "📉 **SHORT 信号**"
                conf    = proba_short
                t_price, s_price = short_target, short_stop
            elif proba_condor > max(proba_long, proba_short):
                verdict = "🦅 **CONDOR 候选**"
                conf    = proba_condor
                t_price, s_price = long_target, long_stop
            else:
                verdict = "⬜ 未达阈值"
                conf    = max(proba_long, proba_short, proba_condor)
                t_price, s_price = long_target, long_stop

            regime_cn  = {"intraday": "日内", "weekly": "1 周", "monthly": "1 个月"}.get(regime, regime)
            days       = max_hold // 20
            proba_line = f"P(多) `{proba_long:.1%}`  P(空) `{proba_short:.1%}`"
            if proba_condor > 0:
                proba_line += f"  P(condor) `{proba_condor:.1%}`"

            lines += [
                f"**{regime_cn}模型**  (多阈值 {thr_long}  空阈值 {thr_short})",
                f"  {proba_line}  → {verdict}",
                f"  置信度 `{conf:.1%}`  |  波动率 `{vol*100:.2f}%/bar`",
                f"  目标价 `${t_price:,.2f}` ({(t_price/price-1)*100:+.2f}%)  "
                f"止损价 `${s_price:,.2f}` ({(s_price/price-1)*100:+.2f}%)",
                f"  持仓周期 {regime_cn} (~{days} 交易日)",
                "",
            ]

        return "\n".join(lines)

    loop = asyncio.get_event_loop()
    for t in tickers:
        msg = await loop.run_in_executor(None, _analyze_ticker, t)
        await ctx.send(msg)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_TOKEN not set")
    bot.run(token)
