import asyncio
import os
import sys
import logging
import datetime
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from signals.engine    import SignalEngine, SIGNAL_CONFIG, TARGET_RETURN_FLOOR
from signals.formatter import (format_signal, format_status, format_predictions,
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
active_signal_recommendations: dict = {}  # {ticker: dict}
daily_signal_stats: dict = {}             # {ticker: {regime: {direction: {count, conf_sum}}}}
daily_signal_stats_date = None
daily_summary_sent_date = None
daily_hf_upload_sent_date = None

_ET = ZoneInfo("America/New_York")
_CT = ZoneInfo("America/Chicago")
MARKET_OPEN    = datetime.time(4, 0)   # 4:00 AM ET — pre-market start
MARKET_CLOSE   = datetime.time(20, 0)  # 8:00 PM ET — after-hours end
REGULAR_CLOSE  = datetime.time(16, 0)  # 4:00 PM ET — regular session close
STATUS_INTERVAL_HOURS = 1
MAX_SIGNALS_PER_POLL = int(os.getenv("MAX_SIGNALS_PER_POLL", "3"))
MAX_SCAN_ROWS = int(os.getenv("MAX_SCAN_ROWS", "5"))
SUMMARY_REGIMES = {"weekly", "monthly"}
SUMMARY_SEND_AFTER_CT = datetime.time(15, 10)  # 10 min after regular close in CT
AUTO_HF_UPLOAD_ENABLED = os.getenv("AUTO_HF_UPLOAD_AFTER_CLOSE", "0").lower() in ("1", "true", "yes")
AUTO_HF_UPLOAD_AFTER_CT = datetime.time(15, 20)  # after summary, still close to market close


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


def _parse_central_time(time_str: str | None) -> datetime.datetime:
    now_ct = datetime.datetime.now(_CT)
    if not time_str:
        return now_ct.astimezone(datetime.timezone.utc)
    h, m = map(int, time_str.split(":"))
    trade_ct = now_ct.replace(hour=h, minute=m, second=0, microsecond=0)
    return trade_ct.astimezone(datetime.timezone.utc)


def _fmt_central(dt: datetime.datetime, fmt: str = "%H:%M:%S %Z") -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt.astimezone(_CT).strftime(fmt)


def _current_ct_date(now: datetime.datetime | None = None):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return now.astimezone(_CT).date()


def _ensure_signal_stats_day(now: datetime.datetime | None = None):
    global daily_signal_stats_date, daily_signal_stats
    today = _current_ct_date(now)
    if daily_signal_stats_date != today:
        daily_signal_stats_date = today
        daily_signal_stats = {}


def _record_daily_signal_stats(signals: list):
    _ensure_signal_stats_day()
    for sig in signals:
        if sig.regime not in SUMMARY_REGIMES:
            continue
        by_regime = daily_signal_stats.setdefault(sig.ticker, {})
        by_direction = by_regime.setdefault(sig.regime, {})
        row = by_direction.setdefault(sig.direction, {"count": 0, "conf_sum": 0.0})
        row["count"] += 1
        row["conf_sum"] += sig.confidence


def _empty_signal_stat() -> dict:
    return {"count": 0, "conf_sum": 0.0}


def _sum_direction_stats(by_direction: dict) -> dict:
    out = _empty_signal_stat()
    for row in by_direction.values():
        out["count"] += row["count"]
        out["conf_sum"] += row["conf_sum"]
    return out


def _direction_summary(by_direction: dict) -> str:
    labels = [("LONG", "多"), ("SHORT", "空"), ("NEUTRAL", "铁鹰")]
    parts = []
    for direction, label in labels:
        row = by_direction.get(direction)
        if not row or not row["count"]:
            continue
        avg = row["conf_sum"] / row["count"]
        parts.append(f"{label} `{row['count']}` 次 / `{avg:.1f}%`")
    return "，".join(parts)


def _format_daily_signal_summary() -> str:
    _ensure_signal_stats_day()
    date_str = daily_signal_stats_date.strftime("%Y-%m-%d") if daily_signal_stats_date else "今天"
    if not daily_signal_stats:
        return f"📋 **高周期信号总结** `{date_str}`\n\n今天没有触发 weekly/monthly 信号。"

    rows = []
    for ticker, by_regime in daily_signal_stats.items():
        weekly_by_direction = by_regime.get("weekly", {})
        monthly_by_direction = by_regime.get("monthly", {})
        weekly = _sum_direction_stats(weekly_by_direction)
        monthly = _sum_direction_stats(monthly_by_direction)
        total_count = weekly["count"] + monthly["count"]
        best_avg = 0.0
        for by_direction in (weekly_by_direction, monthly_by_direction):
            for row in by_direction.values():
                if row["count"]:
                    best_avg = max(best_avg, row["conf_sum"] / row["count"])
        rows.append((ticker, weekly_by_direction, monthly_by_direction, weekly, monthly, total_count, best_avg))

    rows.sort(key=lambda r: (r[5], r[6]), reverse=True)
    lines = [f"📋 **高周期信号总结** `{date_str}`", ""]
    for ticker, weekly_by_direction, monthly_by_direction, weekly, monthly, total_count, avg_conf in rows:
        parts = []
        if weekly["count"]:
            parts.append(f"周线：{_direction_summary(weekly_by_direction)}")
        if monthly["count"]:
            parts.append(f"月线：{_direction_summary(monthly_by_direction)}")
        lines.append(f"**{ticker}**  总计 `{total_count}` 次  ·  {'  ·  '.join(parts)}")

    lines.append("")
    lines.append("-# 统计对象：当天所有触发阈值的 weekly/monthly 候选；多=LONG，空=SHORT，铁鹰=monthly NEUTRAL。")
    return "\n".join(lines)


def _prune_signal_recommendations(now: datetime.datetime):
    expired = [
        ticker for ticker, rec in active_signal_recommendations.items()
        if rec["expires_at"] <= now
    ]
    for ticker in expired:
        del active_signal_recommendations[ticker]


def _select_signals_for_send(signals: list) -> list:
    """
    Turn raw model events into a concise recommendation list.
    Pick the strongest tickers for primary display while retaining same-ticker
    secondary regime candidates for comparison.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    _prune_signal_recommendations(now)

    tradable = [sig for sig in signals if sig.direction in ("LONG", "SHORT")]
    grouped = {}
    for sig in sorted(tradable, key=lambda s: s.confidence, reverse=True):
        grouped.setdefault(sig.ticker, []).append(sig)

    primary = sorted((rows[0] for rows in grouped.values()), key=lambda s: s.confidence, reverse=True)
    selected = []
    for sig in primary[:MAX_SIGNALS_PER_POLL]:
        selected.append({
            "primary": sig,
            "secondary": grouped[sig.ticker][1:],
        })
    return selected


def _format_secondary_signals(signals: list) -> str:
    if not signals:
        return ""

    regime_label = {"intraday": "日内", "weekly": "周线", "monthly": "月线"}
    direction_label = {"LONG": "多", "SHORT": "空", "NEUTRAL": "铁鹰"}
    parts = []
    for sig in sorted(signals, key=lambda s: s.confidence, reverse=True):
        target = f"{sig.target_pct:+.1f}%" if sig.direction != "NEUTRAL" else "—"
        parts.append(
            f"{regime_label.get(sig.regime, sig.regime)}"
            f"{direction_label.get(sig.direction, sig.direction)}"
            f" `{sig.confidence:.1f}%` 目标 `{target}`"
        )

    return "\n\n-# 同 ticker 其他周期： " + "  ·  ".join(parts)


def _signal_context_note(sig) -> str:
    active = active_signal_recommendations.get(sig.ticker)
    if not active:
        return ""
    if active["direction"] != sig.direction:
        return (
            f"\n\n-# 🔄 观点更新：此前 {active['regime']} `{active['direction']}`，"
            f"现在 {sig.regime} `{sig.direction}`。如已有仓位，请优先按止损/减仓规则处理。"
        )
    if active["regime"] != sig.regime:
        return (
            f"\n\n-# 同向补充：此前 {active['regime']} `{active['direction']}`，"
            f"现在 {sig.regime} 也给出 `{sig.direction}`。"
        )
    return ""


def _record_signal_sent(sig):
    now = datetime.datetime.now(datetime.timezone.utc)
    hold_days = max(sig.timeframe_days, 1)
    active_signal_recommendations[sig.ticker] = {
        "ticker": sig.ticker,
        "regime": sig.regime,
        "direction": sig.direction,
        "sent_at": now,
        "expires_at": now + datetime.timedelta(days=hold_days),
        "confidence": sig.confidence,
    }


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

    _record_daily_signal_stats(signals)
    signal_groups = _select_signals_for_send(signals)
    for group in signal_groups:
        sig = group["primary"]
        msg = format_signal(sig) + _signal_context_note(sig) + _format_secondary_signals(group["secondary"])
        await _send(msg)
        engine.mark_signal_sent(sig)
        _record_signal_sent(sig)
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


@tasks.loop(minutes=15)
async def daily_summary_loop():
    global daily_summary_sent_date
    now_ct = datetime.datetime.now(_CT)
    if now_ct.weekday() >= 5 or now_ct.time() < SUMMARY_SEND_AFTER_CT:
        return
    today = now_ct.date()
    if daily_summary_sent_date == today:
        return
    await _send(_format_daily_signal_summary())
    daily_summary_sent_date = today


def _run_hf_upload_once() -> tuple[bool, str]:
    cmd = [sys.executable, "scripts/upload_to_hf.py"]
    proc = subprocess.run(
        cmd,
        cwd=Path(__file__).parent,
        text=True,
        capture_output=True,
        timeout=60 * 60,
    )
    output = "\n".join(part for part in (proc.stdout.strip(), proc.stderr.strip()) if part)
    return proc.returncode == 0, output[-1500:]


@tasks.loop(minutes=15)
async def daily_hf_upload_loop():
    global daily_hf_upload_sent_date
    if not AUTO_HF_UPLOAD_ENABLED:
        return
    now_ct = datetime.datetime.now(_CT)
    if now_ct.weekday() >= 5 or now_ct.time() < AUTO_HF_UPLOAD_AFTER_CT:
        return
    today = now_ct.date()
    if daily_hf_upload_sent_date == today:
        return

    loop = asyncio.get_event_loop()
    ok, output = await loop.run_in_executor(None, _run_hf_upload_once)
    daily_hf_upload_sent_date = today
    if ok:
        await _send("☁️ **Hugging Face 上传完成**\n\n-# raw data / models 已同步。")
        log.info("Daily Hugging Face upload complete: %s", output)
    else:
        await _send(f"⚠️ **Hugging Face 上传失败**\n\n```text\n{output or 'No output'}\n```")
        log.error("Daily Hugging Face upload failed: %s", output)


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
        daily_summary_loop.start()
        daily_hf_upload_loop.start()
        log.info(f"Signal loop started — polling every 1 min, window {MARKET_OPEN}–{MARKET_CLOSE} ET (weekdays only)")
        log.info("Daily weekly/monthly summary loop started")
        if AUTO_HF_UPLOAD_ENABLED:
            log.info("Daily Hugging Face upload loop started")


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
    lines = [f"**── Engine Debug  {_fmt_central(now)} ──**", ""]

    for ticker in list(engine.tickers)[:3]:   # first 3 tickers to avoid message limit
        acc      = engine.accumulators[ticker]
        last_ts  = engine.last_ts.get(ticker)
        hist_len = len(engine.histories[ticker])
        hist_last = engine.histories[ticker][-1] if engine.histories[ticker] else None

        ts_str  = (_fmt_central(datetime.datetime.fromtimestamp(
                       last_ts / 1000, tz=datetime.timezone.utc
                   )) if last_ts else "None")
        if hist_last:
            hist_ts = hist_last["timestamp"]
            if hasattr(hist_ts, "to_pydatetime"):
                hist_ts = hist_ts.to_pydatetime()
            elif not hasattr(hist_ts, "tzinfo"):
                hist_ts = datetime.datetime.fromtimestamp(hist_ts / 1000, tz=datetime.timezone.utc)
            bar_str = _fmt_central(hist_ts, "%m/%d %H:%M %Z")
        else:
            bar_str = "—"

        lines += [
            f"**{ticker}**",
            f"  last_ts fetched : `{ts_str}`",
            f"  acc cum_dollar  : `${acc._cum_dollar:,.0f}`",
            f"  acc threshold   : `${acc.threshold:,.0f}`",
            f"  acc bucket bars : `{len(acc._bucket)}`",
            f"  history bars    : `{hist_len}`  (latest {bar_str})",
            "",
        ]

    await ctx.send("\n".join(lines))


@bot.command(name="status")
async def status(ctx: commands.Context):
    """Manually trigger a status update."""
    msg = format_status(last_signal_time, list(SIGNAL_CONFIG.keys()))
    await ctx.send(msg)


@bot.command(name="summary")
async def summary(ctx: commands.Context):
    """Show today's weekly/monthly trigger summary."""
    await ctx.send(_format_daily_signal_summary())


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

    if time_str:
        try:
            now_utc = _parse_central_time(time_str)
        except ValueError:
            await ctx.send("时间格式错误，请用美中时间 `HH:MM`，例如 `14:30`")
            return
    else:
        now_utc = _parse_central_time(None)

    key = f"{ticker}_{regime}"
    if key in open_positions:
        existing = open_positions[key]
        await ctx.send(
            f"⚠️ `{ticker} {regime}` 已有未平仓记录  "
            f"({existing['direction']} @ ${existing['entry_price']:,.2f})，"
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
            target_floor = max(
                TARGET_RETURN_FLOOR.get(regime, 0.0),
                cfg.get("min_target_return", 0.0),
            )
            target_distance = max(h_target * vol, target_floor)
            target   = price * (1 + target_distance) if direction == "LONG" else price * (1 - target_distance)
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

    if time_str:
        try:
            now_utc = _parse_central_time(time_str)
        except ValueError:
            await ctx.send("时间格式错误，请用美中时间 `HH:MM`，例如 `16:00`")
            return
    else:
        now_utc = _parse_central_time(None)

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
async def cmd_signal(ctx: commands.Context, ticker: str = None):
    """
    显示当前模型预测。
    用法: !signal          — 所有标的
          !signal SPY      — 指定标的
    """
    loop  = asyncio.get_event_loop()
    scans = await loop.run_in_executor(None, engine.scan)
    if ticker:
        scans = [s for s in scans if s.ticker == ticker.upper()]
    else:
        scans = sorted(
            scans,
            key=lambda s: (s.above_threshold, s.direction != "NEUTRAL", s.confidence),
            reverse=True,
        )[:MAX_SCAN_ROWS]
    if scans:
        await ctx.send(format_predictions(scans))
    else:
        await ctx.send("暂无预测数据（引擎数据不足，请等待 warmup 完成）")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_TOKEN not set")
    bot.run(token)
