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
                                format_sl_alarm)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

def _env_int(name: str, default: int = 0) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


LEGACY_CHANNEL_ID = _env_int("SCHEDULE_CHANNEL_ID")
SIGNAL_CHANNEL_ID = _env_int("SIGNAL_CHANNEL_ID", LEGACY_CHANNEL_ID)
INTERACTION_CHANNEL_ID = _env_int("INTERACTION_CHANNEL_ID", LEGACY_CHANNEL_ID)
POSITION_CHANNEL_ID = _env_int("POSITION_CHANNEL_ID", SIGNAL_CHANNEL_ID)
MOOMOO_POSITIONS_ENABLED = os.getenv("MOOMOO_POSITIONS_ENABLED", "0").lower() in ("1", "true", "yes")
MOOMOO_AUTO_POSITION_SNAPSHOT = os.getenv("MOOMOO_AUTO_POSITION_SNAPSHOT", "0").lower() in ("1", "true", "yes")

# ── state ─────────────────────────────────────────────────────────────────────
engine = None
last_signal_time: dict = {}   # {ticker: datetime}
last_status_sent = None

broker_positions: dict = {}  # {ticker: BrokerPosition}
broker_sl_alerted: set[str] = set()
tracked_tickers: set[str] = set()
active_signal_recommendations: dict = {}  # {ticker: dict}
daily_signal_stats: dict = {}             # {ticker: {regime: {direction: {count, conf_sum}}}}
daily_signal_stats_date = None
daily_summary_sent_date = None
daily_hf_upload_sent_date = None
last_broker_position_refresh = None
last_broker_position_snapshot = None

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


def _direction_count(by_direction: dict, direction: str) -> int:
    return by_direction.get(direction, {}).get("count", 0)


def _direction_avg(by_direction: dict, direction: str) -> float:
    row = by_direction.get(direction)
    if not row or not row["count"]:
        return 0.0
    return row["conf_sum"] / row["count"]


def _direction_score(by_direction: dict, direction: str) -> float:
    return _direction_count(by_direction, direction) * _direction_avg(by_direction, direction)


def _classify_signal_summary(weekly_by_direction: dict, monthly_by_direction: dict) -> tuple[int, str]:
    w_long = _direction_count(weekly_by_direction, "LONG")
    w_short = _direction_count(weekly_by_direction, "SHORT")
    m_long = _direction_count(monthly_by_direction, "LONG")
    m_short = _direction_count(monthly_by_direction, "SHORT")
    m_neutral = _direction_count(monthly_by_direction, "NEUTRAL")

    long_score = _direction_score(weekly_by_direction, "LONG") + _direction_score(monthly_by_direction, "LONG")
    short_score = _direction_score(weekly_by_direction, "SHORT") + _direction_score(monthly_by_direction, "SHORT")
    neutral_score = _direction_score(monthly_by_direction, "NEUTRAL")

    has_long = w_long + m_long > 0
    has_short = w_short + m_short > 0
    has_monthly_long = m_long > 0
    has_monthly_short = m_short > 0
    has_weekly_long = w_long > 0
    has_weekly_short = w_short > 0

    if has_long and has_short and min(long_score, short_score) >= max(long_score, short_score) * 0.25:
        return 1, "⚠️ 多空冲突"
    if has_weekly_long and has_monthly_long and short_score <= long_score * 0.25:
        return 0, "✅ 多头共振"
    if has_weekly_short and has_monthly_short and long_score <= short_score * 0.25:
        return 0, "✅ 空头共振"
    if m_neutral and neutral_score >= max(long_score, short_score) * 0.6:
        return 2, "🦅 铁鹰/震荡"
    if long_score > short_score:
        return 3, "📈 偏多观察"
    if short_score > long_score:
        return 3, "📉 偏空观察"
    return 4, "⬜ 观察"


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
        rank, label = _classify_signal_summary(weekly_by_direction, monthly_by_direction)
        rows.append((rank, ticker, label, weekly_by_direction, monthly_by_direction, weekly, monthly, total_count, best_avg))

    rows.sort(key=lambda r: (r[0], -r[7], -r[8]))
    lines = [f"📋 **高周期信号总结** `{date_str}`", ""]
    for _, ticker, label, weekly_by_direction, monthly_by_direction, weekly, monthly, total_count, _ in rows:
        parts = []
        if weekly["count"]:
            parts.append(f"周线：{_direction_summary(weekly_by_direction)}")
        if monthly["count"]:
            parts.append(f"月线：{_direction_summary(monthly_by_direction)}")
        lines.append(f"{label}  **{ticker}**  总计 `{total_count}` 次  ·  {'  ·  '.join(parts)}")

    lines.append("")
    lines.append("-# 标签按高周期方向一致性和强度自动归类；多=LONG，空=SHORT，铁鹰=monthly NEUTRAL。")
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


def _filter_scans_for_tickers(scans: list, tickers: set[str]) -> list:
    wanted = {ticker.upper() for ticker in tickers}
    return [s for s in scans if s.ticker in wanted]


def _broker_direction(pos) -> str:
    side = str(pos.side).upper()
    if "SHORT" in side:
        return "SHORT"
    if pos.qty < 0:
        return "SHORT"
    return "LONG"


def _broker_position_alarm_dict(pos, regime: str, direction: str) -> dict:
    return {
        "ticker": pos.ticker,
        "regime": regime,
        "direction": direction,
        "entry_price": pos.cost or pos.price,
        "last_price": pos.price,
    }


def _check_alarms() -> list[str]:
    """
    Sync function (run in executor).
    Returns model-driven risk alarm messages for actual broker positions.

    SL: P(against_direction) > threshold for that regime.
    """
    if not engine or not broker_positions:
        return []

    msgs = []
    for ticker, broker_pos in list(broker_positions.items()):
        direction = _broker_direction(broker_pos)
        for regime_entry in SIGNAL_CONFIG.get(ticker, []):
            regime, thr_long, thr_short, *_ = regime_entry
            proba = engine.get_proba(ticker, regime)
            if proba is None:
                continue
            proba_long, proba_short, _ = proba
            proba_against = proba_short if direction == "LONG" else proba_long
            thr = thr_short if direction == "LONG" else thr_long
            key = f"{ticker}_{regime}"
            alarm_pos = _broker_position_alarm_dict(broker_pos, regime, direction)
            if proba_against > thr:
                if key not in broker_sl_alerted:
                    msgs.append(format_sl_alarm(alarm_pos, proba_against, proba_long, proba_short))
                    broker_sl_alerted.add(key)
            else:
                broker_sl_alerted.discard(key)

    return msgs


async def _send(content: str, channel_id: int | None = None):
    channel_id = channel_id or SIGNAL_CHANNEL_ID
    channel = bot.get_channel(channel_id)
    if channel:
        await channel.send(content)
    else:
        log.warning(f"Channel {channel_id} not found")


async def _send_signal(content: str):
    await _send(content, SIGNAL_CHANNEL_ID)


async def _send_interaction(content: str):
    await _send(content, INTERACTION_CHANNEL_ID)


async def _send_position(content: str):
    await _send(content, POSITION_CHANNEL_ID)


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
        await _send_signal(msg)
        engine.mark_signal_sent(sig)
        _record_signal_sent(sig)
        last_signal_time[sig.ticker] = datetime.datetime.now(datetime.timezone.utc)
        log.info(f"Signal: {sig.ticker} {sig.regime} {sig.direction} @ {sig.entry_price}")

    if tracked_tickers:
        try:
            scans = await loop.run_in_executor(None, engine.scan)
            scans = _filter_scans_for_tickers(scans, tracked_tickers)
            if scans:
                await _send_interaction(format_predictions(scans))
        except Exception as e:
            log.error(f"Track scan error: {e}")

    if MOOMOO_POSITIONS_ENABLED:
        global last_broker_position_refresh, last_broker_position_snapshot
        now = datetime.datetime.now(datetime.timezone.utc)
        try:
            if (
                last_broker_position_refresh is None
                or (now - last_broker_position_refresh).total_seconds() >= 5 * 60
            ):
                await loop.run_in_executor(None, _refresh_broker_positions, False)
                last_broker_position_refresh = now
        except Exception as e:
            log.error(f"Moomoo position refresh error: {e}")

    # Check model-driven risk alarms for actual broker positions
    if broker_positions:
        alarms = await loop.run_in_executor(None, _check_alarms)
        for alarm_msg in alarms:
            await _send_position(alarm_msg)

    if MOOMOO_POSITIONS_ENABLED and MOOMOO_AUTO_POSITION_SNAPSHOT:
        now = datetime.datetime.now(datetime.timezone.utc)
        if (
            last_broker_position_snapshot is None
            or (now - last_broker_position_snapshot).total_seconds() >= 30 * 60
        ):
            try:
                msg = await loop.run_in_executor(None, _refresh_broker_positions, False)
                await _send_position(msg)
                last_broker_position_snapshot = now
            except Exception as e:
                log.error(f"Moomoo position snapshot error: {e}")

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
        await _send_signal(msg)
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
    await _send_signal(_format_daily_signal_summary())
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


def _refresh_broker_positions(refresh_cache: bool = False) -> str:
    global broker_positions
    from integrations.moomoo_positions import fetch_positions, format_broker_positions
    positions = fetch_positions(refresh_cache=refresh_cache)
    broker_positions = {pos.ticker: pos for pos in positions}
    active = set(broker_positions)
    broker_sl_alerted.intersection_update(active)
    return format_broker_positions(positions)


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
        await _send_interaction("☁️ **Hugging Face 上传完成**\n\n-# raw data / models 已同步。")
        log.info("Daily Hugging Face upload complete: %s", output)
    else:
        await _send_interaction(f"⚠️ **Hugging Face 上传失败**\n\n```text\n{output or 'No output'}\n```")
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

    if SIGNAL_CHANNEL_ID:
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
async def cmd_open(ctx: commands.Context, *_):
    """Deprecated: positions now come from moomoo/OpenD only."""
    await ctx.send("`!open` 已停用。现在仓位只读取 moomoo 实际持仓，请用 `!positions` 查看。")


@bot.command(name="close")
async def cmd_close(ctx: commands.Context, *_):
    """Deprecated: positions now come from moomoo/OpenD only."""
    await ctx.send("`!close` 已停用。请在 moomoo 里实际平仓；bot 会从 moomoo 读取最新持仓。")


@bot.command(name="positions")
async def cmd_positions(ctx: commands.Context):
    """Fetch actual moomoo/OpenD positions and send them to the position channel."""
    if not MOOMOO_POSITIONS_ENABLED:
        await ctx.send("moomoo 持仓读取未启用，请设置 `MOOMOO_POSITIONS_ENABLED=1`。")
        return

    loop = asyncio.get_event_loop()
    try:
        msg = await loop.run_in_executor(None, _refresh_broker_positions, True)
    except Exception as e:
        await ctx.send(f"读取 moomoo 持仓失败：`{e}`")
        return

    await _send_position(msg)
    if ctx.channel.id != POSITION_CHANNEL_ID:
        await ctx.send("实际持仓已发送到 position channel。")


@bot.command(name="broker_positions")
async def cmd_broker_positions(ctx: commands.Context):
    """Fetch actual moomoo/OpenD positions and send them to the position channel."""
    if not MOOMOO_POSITIONS_ENABLED:
        await ctx.send("moomoo 持仓读取未启用，请设置 `MOOMOO_POSITIONS_ENABLED=1`。")
        return
    loop = asyncio.get_event_loop()
    try:
        msg = await loop.run_in_executor(None, _refresh_broker_positions, True)
    except Exception as e:
        await ctx.send(f"读取 moomoo 持仓失败：`{e}`")
        return
    await _send_position(msg)
    if ctx.channel.id != POSITION_CHANNEL_ID:
        await ctx.send("实际持仓已发送到 position channel。")


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


@bot.command(name="track")
async def cmd_track(ctx: commands.Context, ticker: str):
    """
    Add a ticker to the continuous interaction-channel tracker.
    用法: !track SPY
    """
    ticker = ticker.upper()
    if ticker not in SIGNAL_CONFIG:
        await ctx.send(f"`{ticker}` 未启用。")
        return
    tracked_tickers.add(ticker)

    loop = asyncio.get_event_loop()
    scans = await loop.run_in_executor(None, engine.scan)
    scans = [s for s in scans if s.ticker == ticker]
    if scans:
        await _send_interaction(format_predictions(scans))
        if ctx.channel.id != INTERACTION_CHANNEL_ID:
            await ctx.send(f"`{ticker}` 已加入持续追踪，并已发送当前状态到交互频道。")
    else:
        await _send_interaction(f"`{ticker}` 已加入持续追踪，但当前暂无预测数据（引擎数据不足）。")
        if ctx.channel.id != INTERACTION_CHANNEL_ID:
            await ctx.send(f"`{ticker}` 已加入持续追踪，但当前暂无预测数据。")


@bot.command(name="untrack")
async def cmd_untrack(ctx: commands.Context, ticker: str):
    """Remove a ticker from the continuous tracker."""
    ticker = ticker.upper()
    if ticker in tracked_tickers:
        tracked_tickers.remove(ticker)
        await ctx.send(f"`{ticker}` 已停止追踪。")
    else:
        await ctx.send(f"`{ticker}` 当前不在追踪列表。")


@bot.command(name="tracklist")
async def cmd_tracklist(ctx: commands.Context):
    """Show tracked tickers."""
    if not tracked_tickers:
        await ctx.send("当前没有持续追踪的 ticker。")
        return
    await ctx.send("持续追踪：" + " ".join(f"`{ticker}`" for ticker in sorted(tracked_tickers)))


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_TOKEN not set")
    bot.run(token)
