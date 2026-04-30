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
    await ctx.send("Polling tickers ...")
    try:
        signals = engine.poll()
    except Exception as e:
        await ctx.send(f"Error: {e}")
        return
    if signals:
        for sig in signals:
            await ctx.send(format_signal(sig))
    else:
        await ctx.send("No signals above threshold right now.")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_TOKEN not set")
    bot.run(token)
