import os
import datetime
import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

load_dotenv()

intents = discord.Intents.default()
intents.message_content = True  # needed to read message text

bot = commands.Bot(command_prefix="!", intents=intents)

CHANNEL_ID = int(os.getenv("SCHEDULE_CHANNEL_ID", "0"))
WINDOW_START = datetime.time(8, 30)
WINDOW_END = datetime.time(15, 30)


@tasks.loop(minutes=1)  # TEST: 改回 minutes=10 并取消注释下面的时间窗口判断
async def scheduled_message():
    # now = datetime.datetime.now().time()  # TEST: 取消注释恢复时间窗口
    # if not (WINDOW_START <= now <= WINDOW_END):  # TEST: 取消注释恢复时间窗口
    #     return  # TEST: 取消注释恢复时间窗口

    channel = bot.get_channel(CHANNEL_ID)
    if channel is None:
        print(f"Channel {CHANNEL_ID} not found — check SCHEDULE_CHANNEL_ID in .env")
        return

    await channel.send(f"定时消息 · {datetime.datetime.now().strftime('%H:%M')}")


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    print(f"Connected to {len(bot.guilds)} guild(s)")
    if CHANNEL_ID:
        scheduled_message.start()
        print(f"Scheduler started — every 10 min, window {WINDOW_START}–{WINDOW_END}")


@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return

    print(f"[#{message.channel}] {message.author}: {message.content}")

    # process commands before any other logic
    await bot.process_commands(message)


@bot.command(name="ping")
async def ping(ctx: commands.Context):
    """Reply with pong — basic connectivity check."""
    await ctx.send(f"Pong! Latency: {round(bot.latency * 1000)}ms")


@bot.command(name="say")
async def say(ctx: commands.Context, *, text: str):
    """Echo text back to the channel."""
    await ctx.message.delete()
    await ctx.send(text)


if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_TOKEN not set — copy .env.example to .env and fill it in")
    bot.run(token)
