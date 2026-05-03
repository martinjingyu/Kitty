"""Discord message formatting for trading signals and status updates."""
from datetime import datetime, timezone, timedelta
from signals.engine import Signal, ProbaScan

REGIME_LABEL = {"intraday": "日内", "weekly": "1 周", "monthly": "1 个月"}
REGIME_TYPE  = {"intraday": "日内方向", "weekly": "反转信号", "monthly": "月度趋势"}


def format_signal(sig: Signal) -> str:
    direction_icon = "📈" if sig.direction == "LONG" else "📉"
    regime_label   = REGIME_LABEL.get(sig.regime, sig.regime)
    regime_type    = REGIME_TYPE.get(sig.regime, sig.regime)
    exit_date      = (datetime.now(timezone.utc) + timedelta(days=max(sig.timeframe_days, 1)))
    exit_str       = exit_date.strftime("%m/%d")

    target_sign = "+" if sig.target_pct > 0 else ""
    stop_sign   = "+" if sig.stop_pct   > 0 else ""

    # risk/reward ratio: target distance / stop distance (always positive)
    rr = abs(sig.target_pct) / abs(sig.stop_pct) if sig.stop_pct != 0 else 0.0

    lines = [
        f"{direction_icon} **{sig.ticker}  {sig.direction}  [{regime_label}]**",
        f"-# {regime_type}",
        "",
        f"**入场价**   `${sig.entry_price:,.2f}`",
        f"**目标价**   `${sig.target_price:,.2f}`  ({target_sign}{sig.target_pct:.2f}%)",
        f"**止损价**   `${sig.stop_price:,.2f}`  ({stop_sign}{sig.stop_pct:.2f}%)",
        f"**盈亏比**   `{rr:.2f}x`  (波动率 {sig.vol:.2f}%/bar)",
        "",
        f"**持仓周期**  {regime_label}  · 建议 {exit_str} 前平仓",
        f"**置信度**    `{sig.confidence:.1f}%`",
        "",
        f"-# {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC  ·  数据延迟约 15 分钟",
    ]
    return "\n".join(lines)


def format_open_confirm(ticker: str, regime: str, direction: str,
                        price: float, trade_time: datetime,
                        target: float | None = None) -> str:
    icon         = "📈" if direction == "LONG" else "📉"
    regime_label = REGIME_LABEL.get(regime, regime)
    time_str     = trade_time.strftime("%Y-%m-%d %H:%M")
    target_line  = f"**止盈目标**  `${target:,.2f}`" if target else "**止盈目标**  —"
    lines = [
        f"{icon} **{ticker}  开仓记录  [{regime_label}]**",
        "",
        f"**方向**     `{direction}`",
        f"**入场价**   `${price:,.2f}`",
        target_line,
        f"**时间**     `{time_str} UTC`",
        "",
        f"-# `!close {ticker} {regime} <出场价> [50%] [HH:MM]` 全仓或部分平仓",
    ]
    return "\n".join(lines)


def _pnl(direction: str, entry: float, exit_p: float) -> float:
    return (exit_p / entry - 1) * 100 if direction == "LONG" else (entry / exit_p - 1) * 100


def format_close_confirm(ticker: str, regime: str, direction: str,
                         entry_price: float, exit_price: float,
                         entry_time: datetime, exit_time: datetime,
                         close_pct: float = 1.0, remaining_pct: float = 0.0) -> str:
    pnl_pct      = _pnl(direction, entry_price, exit_price)
    icon         = "✅" if pnl_pct >= 0 else "🛑"
    pnl_sign     = "+" if pnl_pct >= 0 else ""
    held_min     = int((exit_time - entry_time).total_seconds() / 60)
    held_str     = f"{held_min // 60}h {held_min % 60:02d}m" if held_min >= 60 else f"{held_min}m"
    regime_label = REGIME_LABEL.get(regime, regime)
    partial_note = f"已平 {close_pct:.0%}，剩余 {remaining_pct:.0%}" if remaining_pct > 0 else "全仓平仓"

    lines = [
        f"{icon} **{ticker}  平仓记录  [{regime_label}]**",
        "",
        f"**方向**     `{direction}`",
        f"**入场价**   `${entry_price:,.2f}`  @ `{entry_time.strftime('%m/%d %H:%M')} UTC`",
        f"**出场价**   `${exit_price:,.2f}`  @ `{exit_time.strftime('%m/%d %H:%M')} UTC`",
        f"**盈亏**     `{pnl_sign}{pnl_pct:.2f}%`  ·  {partial_note}",
        f"**持仓时长** `{held_str}`",
    ]
    return "\n".join(lines)


def format_sl_alarm(pos: dict, proba_against: float,
                    proba_long: float, proba_short: float) -> str:
    """Model direction flipped against open position."""
    direction    = pos["direction"]
    against_dir  = "空" if direction == "LONG" else "多"
    regime_label = REGIME_LABEL.get(pos["regime"], pos["regime"])
    pnl_now      = _pnl(direction, pos["entry_price"], pos.get("last_price", pos["entry_price"]))
    pnl_sign     = "+" if pnl_now >= 0 else ""

    lines = [
        f"⚠️ **{pos['ticker']}  止损预警  [{regime_label}]**",
        f"-# 模型 {against_dir}信号置信度偏高，建议减仓或止损",
        "",
        f"**持仓方向**   `{direction}`  入场价 `${pos['entry_price']:,.2f}`",
        f"**当前盈亏**   `{pnl_sign}{pnl_now:.2f}%`",
        f"**模型概率**   多 `{proba_long:.1%}`  空 `{proba_short:.1%}`",
        f"**反方向置信** `{proba_against:.1%}`",
        "",
        f"-# `!close {pos['ticker']} {pos['regime']} <价格>` 平仓",
    ]
    return "\n".join(lines)


def format_tp_alarm(pos: dict, current_price: float,
                    proba_long: float, proba_short: float) -> str:
    """Price reached TP target. Annotate current model signal."""
    direction    = pos["direction"]
    target       = pos["target"]
    regime_label = REGIME_LABEL.get(pos["regime"], pos["regime"])
    pnl_now      = _pnl(direction, pos["entry_price"], current_price)
    pnl_sign     = "+" if pnl_now >= 0 else ""

    # Model annotation
    if direction == "LONG" and proba_long > proba_short * 1.2:
        model_note = f"📈 模型仍看多 (`{proba_long:.1%}`)，可考虑部分止盈继续持仓"
    elif direction == "SHORT" and proba_short > proba_long * 1.2:
        model_note = f"📉 模型仍看空 (`{proba_short:.1%}`)，可考虑部分止盈继续持仓"
    else:
        model_note = f"⬜ 模型信号趋于中性（多 `{proba_long:.1%}` 空 `{proba_short:.1%}`），建议止盈"

    lines = [
        f"🎯 **{pos['ticker']}  止盈预警  [{regime_label}]**",
        "",
        f"**目标价**     `${target:,.2f}`  已触达",
        f"**当前价**     `${current_price:,.2f}`",
        f"**持仓盈亏**   `{pnl_sign}{pnl_now:.2f}%`  (持仓 {pos['remaining_pct']:.0%})",
        "",
        model_note,
        "",
        f"-# `!close {pos['ticker']} {pos['regime']} <价格> [50%]` 全仓或部分平仓",
    ]
    return "\n".join(lines)


def format_scan(scans: list[ProbaScan]) -> str:
    now = datetime.now(timezone.utc)
    lines = [f"**── 市场扫描  {now.strftime('%m/%d %H:%M')} UTC ──**", ""]

    # group by ticker
    by_ticker: dict[str, list[ProbaScan]] = {}
    for s in scans:
        by_ticker.setdefault(s.ticker, []).append(s)

    dir_icon = {"LONG": "📈", "SHORT": "📉", "NEUTRAL": "⬜", "CONDOR": "🦅"}

    for ticker, rows in by_ticker.items():
        price_str = f"${rows[0].price:,.2f}"
        lines.append(f"**{ticker}**  `{price_str}`")
        for s in rows:
            regime_cn = REGIME_LABEL.get(s.regime, s.regime)
            # monthly 3-class: NEUTRAL means CONDOR (iron condor candidate)
            display_dir = "CONDOR" if s.direction == "NEUTRAL" and s.regime == "monthly" else s.direction
            icon        = dir_icon.get(display_dir, "⬜")
            conf_str    = f"{s.confidence:.1%}"
            if s.proba_neutral > 0:
                detail = (f"多 {s.proba_long:.1%}  空 {s.proba_short:.1%}"
                          f"  condor {s.proba_neutral:.1%}")
            else:
                detail = f"多 {s.proba_long:.1%}  空 {s.proba_short:.1%}"
            lines.append(
                f"  {regime_cn:<5} {icon} **{display_dir:<7}** `{conf_str}`"
                f"  ·  {detail}"
            )
        lines.append("")

    lines.append(f"-# 每 30 分钟更新  ·  数据延迟约 15 分钟")
    return "\n".join(lines)


def format_status(last_signals: dict[str, datetime], tickers: list[str]) -> str:
    now   = datetime.now(timezone.utc)
    lines = ["**── 信号监控状态 ──**", ""]
    for ticker in tickers:
        last = last_signals.get(ticker)
        if last:
            delta = now - last
            h, m  = divmod(int(delta.total_seconds()), 3600)
            m     = m // 60
            ago   = f"{h}h {m:02d}m 前" if h else f"{m}m 前"
            lines.append(f"`{ticker}`  上次信号: {ago}")
        else:
            lines.append(f"`{ticker}`  暂无信号")
    lines.append("")
    lines.append(f"-# 过去 1 小时内无新信号  ·  {now.strftime('%H:%M')} UTC")
    return "\n".join(lines)
