"""Discord message formatting for trading signals and status updates."""
from datetime import datetime, timezone, timedelta
from signals.engine import Signal


def format_signal(sig: Signal) -> str:
    direction_icon = "📈" if sig.direction == "LONG" else "📉"
    regime_label   = {"weekly": "1 周", "monthly": "1 个月"}[sig.regime]
    exit_date      = (datetime.now(timezone.utc) + timedelta(days=sig.timeframe_days))
    exit_str       = exit_date.strftime("%m/%d")

    target_sign = "+" if sig.target_pct > 0 else ""
    stop_sign   = "+" if sig.stop_pct   > 0 else ""

    lines = [
        f"{direction_icon} **{sig.ticker}  {sig.direction}  [{regime_label}]**",
        "",
        f"**入场价**   `${sig.entry_price:,.2f}`",
        f"**目标价**   `${sig.target_price:,.2f}`  ({target_sign}{sig.target_pct:.2f}%)",
        f"**止损价**   `${sig.stop_price:,.2f}`  ({stop_sign}{sig.stop_pct:.2f}%)",
        "",
        f"**预计涨幅**  `{abs(sig.target_pct):.2f}%`  (波动率 {sig.vol:.2f}%/bar)",
        f"**持仓周期**  {regime_label}  · 建议 {exit_str} 前平仓",
        f"**置信度**    `{sig.confidence:.1f}%`",
        "",
        f"-# {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC  ·  数据延迟约 15 分钟",
    ]
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
