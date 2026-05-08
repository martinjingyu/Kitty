"""Read-only moomoo / OpenD position integration."""
import importlib
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class BrokerPosition:
    ticker: str
    name: str
    side: str
    qty: float
    price: float
    cost: float
    market_value: float
    pl_value: float
    pl_ratio: float
    currency: str


def _load_sdk():
    preferred = os.getenv("MOOMOO_SDK_MODULE", "moomoo")
    candidates = [preferred]
    if preferred != "futu":
        candidates.append("futu")
    if preferred != "moomoo":
        candidates.append("moomoo")

    last_error = None
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            last_error = exc
    raise RuntimeError(
        "moomoo/futu Python SDK is not installed. Install `moomoo-api` "
        "or set MOOMOO_SDK_MODULE to the installed SDK module."
    ) from last_error


def _sdk_attr(sdk: Any, name: str):
    try:
        return getattr(sdk, name)
    except AttributeError as exc:
        raise RuntimeError(f"Moomoo SDK missing expected attribute: {name}") from exc


def _enum_value(enum_obj: Any, name: str):
    if hasattr(enum_obj, name):
        return getattr(enum_obj, name)
    upper_name = name.upper()
    if hasattr(enum_obj, upper_name):
        return getattr(enum_obj, upper_name)
    raise RuntimeError(f"Moomoo SDK enum {enum_obj} has no value {name}")


def _normalize_code(code: str) -> str:
    if "." in code:
        return code.split(".", 1)[1]
    return code


def _num(row: Any, key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
    except AttributeError:
        value = row[key] if key in row else default
    if value is None or value == "":
        return default
    return float(value)


def _text(row: Any, key: str, default: str = "") -> str:
    try:
        value = row.get(key, default)
    except AttributeError:
        value = row[key] if key in row else default
    return default if value is None else str(value)


def fetch_positions(refresh_cache: bool = False) -> list[BrokerPosition]:
    """
    Query actual broker positions through local OpenD.

    Requires OpenD running and logged in. This function only queries positions;
    it does not unlock trading or place orders.
    """
    sdk = _load_sdk()
    OpenSecTradeContext = _sdk_attr(sdk, "OpenSecTradeContext")
    TrdMarket = _sdk_attr(sdk, "TrdMarket")
    TrdEnv = _sdk_attr(sdk, "TrdEnv")
    RET_OK = _sdk_attr(sdk, "RET_OK")

    host = os.getenv("MOOMOO_HOST", "127.0.0.1")
    port = int(os.getenv("MOOMOO_PORT", "11111"))
    market = _enum_value(TrdMarket, os.getenv("MOOMOO_TRD_MARKET", "US"))
    trd_env = _enum_value(TrdEnv, os.getenv("MOOMOO_TRD_ENV", "REAL"))
    acc_id = int(os.getenv("MOOMOO_ACC_ID", "0"))
    acc_index = int(os.getenv("MOOMOO_ACC_INDEX", "0"))

    kwargs = {
        "filter_trdmarket": market,
        "host": host,
        "port": port,
    }
    security_firm_name = os.getenv("MOOMOO_SECURITY_FIRM")
    if security_firm_name:
        SecurityFirm = _sdk_attr(sdk, "SecurityFirm")
        kwargs["security_firm"] = _enum_value(SecurityFirm, security_firm_name)

    trd_ctx = OpenSecTradeContext(**kwargs)
    try:
        ret, data = trd_ctx.position_list_query(
            trd_env=trd_env,
            acc_id=acc_id,
            acc_index=acc_index,
            refresh_cache=refresh_cache,
        )
        if ret != RET_OK:
            raise RuntimeError(f"position_list_query failed: {data}")

        positions = []
        for _, row in data.iterrows():
            qty = _num(row, "qty")
            if qty == 0:
                continue
            code = _text(row, "code")
            side = _text(row, "position_side", "LONG")
            cost = _num(row, "cost_price", _num(row, "average_cost", _num(row, "diluted_cost")))
            positions.append(BrokerPosition(
                ticker=_normalize_code(code),
                name=_text(row, "stock_name", code),
                side=side,
                qty=qty,
                price=_num(row, "nominal_price"),
                cost=cost,
                market_value=_num(row, "market_val"),
                pl_value=_num(row, "pl_val", _num(row, "unrealized_pl")),
                pl_ratio=_num(row, "pl_ratio"),
                currency=_text(row, "currency", "USD"),
            ))
        return positions
    finally:
        trd_ctx.close()


def format_broker_positions(positions: list[BrokerPosition]) -> str:
    if not positions:
        return "📒 **实际持仓**\n\n当前 moomoo 账户没有持仓。"

    lines = ["📒 **实际持仓**", ""]
    total_value = sum(pos.market_value for pos in positions)
    total_pl = sum(pos.pl_value for pos in positions)
    for pos in sorted(positions, key=lambda p: abs(p.market_value), reverse=True):
        pl_sign = "+" if pos.pl_value >= 0 else ""
        ratio_sign = "+" if pos.pl_ratio >= 0 else ""
        lines.append(
            f"**{pos.ticker}** `{pos.side}`  qty `{pos.qty:g}`  "
            f"现价 `${pos.price:,.2f}`  成本 `${pos.cost:,.2f}`  "
            f"市值 `${pos.market_value:,.0f}`  "
            f"盈亏 `{pl_sign}${pos.pl_value:,.0f}` (`{ratio_sign}{pos.pl_ratio:.2f}%`)"
        )
    lines.append("")
    lines.append(f"-# 总市值 `${total_value:,.0f}`  ·  未实现盈亏 `${total_pl:,.0f}`")
    return "\n".join(lines)
