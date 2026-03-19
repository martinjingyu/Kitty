import yfinance as yf
from typing import Any, Dict
from datetime import datetime
from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry

class StockAnalysisTool(Tool):

    @property
    def name(self) -> str:
        return "stock_analysis"

    @property
    def description(self) -> str:
        return (
            "Get stock data including current price, historical OHLC (K-line), "
            "and fundamental analysis using Yahoo Finance."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g. NVDA, AAPL)"
                },
                "mode": {
                    "type": "string",
                    "enum": ["quote", "history", "analysis"],
                    "description": "Type of data to return"
                },
                "period": {
                    "type": "string",
                    "description": "Time period (e.g. 1d, 5d, 1mo, 6mo, 1y)",
                    "default": "1mo"
                },
                "interval": {
                    "type": "string",
                    "description": "Data interval (e.g. 1m, 5m, 1h, 1d)",
                    "default": "1d"
                }
            },
            "required": ["symbol", "mode"]
        }

    async def execute(self, **kwargs: Any) -> str:
        symbol = kwargs.get("symbol")
        mode = kwargs.get("mode")
        period = kwargs.get("period", "1mo")
        interval = kwargs.get("interval", "1d")

        stock = yf.Ticker(symbol)

        if mode == "quote":
            info = stock.info
            return str({
                "price": info.get("currentPrice"),
                "change": info.get("regularMarketChange"),
                "changePercent": info.get("regularMarketChangePercent"),
                "high": info.get("dayHigh"),
                "low": info.get("dayLow"),
                "volume": info.get("volume")
            })

        elif mode == "history":
            hist = stock.history(period=period, interval=interval)
            return hist.tail(20).to_string()

        elif mode == "analysis":
            info = stock.info

            result = {
                "symbol": symbol,
                "price": info.get("currentPrice"),

                # 趋势
                "trend": {
                    "50d_avg": info.get("fiftyDayAverage"),
                    "200d_avg": info.get("twoHundredDayAverage"),
                },

                # 估值
                "valuation": {
                    "pe": info.get("trailingPE"),
                    "forwardPE": info.get("forwardPE"),
                    "priceToBook": info.get("priceToBook"),
                },

                # 成长
                "growth": {
                    "revenueGrowth": info.get("revenueGrowth"),
                    "earningsGrowth": info.get("earningsGrowth"),
                },

                # 盈利能力
                "profitability": {
                    "profitMargins": info.get("profitMargins"),
                    "grossMargins": info.get("grossMargins"),
                },

                # 分析师观点
                "analyst": {
                    "rating": info.get("recommendationKey"),
                    "targetPrice": info.get("targetMeanPrice"),
                }
            }

            return str(result)

        else:
            return "Invalid mode"