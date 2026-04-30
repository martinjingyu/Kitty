"""
Fetch SPY minute-level aggregate bars from Polygon.io and cache locally as parquet.
Free tier: 5 requests/min — each page fetch sleeps 13s to stay within limit.
"""
import os
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY")
RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(exist_ok=True)

TICKER = "SPY"
BASE_URL = "https://api.polygon.io"
RATE_LIMIT_SLEEP = 13  # seconds between requests (free tier: 5 req/min)


def _get(url: str, params: dict, retries: int = 5) -> dict:
    params = dict(params)
    params["apiKey"] = API_KEY
    for attempt in range(retries):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            wait = 60 * (attempt + 1)
            print(f"  Rate limited — waiting {wait}s ...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("Max retries exceeded on rate limit")


def fetch_minute_aggs(
    ticker: str = TICKER,
    start: str = "2021-04-29",
    end: str = "2026-04-29",
) -> pd.DataFrame:
    """
    Fetch 1-minute OHLCV aggregates from Polygon.io with auto-pagination.
    Results are cached as parquet — subsequent calls return instantly.
    """
    cache_path = RAW_DIR / f"{ticker}_1m_{start}_{end}.parquet"
    if cache_path.exists():
        print(f"Loading from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Fetching {ticker} 1-minute bars {start} → {end} from Polygon.io ...")
    print(f"(Free tier: ~13s between pages — this may take a few minutes)")
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/minute/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000}

    all_results = []
    page = 0
    next_url = None

    while True:
        data = _get(next_url or url, {} if next_url else params)
        results = data.get("results", [])
        all_results.extend(results)
        page += 1
        print(f"  page {page}: +{len(results):,} rows (total {len(all_results):,})")

        next_url = data.get("next_url")
        if not next_url:
            break
        time.sleep(RATE_LIMIT_SLEEP)

    if not all_results:
        raise RuntimeError("No data returned — check API key or date range")

    df = pd.DataFrame(all_results)
    df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high",
                             "l": "low", "c": "close", "v": "volume", "vw": "vwap"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[["timestamp", "open", "high", "low", "close", "volume", "vwap"]].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    df.to_parquet(cache_path, index=False)
    print(f"Saved {len(df):,} rows → {cache_path}")
    return df


if __name__ == "__main__":
    df = fetch_minute_aggs()
    print(df.head())
    print(f"\nDate range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"Total rows: {len(df):,}")
