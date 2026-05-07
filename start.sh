#!/usr/bin/env bash
# Kitty startup script
# Usage:
#   bash start.sh bot       — start the Discord bot
#   bash start.sh pipeline  — rebuild data & retrain models for all tickers
#   bash start.sh upload    — upload raw data and trained models to Hugging Face
#   bash start.sh download  — download raw data and trained models from Hugging Face
#   bash start.sh pipeline SPY SNDK   — rebuild specific tickers only

set -e
cd "$(dirname "$0")"

# ── check .env ────────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
  echo "ERROR: .env not found. Copy .env.example and fill in your tokens."
  exit 1
fi

# source .env so we can validate keys
set -a; source .env; set +a

if [ -z "$DISCORD_TOKEN" ] || [ -z "$POLYGON_API_KEY" ]; then
  echo "ERROR: DISCORD_TOKEN and POLYGON_API_KEY must be set in .env"
  exit 1
fi

# ── helpers ───────────────────────────────────────────────────────────────────
MODE="${1:-bot}"
shift || true   # remaining args are optional ticker list
DEFAULT_TICKERS="AAPL MSFT NVDA GOOGL AMZN META TSLA WMT MS JPM BE PLTR SPY IBM IWM MU SNDK CRWV NBIS INTC AMD ORCL COIN MSTR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

maybe_download_models() {
  if [ -n "$HF_MODEL_REPO_ID" ] && [ -z "$(ls models/saved/*.pkl 2>/dev/null)" ]; then
    log "No local trained models found; downloading from Hugging Face ..."
    python3 scripts/download_from_hf.py --models-only
  fi
}

maybe_download_raw() {
  if [ -n "$HF_RAW_REPO_ID" ] && [ -z "$(ls data/raw/*.parquet 2>/dev/null)" ]; then
    log "No local raw data found; downloading from Hugging Face ..."
    python3 scripts/download_from_hf.py --raw-only
  fi
}

# ── modes ─────────────────────────────────────────────────────────────────────
case "$MODE" in

  bot)
    log "Starting Kitty Discord bot ..."
    maybe_download_models
    # check models exist
    if [ -z "$(ls models/saved/*.pkl 2>/dev/null)" ]; then
      echo "WARNING: No trained models found in models/saved/."
      echo "         Run 'bash start.sh pipeline' first to build data and train models."
    fi
    exec python3 bot.py
    ;;

  pipeline)
    maybe_download_raw
    TICKERS="${*:-$DEFAULT_TICKERS}"
    log "Running pipeline for: $TICKERS"
    python3 scripts/run_pipeline.py --tickers $TICKERS
    log "Pipeline complete. Start the bot with: bash start.sh bot"
    ;;

  upload)
    log "Uploading Kitty artifacts to Hugging Face ..."
    python3 scripts/upload_to_hf.py "$@"
    ;;

  download)
    log "Downloading Kitty artifacts from Hugging Face ..."
    python3 scripts/download_from_hf.py "$@"
    ;;

  *)
    echo "Usage: bash start.sh [bot|pipeline|upload|download] [ARGS ...]"
    echo ""
    echo "  bot                  Start the Discord bot (default)"
    echo "  pipeline             Fetch data + train models for the default ticker basket"
    echo "  pipeline SPY SNDK    Fetch data + train models for specific tickers"
    echo "  upload               Upload data/raw and/or models/saved to Hugging Face"
    echo "  download             Download data/raw and/or models/saved from Hugging Face"
    exit 1
    ;;

esac
