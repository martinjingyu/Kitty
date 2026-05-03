"""
Backtest: Long / Short / Iron Condor

Two modes:

MODE: log (default)
  Decision per sample, P&L = actual log returns:
    P(long)  > thr_long   → LONG    pnl = +actual_log_ret
    P(short) > thr_short  → SHORT   pnl = -actual_log_ret
    P(neutral)> thr_condor → CONDOR  pnl = +credit if label==0 else -loss
    else                   → NO_TRADE pnl = 0

MODE: options
  Same decision, P&L = Black-Scholes credit spread pricing with 10% capital/trade:
    LONG   → Sell put  credit spread  (short put @ lower barrier, long put @ 2× barrier)
    SHORT  → Sell call credit spread  (short call @ upper barrier, long call @ 2× barrier)
    CONDOR → Sell iron condor         (put spread + call spread)
    NO_TRADE → skip

  Capital: starts at --capital (default $100k).
  Each trade: allocate 10% of current capital as margin for max_loss.
    Win → +credit/max_loss × allocated   (ROM = return on margin)
    Lose → −1 × allocated                (full margin lost)

NOTE: models include 'er' (Efficiency Ratio) which leaks future price info.
Performance numbers are optimistic; retrain after removing 'er' from META_COLS.
"""
import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR    = Path("data/dataset")
MODEL_DIR   = Path("models/saved")
TRAIN_RATIO = 0.7

INITIAL_CAPITAL = 100_000.0   # $ starting capital for options mode
ALLOC_FRAC      = 0.10        # fraction of capital per trade
RISK_FREE       = 0.05        # annual risk-free rate

# Position-sizing model:
#   "fixed"    → each trade uses ALLOC_FRAC × initial_capital (constant dollar risk)
#                realistic for concurrent positions (multiple open at once)
#   "compound" → each trade uses ALLOC_FRAC × current_capital (Kelly-style)
#                realistic only if positions are strictly sequential
DEFAULT_SIZING  = "fixed"


# ── Black-Scholes helpers ──────────────────────────────────────────────────────

def _bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T < 1e-8 or sigma < 1e-8 or K <= 0:
        return float(max(K - S, 0.0))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def _bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T < 1e-8 or sigma < 1e-8:
        return float(max(S - K, 0.0))
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _options_rom(
    action:   str,
    label:    int,
    h:        float,
    vol:      float,
    stride:   int,
    max_hold: int,
    r:        float = RISK_FREE,
) -> tuple[float, float, float]:
    """Return (rom, credit_frac, max_loss_frac) for a credit-spread strategy.

    Spreads use barrier width as the spread width (wing = h * vol).
      LONG   → put  credit spread: sell put @ S*(1-h*v),  buy put @ S*(1-2*h*v)
      SHORT  → call credit spread: sell call @ S*(1+h*v), buy call @ S*(1+2*h*v)
      CONDOR → iron condor = put spread + call spread

    Win / Lose (based on triple-barrier label, not just terminal price):
      LONG:   win if label != -1  (lower barrier not hit during max_hold)
      SHORT:  win if label !=  1  (upper barrier not hit during max_hold)
      CONDOR: win if label ==  0  (neither barrier hit)

    rom = +credit/max_loss (win)  or  -1.0 (lose)
    """
    if action == "NO_TRADE":
        return 0.0, 0.0, 0.0

    bars_per_year = 252 * 20 / stride
    sigma_ann     = vol * np.sqrt(bars_per_year)   # annualised vol
    T             = max_hold / bars_per_year        # option expiry in years
    S             = 1.0
    wing          = h * vol                         # spread width = barrier width

    if action == "LONG":
        K_s    = S * (1.0 - h * vol)
        K_l    = max(S * (1.0 - 2.0 * h * vol), 1e-4)
        credit = _bs_put(S, K_s, T, r, sigma_ann) - _bs_put(S, K_l, T, r, sigma_ann)
        win    = (label != -1)

    elif action == "SHORT":
        K_s    = S * (1.0 + h * vol)
        K_l    = S * (1.0 + 2.0 * h * vol)
        credit = _bs_call(S, K_s, T, r, sigma_ann) - _bs_call(S, K_l, T, r, sigma_ann)
        win    = (label != 1)

    elif action == "CONDOR":
        K_ps = S * (1.0 - h * vol);   K_pl = max(S * (1.0 - 2.0 * h * vol), 1e-4)
        K_cs = S * (1.0 + h * vol);   K_cl = S * (1.0 + 2.0 * h * vol)
        credit = (
            _bs_put (S, K_ps, T, r, sigma_ann) - _bs_put (S, K_pl, T, r, sigma_ann)
            + _bs_call(S, K_cs, T, r, sigma_ann) - _bs_call(S, K_cl, T, r, sigma_ann)
        )
        win = (label == 0)

    else:
        return 0.0, 0.0, 0.0

    credit   = max(credit, 0.0)
    max_loss = max(wing - credit, 1e-10)
    rom      = credit / max_loss if win else -1.0
    return float(rom), float(credit), float(max_loss)


# ── original log-ret helpers ───────────────────────────────────────────────────

def _get_probas(model, X):
    classes   = list(model.classes_)
    proba_all = model.predict_proba(X)
    if len(classes) == 3:
        p_long    = proba_all[:, classes.index(2)]
        p_short   = proba_all[:, classes.index(0)]
        p_neutral = proba_all[:, classes.index(1)]
    else:
        p_long    = proba_all[:, 1]
        p_short   = proba_all[:, 0]
        p_neutral = np.zeros(len(p_long))
    return p_long, p_short, p_neutral


def _sharpe(pnl: pd.Series, periods_per_year: int) -> float:
    if len(pnl) < 2 or pnl.std() == 0:
        return 0.0
    return float(pnl.mean() / pnl.std() * np.sqrt(periods_per_year))


def _cum_ret(pnl: pd.Series) -> float:
    return float((np.exp(pnl.sum()) - 1) * 100)


# ── core backtest ─────────────────────────────────────────────────────────────

def run_regime(
    regime:             str,
    thr_long:           float = 0.63,
    thr_short:          float = 0.50,
    thr_condor:         float = 0.50,
    condor_credit_frac: float = 0.35,
    condor_loss_frac:   float = 0.65,
    use_options:        bool  = False,
    initial_capital:    float = INITIAL_CAPITAL,
    sizing:             str   = DEFAULT_SIZING,   # "fixed" or "compound"
) -> tuple[pd.DataFrame, dict]:
    """
    Decision logic per sample:
      P(long)    > thr_long   → LONG
      P(short)   > thr_short  → SHORT
      P(neutral) > thr_condor → CONDOR
      else                    → NO_TRADE

    log mode:   pnl = log return (or credit/loss fraction for condor)
    options mode: pnl = ROM on 10% capital; capital compounds across trades
    """
    pkl_path = MODEL_DIR / f"multi_xgb_{regime}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Model not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    model     = obj["model"]
    feat_cols = obj["features"]
    cfg       = obj["config"]
    h         = cfg["h"]
    stride    = cfg["stride"]
    max_hold  = cfg["max_hold"]

    ds    = pd.read_parquet(DATA_DIR / f"multi_{regime}_dataset.parquet")
    split = int(len(ds) * TRAIN_RATIO)
    test  = ds.iloc[split:].reset_index(drop=True)

    X     = test[feat_cols].values
    y_raw = test["label"].values   # -1 / 0 / +1
    ret   = test["ret"].values
    vol   = test["vol"].values

    p_long, p_short, p_neutral = _get_probas(model, X)

    capital = initial_capital
    records = []

    for i in range(len(test)):
        pl, ps, pn = p_long[i], p_short[i], p_neutral[i]
        barrier = h * vol[i]

        if pl > thr_long:
            action = "LONG"
        elif ps > thr_short:
            action = "SHORT"
        elif pn > thr_condor:
            action = "CONDOR"
        else:
            action = "NO_TRADE"

        if use_options:
            rom, credit_frac, ml_frac = _options_rom(
                action, y_raw[i], h, vol[i], stride, max_hold
            )
            if action != "NO_TRADE":
                # fixed: always risk 10% of starting capital per trade
                # compound: risk 10% of current capital (sequential positions only)
                alloc      = ALLOC_FRAC * (initial_capital if sizing == "fixed" else capital)
                pnl_dollar = alloc * rom
                capital   += pnl_dollar
            else:
                rom        = 0.0
                pnl_dollar = 0.0

            records.append({
                "ticker":       test["ticker"].iloc[i],
                "timestamp":    test["timestamp"].iloc[i],
                "action":       action,
                "pnl":          rom,          # return on margin (ROM)
                "pnl_dollar":   pnl_dollar,
                "capital":      capital,
                "credit_frac":  credit_frac,
                "ml_frac":      ml_frac,
                "label":        y_raw[i],
                "p_long":       pl,
                "p_short":      ps,
                "p_neutral":    pn,
                "vol":          vol[i],
                "barrier":      barrier,
            })

        else:
            # original log-ret mode
            if action == "LONG":
                pnl = ret[i]
            elif action == "SHORT":
                pnl = -ret[i]
            elif action == "CONDOR":
                pnl = (condor_credit_frac * barrier
                       if y_raw[i] == 0 else -condor_loss_frac * barrier)
            else:
                pnl = 0.0

            records.append({
                "ticker":    test["ticker"].iloc[i],
                "timestamp": test["timestamp"].iloc[i],
                "action":    action,
                "pnl":       pnl,
                "label":     y_raw[i],
                "p_long":    pl,
                "p_short":   ps,
                "p_neutral": pn,
                "vol":       vol[i],
                "barrier":   barrier,
            })

    return pd.DataFrame(records), cfg


# ── reporting ─────────────────────────────────────────────────────────────────

def _action_block(sub: pd.DataFrame, total_n: int, periods_per_year: int, label: str):
    if len(sub) == 0:
        return
    pnl = sub["pnl"]
    win_rate = (pnl > 0).mean()
    shr      = _sharpe(pnl, periods_per_year)
    cum      = _cum_ret(pnl)
    avg_pnl  = pnl.mean() * 100
    print(f"  {label:<8}  n={len(sub):>6,} ({len(sub)/total_n:>5.1%})"
          f"  win={win_rate:.1%}"
          f"  avg={avg_pnl:>+6.3f}%"
          f"  sharpe={shr:>7.3f}"
          f"  cum={cum:>9.2f}%")


def report(df: pd.DataFrame, cfg: dict, regime: str,
           thr_long: float, thr_short: float, thr_condor: float,
           condor_credit_frac: float, condor_loss_frac: float):
    stride           = cfg["stride"]
    periods_per_year = int(252 * 20 / stride)
    n                = len(df)

    print(f"\n{'█'*60}")
    print(f"  REGIME: {regime.upper()}  "
          f"(h={cfg['h']}  max_hold={cfg['max_hold']} bars)")
    print(f"  thr_long={thr_long}  thr_short={thr_short}  thr_condor={thr_condor}")
    print(f"  condor credit={condor_credit_frac:.0%}  loss={condor_loss_frac:.0%} of barrier")
    print(f"  Test period: "
          f"{pd.to_datetime(df['timestamp'].min()).date()} → "
          f"{pd.to_datetime(df['timestamp'].max()).date()}")
    print(f"  Test samples: {n:,}  |  "
          f"tickers: {sorted(df['ticker'].unique())}")
    print(f"{'─'*60}")

    traded   = df[df["action"] != "NO_TRADE"]
    n_traded = max(len(traded), 1)

    for action in ["LONG", "SHORT", "CONDOR"]:
        sub = df[df["action"] == action]
        _action_block(sub, n, periods_per_year, action)

    condor = df[df["action"] == "CONDOR"]
    if len(condor) > 0:
        n_win  = (condor["label"] == 0).sum()
        n_lose = (condor["label"] != 0).sum()
        print(f"  {'':8}  ↳ label=0 (win): {n_win:,}"
              f"  label≠0 (loss): {n_lose:,}"
              f"  win_rate={n_win/len(condor):.1%}")

    no_trade = (df["action"] == "NO_TRADE").sum()
    print(f"  NO_TRADE  n={no_trade:>6,} ({no_trade/n:>5.1%})")

    print(f"{'─'*60}")
    _action_block(traded, n_traded, periods_per_year, "TRADED")

    df_sorted = traded.sort_values("timestamp")
    if len(df_sorted) > 1:
        cum_pnl = df_sorted["pnl"].cumsum()
        max_dd  = (cum_pnl - cum_pnl.cummax()).min() * 100
        print(f"  {'':8}  max_drawdown={max_dd:.2f}%")

    print(f"\n  Per-ticker (traded only):")
    for ticker, grp in traded.groupby("ticker"):
        pnl = grp["pnl"]
        print(f"    {ticker:<6}  n={len(grp):>4,}"
              f"  cum={_cum_ret(pnl):>8.2f}%"
              f"  sharpe={_sharpe(pnl, periods_per_year):>6.3f}"
              f"  long={(grp['action']=='LONG').sum():>4,}"
              f"  short={(grp['action']=='SHORT').sum():>4,}"
              f"  condor={(grp['action']=='CONDOR').sum():>4,}")


def report_options(df: pd.DataFrame, cfg: dict, regime: str,
                   thr_long: float, thr_short: float, thr_condor: float,
                   initial_capital: float, sizing: str = DEFAULT_SIZING):
    """Report for options mode: capital curve, ROM stats, dollar P&L."""
    stride           = cfg["stride"]
    periods_per_year = int(252 * 20 / stride)
    bars_per_year    = 252 * 20 / stride
    T_years          = cfg["max_hold"] / bars_per_year
    n                = len(df)

    print(f"\n{'█'*60}")
    print(f"  REGIME: {regime.upper()}  [OPTIONS MODE]")
    print(f"  h={cfg['h']}  max_hold={cfg['max_hold']} bars"
          f"  ≈ {T_years*365:.0f} cal-days per trade")
    print(f"  thr_long={thr_long}  thr_short={thr_short}  thr_condor={thr_condor}")
    alloc_dollar = ALLOC_FRAC * initial_capital
    print(f"  Capital: ${initial_capital:,.0f}  |  Allocation: {ALLOC_FRAC:.0%}/trade"
          f"  (${alloc_dollar:,.0f} fixed per trade)  sizing={sizing}")
    print(f"  Test period: "
          f"{pd.to_datetime(df['timestamp'].min()).date()} → "
          f"{pd.to_datetime(df['timestamp'].max()).date()}")
    print(f"  Test samples: {n:,}  |  tickers: {sorted(df['ticker'].unique())}")
    print(f"{'─'*60}")

    traded   = df[df["action"] != "NO_TRADE"]
    no_trade = (df["action"] == "NO_TRADE").sum()

    if len(traded) == 0:
        print("  No trades executed.")
        return

    # ── per-strategy block ────────────────────────────────────────────────────
    strategy_map = {
        "LONG":   ("Sell Put  Spread", "label != -1"),
        "SHORT":  ("Sell Call Spread", "label != +1"),
        "CONDOR": ("Iron Condor     ", "label ==  0"),
    }
    for action, (strategy_name, win_cond) in strategy_map.items():
        sub = traded[traded["action"] == action]
        if len(sub) == 0:
            continue
        win_rate  = (sub["pnl"] > 0).mean()
        avg_credit = sub["credit_frac"].mean() * 100
        avg_ml     = sub["ml_frac"].mean() * 100
        avg_rom    = sub["pnl"].mean() * 100
        shr        = _sharpe(sub["pnl"], periods_per_year)
        pnl_dollar = sub["pnl_dollar"].sum()
        print(f"  {action:<7} [{strategy_name}]  n={len(sub):>5,}"
              f"  win={win_rate:.1%}"
              f"  avg_credit={avg_credit:>5.3f}%"
              f"  avg_maxloss={avg_ml:>5.3f}%"
              f"  avg_ROM={avg_rom:>+7.2f}%"
              f"  sharpe={shr:>7.3f}"
              f"  P&L=${pnl_dollar:>+10,.0f}")

        if action == "CONDOR":
            n_win  = (sub["label"] == 0).sum()
            n_lose = (sub["label"] != 0).sum()
            print(f"  {'':7}   ↳ label=0 (win): {n_win:,}"
                  f"  label≠0 (loss): {n_lose:,}"
                  f"  win_rate={n_win/len(sub):.1%}")

    print(f"  NO_TRADE  n={no_trade:>6,} ({no_trade/n:>5.1%})")
    print(f"{'─'*60}")

    # ── overall capital curve ─────────────────────────────────────────────────
    final_capital = df["capital"].iloc[-1]
    total_return  = (final_capital / initial_capital - 1) * 100
    total_pnl     = traded["pnl_dollar"].sum()

    # Sharpe on ROM series (traded only, sorted by time)
    traded_sorted = traded.sort_values("timestamp")
    rom_series    = traded_sorted["pnl"]
    shr_total     = _sharpe(rom_series, periods_per_year)

    # max drawdown on capital curve
    cap_curve = df.sort_values("timestamp")["capital"]
    max_dd_pct = ((cap_curve - cap_curve.cummax()) / cap_curve.cummax()).min() * 100

    print(f"  OVERALL   n={len(traded):>6,} trades")
    print(f"            Final capital : ${final_capital:>12,.2f}")
    print(f"            Total P&L     : ${total_pnl:>+12,.2f}")
    print(f"            Total return  : {total_return:>+9.2f}%")
    print(f"            Sharpe (ROM)  : {shr_total:>9.3f}")
    print(f"            Max drawdown  : {max_dd_pct:>9.2f}%")

    # ── per-ticker breakdown ──────────────────────────────────────────────────
    print(f"\n  Per-ticker (traded only):")
    for ticker, grp in traded.groupby("ticker"):
        pnl_d  = grp["pnl_dollar"].sum()
        shr    = _sharpe(grp["pnl"], periods_per_year)
        wr     = (grp["pnl"] > 0).mean()
        print(f"    {ticker:<6}  n={len(grp):>4,}"
              f"  P&L=${pnl_d:>+10,.0f}"
              f"  win={wr:.1%}"
              f"  sharpe={shr:>6.3f}"
              f"  long={(grp['action']=='LONG').sum():>4,}"
              f"  short={(grp['action']=='SHORT').sum():>4,}"
              f"  condor={(grp['action']=='CONDOR').sum():>4,}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", nargs="+", default=["intraday", "weekly"],
                        choices=["intraday", "weekly", "monthly"])
    parser.add_argument("--mode",        choices=["log", "options"], default="log",
                        help="log = original log-return P&L; options = BS credit spreads")
    parser.add_argument("--capital",     type=float, default=INITIAL_CAPITAL,
                        help="Starting capital for options mode (default $100,000)")
    parser.add_argument("--thr-long",    type=float, default=0.63)
    parser.add_argument("--thr-short",   type=float, default=0.50)
    parser.add_argument("--thr-condor",  type=float, default=0.50)
    parser.add_argument("--sizing",        choices=["fixed", "compound"], default="fixed",
                        help="fixed=constant $10k/trade (concurrent); compound=Kelly-style (sequential)")
    parser.add_argument("--condor-credit", type=float, default=0.35,
                        help="(log mode only) condor credit as fraction of barrier")
    parser.add_argument("--condor-loss",   type=float, default=0.65,
                        help="(log mode only) condor max loss as fraction of barrier")
    args = parser.parse_args()

    use_options = (args.mode == "options")

    for regime in args.regime:
        try:
            df, cfg = run_regime(
                regime,
                thr_long=args.thr_long,
                thr_short=args.thr_short,
                thr_condor=args.thr_condor,
                condor_credit_frac=args.condor_credit,
                condor_loss_frac=args.condor_loss,
                use_options=use_options,
                initial_capital=args.capital,
                sizing=args.sizing,
            )
            if use_options:
                report_options(df, cfg, regime, args.thr_long, args.thr_short,
                               args.thr_condor, args.capital, args.sizing)
            else:
                report(df, cfg, regime, args.thr_long, args.thr_short, args.thr_condor,
                       args.condor_credit, args.condor_loss)
        except FileNotFoundError as e:
            print(f"\n  [{regime}] {e} — skipping")


if __name__ == "__main__":
    main()
