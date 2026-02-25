from pathlib import Path
"""
IAQF L2 Microstructure Pipeline v2
=====================================
Processes Binance aggTrades and 1-second klines for:
  - BTCUSDT: March 1-21, 2023 (full period)
  - BTCUSDC: March 12-21, 2023 (listed on Binance on Mar 12 - during crisis)

Metrics computed per 1-minute bar:
  From aggTrades (tick-level):
    - n_trades, total_vol, buy_vol, sell_vol
    - vwap, first_price, last_price
    - signed_vol = sum(sign * qty)  [sign = +1 if taker buy, -1 if taker sell]
    - trade_obi = (buy_vol - sell_vol) / (buy_vol + sell_vol)
    - amihud = |ret_1m| / total_vol
    - kyle_lambda (rolling 60-min OLS: ret ~ signed_vol)
    - slope_proxy = ret / signed_vol

  From 1-second klines:
    - rv_1s = sum of squared 1s log returns (realized variance)
    - parkinson_var = sum of (ln(H/L))^2 / (4*ln2)  (Parkinson estimator)
    - spread_hl_mean = mean(H-L) per minute
    - rel_spread_hl = spread_hl_mean / mid
    - depth_proxy = 1 / rel_spread_hl  (inverse range = thicker book proxy)
    - kline_obi = 2 * taker_buy / total_vol - 1

  Derived:
    - resiliency = minutes to recover 50% of shock (for |ret| > 0.5%)
    - book_slope_proxy = |ret_1m| / |signed_vol|
"""

import os, zipfile, gc
import numpy as np
import pandas as pd
from datetime import date, timedelta
from scipy import stats

RAW_DIR = str(Path(__file__).parent.parent / "data" / "raw")
OUT_DIR = str(Path(__file__).parent.parent / "data" / "parquet")
os.makedirs(OUT_DIR, exist_ok=True)

START = date(2023, 3, 1)
END   = date(2023, 3, 21)
ALL_DATES = [START + timedelta(days=i) for i in range((END - START).days + 1)]

PAIR_DATES = {
    "BTCUSDT": ALL_DATES,                                          # full 21 days
    "BTCUSDC": [d for d in ALL_DATES if d >= date(2023, 3, 12)],  # listed Mar 12
}

REGIME_MAP = {
    range(1,  10): 'pre_crisis',
    range(10, 13): 'crisis',
    range(13, 16): 'recovery',
    range(16, 22): 'post',
}

def get_regime(d):
    day = d.day
    for r, label in REGIME_MAP.items():
        if day in r: return label
    return 'post'

# ─── aggTrades parser ──────────────────────────────────────────────────────────
AGG_COLS = ['agg_id','price','qty','first_trade_id','last_trade_id',
            'timestamp','buyer_maker','best_match']

def parse_agg_trades(pair, d):
    ds   = d.strftime("%Y-%m-%d")
    path = f"{RAW_DIR}/{pair}/aggTrades/{pair}-aggTrades-{ds}.zip"
    if not os.path.exists(path) or os.path.getsize(path) < 50000:
        return None
    try:
        zf = zipfile.ZipFile(path)
        with zf.open(zf.namelist()[0]) as f:
            df = pd.read_csv(
                f, header=None, names=AGG_COLS,
                usecols=['price','qty','timestamp','buyer_maker'],
                dtype={'price':'float32','qty':'float32','buyer_maker':'bool'})
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        # buyer_maker=True → taker is seller → sign = -1
        df['sign'] = np.where(df['buyer_maker'], np.float32(-1.0), np.float32(1.0))
        return df
    except Exception as e:
        print(f"    ERROR parsing {path}: {e}")
        return None

def agg_to_1min(df):
    df = df.set_index('timestamp')
    g  = df.resample('1min')

    out = pd.DataFrame()
    out['vwap']       = g.apply(lambda x: (x['price']*x['qty']).sum()/x['qty'].sum()
                                 if len(x)>0 else np.nan)
    out['total_vol']  = g['qty'].sum()
    out['n_trades']   = g['qty'].count()
    out['buy_vol']    = g.apply(lambda x: x.loc[~x['buyer_maker'],'qty'].sum())
    out['sell_vol']   = g.apply(lambda x: x.loc[ x['buyer_maker'],'qty'].sum())
    out['signed_vol'] = g.apply(lambda x: (x['sign']*x['qty']).sum())
    out['last_price'] = g['price'].last()
    out['first_price']= g['price'].first()

    denom = out['buy_vol'] + out['sell_vol']
    out['trade_obi']  = np.where(denom>0,
                                  (out['buy_vol']-out['sell_vol'])/denom, np.nan)
    out['ret_1m']     = np.log(out['vwap'] / out['vwap'].shift(1))
    out['amihud']     = np.abs(out['ret_1m']) / out['total_vol'].clip(1e-10)
    return out.reset_index()

# ─── 1-second klines parser ───────────────────────────────────────────────────
K1S_COLS = ['open_time','open','high','low','close','volume',
            'close_time','quote_vol','n_trades','taker_buy_base','taker_buy_quote','ignore']

def parse_klines_1s(pair, d):
    ds   = d.strftime("%Y-%m-%d")
    path = f"{RAW_DIR}/{pair}/klines_1s/{pair}-1s-{ds}.zip"
    if not os.path.exists(path) or os.path.getsize(path) < 10000:
        return None
    try:
        zf = zipfile.ZipFile(path)
        with zf.open(zf.namelist()[0]) as f:
            df = pd.read_csv(
                f, header=None, names=K1S_COLS,
                usecols=['open_time','open','high','low','close',
                         'volume','taker_buy_base'],
                dtype={'open':'float32','high':'float32','low':'float32',
                       'close':'float32','volume':'float32','taker_buy_base':'float32'})
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        return df
    except Exception as e:
        print(f"    ERROR parsing {path}: {e}")
        return None

def klines_1s_to_1min(df):
    df = df.set_index('timestamp')
    df['ret_1s']      = np.log(df['close'] / df['close'].shift(1))
    hl                = df['high'] / df['low'].clip(1e-10)
    df['parkinson_s'] = (np.log(hl)**2) / (4*np.log(2))
    df['spread_hl']   = df['high'] - df['low']
    df['mid']         = (df['high'] + df['low']) / 2

    g   = df.resample('1min')
    out = pd.DataFrame()
    out['rv_1s']           = g['ret_1s'].apply(lambda x: (x**2).sum())
    out['parkinson_var']   = g['parkinson_s'].sum()
    out['spread_hl_mean']  = g['spread_hl'].mean()
    out['mid_1m']          = g['mid'].last()
    out['vol_1m_k']        = g['volume'].sum()
    out['taker_buy_1m']    = g['taker_buy_base'].sum()
    out['open_1m']         = g['open'].first()
    out['high_1m']         = g['high'].max()
    out['low_1m']          = g['low'].min()
    out['close_1m']        = g['close'].last()

    out['rel_spread_hl']   = out['spread_hl_mean'] / out['mid_1m'].clip(1e-6)
    out['depth_proxy']     = 1.0 / out['rel_spread_hl'].clip(1e-8)
    out['kline_obi']       = np.where(
        out['vol_1m_k']>0,
        2*out['taker_buy_1m']/out['vol_1m_k'] - 1, np.nan)
    return out.reset_index()

# ─── Kyle Lambda (rolling 60-min OLS) ────────────────────────────────────────
def compute_kyle_lambda(df, window=60):
    lambdas = np.full(len(df), np.nan)
    ret_arr = df['ret_1m'].values
    svol_arr= df['signed_vol'].values
    for i in range(window, len(df)):
        y = ret_arr[i-window:i]
        x = svol_arr[i-window:i]
        mask = np.isfinite(y) & np.isfinite(x) & (np.abs(x) > 1e-8)
        if mask.sum() < 20: continue
        try:
            slope, *_ = stats.linregress(x[mask], y[mask])
            lambdas[i] = slope
        except: pass
    return lambdas

# ─── Resiliency ───────────────────────────────────────────────────────────────
def compute_resiliency(df, shock_pct=0.005, window=30):
    rets  = df['ret_1m'].values
    mids  = df['mid_1m'].values if 'mid_1m' in df.columns else df['vwap'].values
    n     = len(df)
    resil = np.full(n, np.nan)
    for i in range(1, n-window):
        if not np.isfinite(rets[i]) or np.abs(rets[i]) < shock_pct: continue
        if not (np.isfinite(mids[i]) and np.isfinite(mids[i-1])): continue
        shock = np.abs(mids[i] - mids[i-1])
        if shock == 0: continue
        for j in range(i+1, min(i+window, n)):
            if np.isfinite(mids[j]) and np.abs(mids[j]-mids[i]) <= 0.5*shock:
                resil[i] = j - i
                break
    return resil

# ─── Main ─────────────────────────────────────────────────────────────────────
all_dfs = []

for pair, dates in PAIR_DATES.items():
    print(f"\n{'='*60}\nProcessing {pair} ({len(dates)} days)\n{'='*60}")
    daily = []

    for d in dates:
        ds = d.strftime("%Y-%m-%d")
        print(f"  {ds}...", end=' ', flush=True)

        df_ticks = parse_agg_trades(pair, d)
        if df_ticks is None:
            print("SKIP (no aggTrades)")
            continue

        df_min = agg_to_1min(df_ticks)
        del df_ticks; gc.collect()

        df_1s = parse_klines_1s(pair, d)
        if df_1s is not None:
            df_k = klines_1s_to_1min(df_1s)
            del df_1s; gc.collect()
            df_min = pd.merge(df_min, df_k, on='timestamp', how='outer')
        else:
            for col in ['rv_1s','parkinson_var','spread_hl_mean','rel_spread_hl',
                        'depth_proxy','kline_obi','mid_1m','open_1m','high_1m',
                        'low_1m','close_1m']:
                df_min[col] = np.nan
            df_min['mid_1m'] = df_min['vwap']

        df_min['pair']   = pair
        df_min['date']   = d
        df_min['regime'] = get_regime(d)
        daily.append(df_min)
        print(f"OK ({len(df_min)} rows, {df_min['n_trades'].sum():,.0f} trades)")

    if not daily:
        print(f"  No data for {pair}")
        continue

    df_pair = pd.concat(daily, ignore_index=True).sort_values('timestamp').reset_index(drop=True)

    print(f"  Computing Kyle Lambda...")
    df_pair['kyle_lambda'] = compute_kyle_lambda(df_pair)

    print(f"  Computing Resiliency...")
    df_pair['resiliency'] = compute_resiliency(df_pair)

    df_pair['slope_proxy'] = np.where(
        np.abs(df_pair['signed_vol']) > 1e-6,
        df_pair['ret_1m'] / df_pair['signed_vol'], np.nan)

    out = f"{OUT_DIR}/{pair}_l2_1min.parquet"
    df_pair.to_parquet(out, index=False)
    print(f"  Saved: {out} ({len(df_pair):,} rows)")
    all_dfs.append(df_pair)

# ─── Master file ──────────────────────────────────────────────────────────────
if all_dfs:
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_parquet(f"{OUT_DIR}/all_pairs_l2_1min.parquet", index=False)
    print(f"\n✓ Master saved: {len(df_all):,} rows, {len(df_all.columns)} columns")
    print("\nSummary by pair/regime:")
    cols = ['kyle_lambda','amihud','trade_obi','rel_spread_hl','rv_1s']
    avail = [c for c in cols if c in df_all.columns]
    print(df_all.groupby(['pair','regime'])[avail].mean().round(8).to_string())
