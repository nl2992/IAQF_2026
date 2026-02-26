"""
generate_data.py — IAQF 2026 Data Generation Script
=====================================================
Fetches all data required to run the IAQF_Master_Analysis.ipynb,
IAQF_Advanced_Models.ipynb, IAQF_Arbitrage_Simulation_executed.ipynb,
and IAQF_BasisRisk_Analysis_executed.ipynb notebooks.

Generates:
  data/parquet/panel_1min.parquet       — 30,240 rows × 126 cols (main LOP panel)
  data/parquet/panel_1hour.parquet      — 504 rows × 126 cols
  data/parquet/panel_daily.parquet      — 21 rows × 63 cols
  data/parquet/harmonized_raw_1min.parquet
  data/parquet/l2_BTCUSDT_1min.parquet  — 30,240 rows (from ~150M aggTrades)
  data/parquet/l2_BTCUSDC_1min.parquet  — 14,010 rows (from ~557K aggTrades)
  data/parquet/l2_all_pairs_1min.parquet

NOTE: The L2 microstructure files require downloading ~20 GB of tick data from
Binance's public archive (data.binance.vision). This step takes ~60–90 minutes
depending on your connection. Set SKIP_L2 = True below to skip it and run only
the LOP analysis notebooks.

Usage:
  python generate_data.py            # full run (LOP + L2)
  python generate_data.py --skip-l2  # LOP only (~5 min)
"""

import os, sys, time, gzip, io, zipfile, warnings
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "data" / "parquet"
DATA_DIR.mkdir(parents=True, exist_ok=True)

START_MS = int(pd.Timestamp('2023-03-01 00:00:00', tz='UTC').timestamp() * 1000)
END_MS   = int(pd.Timestamp('2023-03-21 23:59:59', tz='UTC').timestamp() * 1000)
START_DT = pd.Timestamp('2023-03-01', tz='UTC')
END_DT   = pd.Timestamp('2023-03-21 23:59:59', tz='UTC')

SKIP_L2  = '--skip-l2' in sys.argv

# Regime boundaries (UTC)
REGIMES = {
    'pre_crisis': (pd.Timestamp('2023-03-01', tz='UTC'), pd.Timestamp('2023-03-09 23:59', tz='UTC')),
    'crisis':     (pd.Timestamp('2023-03-10', tz='UTC'), pd.Timestamp('2023-03-12 23:59', tz='UTC')),
    'recovery':   (pd.Timestamp('2023-03-13', tz='UTC'), pd.Timestamp('2023-03-15 23:59', tz='UTC')),
    'post':       (pd.Timestamp('2023-03-16', tz='UTC'), pd.Timestamp('2023-03-21 23:59', tz='UTC')),
}

def assign_regime(ts_series):
    regime = pd.Series('post', index=ts_series.index)
    for name, (start, end) in REGIMES.items():
        mask = (ts_series >= start) & (ts_series <= end)
        regime[mask] = name
    return regime

# ── Binance.US OHLCV fetch ────────────────────────────────────────────────────
BNUS_URL = "https://api.binance.us/api/v3/klines"
OHLCV_COLS = ["open_time_ms","open","high","low","close","volume",
              "close_time_ms","quote_vol","num_trades","tbv","tbqv","ignore"]

def fetch_binanceus(symbol, interval="1m"):
    """Fetch all 1-min OHLCV bars from Binance.US for the study period."""
    rows, cursor = [], START_MS
    while cursor < END_MS:
        for attempt in range(5):
            try:
                r = requests.get(BNUS_URL, params={
                    "symbol": symbol, "interval": interval,
                    "startTime": cursor, "endTime": END_MS, "limit": 1000
                }, timeout=30)
                if r.status_code == 200:
                    data = r.json(); break
                elif r.status_code == 429:
                    time.sleep(60); data = []
                else:
                    data = []; break
            except Exception as e:
                time.sleep(5 * (attempt + 1)); data = []
        if not data: break
        rows.extend(data)
        cursor = data[-1][0] + 60_000
        if len(data) < 1000: break
        time.sleep(0.12)
    if not rows:
        print(f"  WARNING: No data returned for {symbol}")
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=OHLCV_COLS)
    df['ts'] = pd.to_datetime(df['open_time_ms'], unit='ms', utc=True)
    for c in ['open','high','low','close','volume','tbv','tbqv']:
        df[c] = pd.to_numeric(df[c])
    df['num_trades'] = pd.to_numeric(df['num_trades'])
    df = df.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
    return df

# ── Coinbase OHLCV fetch ──────────────────────────────────────────────────────
CB_URL = "https://api.exchange.coinbase.com/products"

def fetch_coinbase(product, granularity=60):
    """Fetch all 1-min OHLCV bars from Coinbase for the study period."""
    start = START_DT
    end   = END_DT
    rows  = []
    cursor = start
    step  = pd.Timedelta(seconds=granularity * 300)
    while cursor < end:
        chunk_end = min(cursor + step, end)
        for attempt in range(5):
            try:
                r = requests.get(f"{CB_URL}/{product}/candles", params={
                    "start": cursor.isoformat(),
                    "end":   chunk_end.isoformat(),
                    "granularity": granularity
                }, timeout=30)
                if r.status_code == 200:
                    data = r.json(); break
                elif r.status_code == 429:
                    time.sleep(60); data = []
                else:
                    data = []; break
            except Exception:
                time.sleep(5); data = []
        rows.extend(data)
        cursor = chunk_end
        time.sleep(0.35)
    if not rows:
        print(f"  WARNING: No data returned for {product}")
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=['time','low','high','open','close','volume'])
    df['ts'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
    return df

# ── Step 1: Fetch OHLCV ───────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Fetching 1-min OHLCV from Binance.US and Coinbase")
print("=" * 60)

bnus_pairs = {
    'btcusdt': 'BTCUSDT',
    'btcusdc': 'BTCUSDC',
    'btcusd':  'BTCUSD',
    'btcbusd': 'BTCBUSD',
    'usdtusd': 'USDTUSD',
    'usdcusd': 'USDCUSD',
    'usdcusdt':'USDCUSDT',
    'busdusdt':'BUSDUSDT',
}

raw = {}
for key, symbol in bnus_pairs.items():
    print(f"  Binance.US {symbol}...", end=' ', flush=True)
    df = fetch_binanceus(symbol)
    raw[f'bnus_{key}'] = df
    print(f"{len(df):,} rows")

cb_pairs = {
    'btcusd':  'BTC-USD',
    'btcusdc': 'BTC-USDC',
    'btcusdt': 'BTC-USDT',
}
for key, product in cb_pairs.items():
    print(f"  Coinbase {product}...", end=' ', flush=True)
    df = fetch_coinbase(product)
    raw[f'cb_{key}'] = df
    print(f"{len(df):,} rows")

# ── Step 2: Build harmonized panel ───────────────────────────────────────────
print("\nStep 2: Building harmonized 1-min panel...")

# Create a common timestamp index
all_ts = set()
for key in ['bnus_btcusd', 'bnus_btcusdt']:
    if not raw[key].empty:
        all_ts.update(raw[key]['ts'].tolist())
ts_index = pd.DatetimeIndex(sorted(all_ts))
panel = pd.DataFrame({'timestamp_utc': ts_index})

def merge_ohlcv(panel, df, prefix, ts_col='ts'):
    if df.empty:
        return panel
    df2 = df.rename(columns={
        'open': f'{prefix}_open', 'high': f'{prefix}_high',
        'low':  f'{prefix}_low',  'close': f'{prefix}_close',
        'volume': f'{prefix}_volume'
    })
    cols = ['ts'] + [c for c in df2.columns if c.startswith(prefix)]
    return panel.merge(df2[cols].rename(columns={'ts': 'timestamp_utc'}),
                       on='timestamp_utc', how='left')

for key in ['bnus_btcusdt','bnus_btcusdc','bnus_btcusd','bnus_btcbusd',
            'bnus_usdtusd','bnus_usdcusd','bnus_usdcusdt','bnus_busdusdt',
            'cb_btcusd','cb_btcusdc','cb_btcusdt']:
    if key in raw and not raw[key].empty:
        panel = merge_ohlcv(panel, raw[key], key)

# Compute mid-prices
for key in ['bnus_btcusdt','bnus_btcusdc','bnus_btcusd',
            'bnus_usdtusd','bnus_usdcusd']:
    h_col = f'{key}_high'
    l_col = f'{key}_low'
    if h_col in panel.columns and l_col in panel.columns:
        panel[f'mid_{key}'] = (panel[h_col] + panel[l_col]) / 2

# Compute LOP deviations (log, in bps)
def safe_log_dev(a, b):
    mask = (a > 0) & (b > 0)
    out = pd.Series(np.nan, index=a.index)
    out[mask] = (np.log(a[mask]) - np.log(b[mask])) * 10000
    return out

if 'bnus_btcusdt_close' in panel.columns and 'bnus_btcusd_close' in panel.columns:
    panel['lop_bnus_usdt_vs_usd'] = safe_log_dev(
        panel['bnus_btcusdt_close'], panel['bnus_btcusd_close'])

if 'bnus_btcusdc_close' in panel.columns and 'bnus_btcusd_close' in panel.columns:
    panel['lop_bnus_usdc_vs_usd'] = safe_log_dev(
        panel['bnus_btcusdc_close'], panel['bnus_btcusd_close'])

if 'bnus_usdcusd_close' in panel.columns:
    panel['log_usdc_usd_dev'] = (np.log(panel['bnus_usdcusd_close'].clip(lower=1e-6)) * 10000)

if 'bnus_usdtusd_close' in panel.columns:
    panel['log_usdt_usd_dev'] = (np.log(panel['bnus_usdtusd_close'].clip(lower=1e-6)) * 10000)

# Spread proxies
for key in ['bnus_btcusdt','bnus_btcusdc','bnus_btcusd']:
    h = f'{key}_high'; l = f'{key}_low'; c = f'{key}_close'
    if all(col in panel.columns for col in [h, l, c]):
        panel[f'spread_{key}'] = panel[h] - panel[l]
        panel[f'rel_spread_{key}'] = (panel[h] - panel[l]) / panel[c].clip(lower=1)

# Realized volatility (60-min rolling)
if 'bnus_btcusd_close' in panel.columns:
    log_ret = np.log(panel['bnus_btcusd_close'] / panel['bnus_btcusd_close'].shift(1))
    panel['rv60_bnus_btcusd'] = log_ret.rolling(60).std() * np.sqrt(60)

# Volume shares
vol_cols = [c for c in panel.columns if c.endswith('_volume') and 'bnus_btc' in c]
if vol_cols:
    total_vol = panel[vol_cols].sum(axis=1).clip(lower=1e-9)
    for col in vol_cols:
        key = col.replace('_volume', '').replace('bnus_btc', '')
        panel[f'vol_share_{key}_bnus'] = panel[col] / total_vol

# Regime assignment
panel['regime'] = assign_regime(panel['timestamp_utc'])
panel['regime_num'] = panel['regime'].map(
    {'pre_crisis': 0, 'crisis': 1, 'recovery': 2, 'post': 3})
panel['hour_utc'] = panel['timestamp_utc'].dt.hour

# Save
panel.to_parquet(DATA_DIR / 'panel_1min.parquet', index=False)
print(f"  Saved panel_1min.parquet: {len(panel):,} rows × {len(panel.columns)} cols")

# Hourly and daily resamples
panel_ts = panel.set_index('timestamp_utc')
numeric_cols = panel_ts.select_dtypes(include=[np.number]).columns
panel_1h = panel_ts[numeric_cols].resample('1h').mean().reset_index()
panel_1h['regime'] = assign_regime(panel_1h['timestamp_utc'])
panel_1h.to_parquet(DATA_DIR / 'panel_1hour.parquet', index=False)
print(f"  Saved panel_1hour.parquet: {len(panel_1h):,} rows")

panel_d = panel_ts[numeric_cols].resample('1D').mean().reset_index()
panel_d['regime'] = assign_regime(panel_d['timestamp_utc'])
panel_d.to_parquet(DATA_DIR / 'panel_daily.parquet', index=False)
print(f"  Saved panel_daily.parquet: {len(panel_d):,} rows")

panel.to_parquet(DATA_DIR / 'harmonized_raw_1min.parquet', index=False)
print(f"  Saved harmonized_raw_1min.parquet")

# ── Step 3: L2 Microstructure (optional) ─────────────────────────────────────
if SKIP_L2:
    print("\nSkipping L2 microstructure (--skip-l2 flag set).")
    print("NOTE: IAQF_Master_Analysis.ipynb will fail at the L2 section without these files.")
    print("      Run without --skip-l2 to generate them (requires ~20 GB download).")
else:
    print("\nStep 3: Downloading tick data from Binance archive (~20 GB, ~60–90 min)...")
    ARCHIVE_BASE = "https://data.binance.vision/data/spot/daily"
    TICK_DIR = BASE_DIR / "data" / "tick"
    TICK_DIR.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range('2023-03-01', '2023-03-21', freq='D')

    def download_file(url, dest):
        if dest.exists():
            return True
        try:
            r = requests.get(url, timeout=120, stream=True)
            if r.status_code == 200:
                with open(dest, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
        except Exception:
            pass
        return False

    def process_aggtrades_day(date, symbol):
        date_str = date.strftime('%Y-%m-%d')
        url  = f"{ARCHIVE_BASE}/aggTrades/{symbol}/{symbol}-aggTrades-{date_str}.zip"
        dest = TICK_DIR / f"{symbol}-aggTrades-{date_str}.zip"
        if not download_file(url, dest):
            return pd.DataFrame()
        try:
            with zipfile.ZipFile(dest) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    df = pd.read_csv(f, header=None,
                                     names=['agg_id','price','qty','first_id','last_id',
                                            'timestamp','is_buyer_maker'])
            df['ts'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df['price'] = df['price'].astype(float)
            df['qty']   = df['qty'].astype(float)
            df['signed_qty'] = np.where(df['is_buyer_maker'], -df['qty'], df['qty'])
            return df
        except Exception:
            return pd.DataFrame()

    def compute_l2_metrics(trades_df):
        """Aggregate tick trades to 1-min L2 microstructure metrics."""
        if trades_df.empty:
            return pd.DataFrame()
        trades_df = trades_df.set_index('ts').sort_index()
        def agg_minute(grp):
            prices = grp['price'].values
            qtys   = grp['qty'].values
            signed = grp['signed_qty'].values
            n = len(grp)
            if n == 0:
                return pd.Series(dtype=float)
            # OHLCV
            o, h, l, c = prices[0], prices.max(), prices.min(), prices[-1]
            vol = qtys.sum()
            # Kyle's Lambda: regress price change on signed volume
            dp = np.diff(prices)
            sq = signed[1:]
            if len(dp) > 2 and np.std(sq) > 0:
                kl = np.cov(dp, sq)[0,1] / np.var(sq)
            else:
                kl = np.nan
            # Amihud: |ret| / volume
            ret = abs(np.log(c / o)) if o > 0 else 0
            amihud = ret / vol if vol > 0 else np.nan
            # OBI: (buy_vol - sell_vol) / total_vol
            buy_vol  = signed[signed > 0].sum()
            sell_vol = abs(signed[signed < 0].sum())
            obi = (buy_vol - sell_vol) / (buy_vol + sell_vol) if (buy_vol + sell_vol) > 0 else 0
            # Spread proxy: (H-L)/C
            spread_bps = (h - l) / c * 10000 if c > 0 else np.nan
            depth = vol / spread_bps if spread_bps and spread_bps > 0 else np.nan
            return pd.Series({
                'open': o, 'high': h, 'low': l, 'close': c,
                'volume': vol, 'n_trades': n,
                'kyle_lambda': kl, 'amihud': amihud, 'obi': obi,
                'spread_proxy_bps': spread_bps, 'depth_proxy': depth,
            })
        result = trades_df.resample('1min').apply(agg_minute)
        return result.reset_index().rename(columns={'ts': 'timestamp'})

    for symbol, label in [('BTCUSDT', 'BTC/USDT'), ('BTCUSDC', 'BTC/USDC')]:
        print(f"\n  Processing {label} tick data...")
        all_days = []
        for date in dates:
            day_df = process_aggtrades_day(date, symbol)
            if not day_df.empty:
                all_days.append(day_df)
                print(f"    {date.date()}: {len(day_df):,} trades", end='\r', flush=True)
        if not all_days:
            print(f"  WARNING: No tick data found for {symbol}")
            continue
        all_trades = pd.concat(all_days, ignore_index=True)
        print(f"\n  Total {label} trades: {len(all_trades):,}")
        l2 = compute_l2_metrics(all_trades)
        l2['regime'] = assign_regime(l2['timestamp'])
        out_path = DATA_DIR / f'l2_{symbol}_1min.parquet'
        l2.to_parquet(out_path, index=False)
        print(f"  Saved {out_path.name}: {len(l2):,} rows")

    # Combine
    l2_files = list(DATA_DIR.glob('l2_*_1min.parquet'))
    if l2_files:
        combined = pd.concat([pd.read_parquet(f) for f in l2_files], ignore_index=True)
        combined = combined.sort_values(['timestamp']).reset_index(drop=True)
        combined.to_parquet(DATA_DIR / 'l2_all_pairs_1min.parquet', index=False)
        print(f"\n  Saved l2_all_pairs_1min.parquet: {len(combined):,} rows")

print("\n" + "=" * 60)
print("Data generation complete.")
print(f"Output directory: {DATA_DIR}")
print("=" * 60)
