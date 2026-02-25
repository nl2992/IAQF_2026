from pathlib import Path
"""
Download Binance aggTrades (tick-level) and 1-second klines for BTCUSDT and BTCBUSD
for March 1–21, 2023 from data.binance.vision
"""
import os, requests, time
from datetime import date, timedelta

BASE = "https://data.binance.vision/data/spot/daily"
OUT  = str(Path(__file__).parent.parent / "data" / "raw")
os.makedirs(OUT, exist_ok=True)

PAIRS = ["BTCUSDT", "BTCBUSD"]
START = date(2023, 3, 1)
END   = date(2023, 3, 21)

dates = [START + timedelta(days=i) for i in range((END - START).days + 1)]

def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 10000:
        print(f"  SKIP (exists): {os.path.basename(dest)}")
        return True
    r = requests.get(url, stream=True, timeout=120)
    if r.status_code != 200:
        print(f"  FAIL {r.status_code}: {url}")
        return False
    with open(dest, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            f.write(chunk)
    size_mb = os.path.getsize(dest) / 1e6
    print(f"  OK  {os.path.basename(dest)} ({size_mb:.1f} MB)")
    return True

total = 0
for pair in PAIRS:
    # aggTrades
    agg_dir = f"{OUT}/{pair}/aggTrades"
    os.makedirs(agg_dir, exist_ok=True)
    # 1s klines
    kline_dir = f"{OUT}/{pair}/klines_1s"
    os.makedirs(kline_dir, exist_ok=True)

    for d in dates:
        ds = d.strftime("%Y-%m-%d")

        # aggTrades
        fname_agg = f"{pair}-aggTrades-{ds}.zip"
        url_agg   = f"{BASE}/aggTrades/{pair}/{fname_agg}"
        dest_agg  = f"{agg_dir}/{fname_agg}"
        print(f"[{pair} aggTrades {ds}]")
        ok = download(url_agg, dest_agg)
        if ok: total += 1

        # 1-second klines
        fname_k = f"{pair}-1s-{ds}.zip"
        url_k   = f"{BASE}/klines/{pair}/1s/{fname_k}"
        dest_k  = f"{kline_dir}/{fname_k}"
        print(f"[{pair} klines_1s {ds}]")
        ok = download(url_k, dest_k)
        if ok: total += 1

        time.sleep(0.1)  # polite rate limiting

print(f"\n✓ Downloaded {total} files total")
