import html
import json
import random
import re
import time
import unicodedata
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from email.utils import parsedate_to_datetime
from getpass import getpass
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4, uuid5, NAMESPACE_URL

import numpy as np
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from polysim_sdk import PolySimClient

ASSETS = ("BTC", "ETH", "SOL", "XRP", "DOGE", "BNB", "HYPE")
TARGET_BETS = 5
PER_BET = Decimal("5.00")
RUN_BUDGET = Decimal("25.00")                 # Notional BEFORE fees.
MAX_SHARE_PRICE = Decimal("0.99")            # Replaces old confidence/65c cap.
MAX_ENTRY_MINUTES = 10
PREPARATION_SECONDS = 180                    # Parallel multi-coin preparation.
HISTORY_PAGES = 4
MIN_TRANSITIONS = 1500
STATES = 5
ALPHA = 1.0
NEWS_STRENGTH = 0.5                          # HEURISTIC, not news-trained.
SPOT_KLINES = "https://data-api.binance.vision/api/v3/klines"
FUTURES_KLINES = "https://fapi.binance.com/fapi/v1/klines"
NO_FILL = {"REJECTED", "CANCELLED", "CANCELED", "EXPIRED"}
NAMES = {
    "BTC": r"\bbitcoin(?!\s+cash)\b", "ETH": r"\bethereum(?!\s+classic)\b",
    "SOL": r"\bsolana\b", "XRP": r"\bxrp\b", "DOGE": r"\bdogecoin\b",
    "BNB": r"\bbinance\s+coin\b", "HYPE": r"\bhyperliquid\b",
}



import hashlib
import stat
import zipfile
from pathlib import Path
import lightgbm as lgb
import matplotlib.pyplot as plt


def utc(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


def safe_error(error, key=""):
    text = f"{type(error).__name__}: {error}"
    return text.replace(key, "[REDACTED]")[:400] if key else text[:400]


class PacedAPI:
    """Private SDK calls are sequential and spaced; writes are never retried."""
    def __init__(self, client):
        self.client, self.last = client, -1e9

    def call(self, name, *args, **kwargs):
        time.sleep(max(0, 0.7 - (time.monotonic() - self.last)))
        self.last = time.monotonic()
        return getattr(self.client, name)(*args, **kwargs)


def wait_for_entry():
    now = datetime.now(timezone.utc)
    hour = now.replace(minute=0, second=0, microsecond=0)
    if (now - hour).total_seconds() <= max(0, MAX_ENTRY_MINUTES * 60 - PREPARATION_SECONDS):
        return
    next_hour = hour + timedelta(hours=1)
    wake = next_hour + timedelta(seconds=20)
    delay = max(0, (wake - datetime.now(timezone.utc)).total_seconds())
    print(f"Waiting {delay/60:.1f} minutes; resume {wake:%Y-%m-%d %H:%M:%S} UTC.", flush=True)
    print("Keep Colab connected. Stop the cell to cancel. No order attempted yet.", flush=True)
    time.sleep(delay)
    if (datetime.now(timezone.utc) - next_hour).total_seconds() > MAX_ENTRY_MINUTES * 60:
        raise ValueError("Colab resumed after the entry window. Rerun; no order attempted.")


def load_feed(symbol):
    response = requests.get(
        "https://feeds.finance.yahoo.com/rss/2.0/headline",
        params={"s": symbol, "region": "US", "lang": "en-US"},
        headers={"User-Agent": "Mozilla/5.0"}, timeout=20,
    )
    response.raise_for_status()
    now, stories = datetime.now(timezone.utc), []
    for item in ET.fromstring(response.content).findall(".//item"):
        try:
            date = parsedate_to_datetime(item.findtext("pubDate"))
            if date.tzinfo is None:
                continue
            date = date.astimezone(timezone.utc)
        except (TypeError, ValueError, AttributeError, OverflowError):
            continue
        if not now - timedelta(hours=24) <= date <= now:
            continue
        title = html.unescape((item.findtext("title") or "").strip())
        link = html.unescape((item.findtext("link") or "").strip())
        try:
            parts = urlsplit(link)
        except ValueError:
            continue
        if not title or parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
            continue
        canonical = urlunsplit(("https", parts.netloc.lower(), parts.path.rstrip("/"), "", ""))
        title_key = re.sub(r"\W+", " ", unicodedata.normalize("NFKC", title).casefold()).strip()
        summary = html.unescape(re.sub(r"<[^>]+>", " ", item.findtext("description") or ""))
        stories.append({"title": title, "summary": summary, "date": date,
                        "url": link, "url_key": canonical, "title_key": title_key})
    return stories


def relevant(asset, story):
    text = story["title"] + " " + story["summary"]
    if re.search(NAMES[asset], text, re.I):
        return True
    # Require a dollar ticker for ambiguous words SOL, DOGE and HYPE.
    if re.search(rf"(?<!\w)\${asset}\b", story["title"]):
        return True
    return asset in {"BTC", "ETH", "XRP", "BNB"} and bool(
        re.search(rf"\b{asset}\b", story["title"])
    )


def get_news_many(assets):
    symbols = [asset + "-USD" for asset in assets]
    if "BTC" in assets:
        symbols += ["MSTR", "IBIT"]
    if "ETH" in assets:
        symbols += ["ETHA", "FETH"]
    pool = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(load_feed, s): s for s in symbols}
        for future in as_completed(futures):
            try:
                pool.extend(future.result())
            except Exception as error:
                print(f"Yahoo {futures[future]} unavailable: {safe_error(error)}")
    pool.sort(key=lambda story: story["date"], reverse=True)
    unique, titles, urls = [], set(), set()
    for story in pool:
        if story["title_key"] in titles or story["url_key"] in urls:
            continue
        titles.add(story["title_key"])
        urls.add(story["url_key"])
        unique.append(story)
    return {asset: [s for s in unique if relevant(asset, s)][:5] for asset in assets}


def reference_for(market):
    asset = market["asset"].upper()
    start, end = utc(market["start_date"]), utc(market["end_date"])
    description = (market.get("description") or "").lower()
    if (end - start != timedelta(hours=1) or start.minute or start.second
            or "binance" not in description or f"{asset}/usdt".lower() not in description
            or "greater than or equal" not in description):
        raise ValueError("Unsupported settlement source, window or tie rule.")
    if f"/futures/{asset}usdt".lower() in description:
        return FUTURES_KLINES, "Binance futures"
    if f"/trade/{asset}_usdt".lower() in description:
        return SPOT_KLINES, "Binance spot"
    raise ValueError("Cannot verify whether the contract resolves on spot or futures.")


def select_markets(windows, now):
    selected = {}
    for market in windows:
        asset = str(market.get("asset", "")).upper()
        if asset not in ASSETS or market.get("closed") or market.get("resolved"):
            continue
        try:
            start, end = utc(market["start_date"]), utc(market["end_date"])
            if not start <= now < end or (end - now).total_seconds() <= 120:
                continue
            if asset not in selected or end < utc(selected[asset]["end_date"]):
                selected[asset] = market
        except (KeyError, ValueError, TypeError):
            continue
    return selected


def order_data(response):
    if not isinstance(response, dict):
        return {}
    return response["order"] if isinstance(response.get("order"), dict) else response


def settle_order(api, response):
    data = order_data(response)
    order_id = data.get("order_id") or data.get("id")
    status = str(data.get("status", "")).upper().removeprefix("ORDER_STATUS_")
    if status == "PENDING" and order_id and str(order_id) != "0":
        for _ in range(4):
            time.sleep(1.5 + random.random() * 0.3)
            data = order_data(api.call("get_order", order_id, source="pending"))
            status = str(data.get("status", "")).upper().removeprefix("ORDER_STATUS_")
            if status != "PENDING":
                break
    if status == "FILLED" and order_id and str(order_id) != "0":
        return "FILLED", order_id, data
    if status in NO_FILL:
        return status, order_id, data
    return "UNCONFIRMED", order_id, data


# Learning storage, model, chart and backup settings.
ROOT = Path("crypto_learning")
BACKUP = Path("crypto_learning_backup.zip")
RESTORE_FROM = ""                 # Optional path to YOUR previously downloaded ZIP.
AUTO_DOWNLOAD_BACKUP = True
SHOW_CHARTS = True
INDICATORS = ("RSI", "MACD", "Volume", "EMA", "VWAP")
VERSION = "gbdt-markov-v1"         # Change this if you change indicator/feature definitions.
PARAMS = dict(n_estimators=400, learning_rate=0.03, max_depth=3, num_leaves=7,
    min_child_samples=60, reg_alpha=0.3, reg_lambda=5.0, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.8, max_bin=63, n_jobs=1,
    random_state=42, verbosity=-1, objective="regression", metric="l1")
S = {}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temporary.replace(path)


def save_state():
    write_json(ROOT / "history.json", S)


def restore_backup(path):
    # No pickle or executable files. Restore only backups you created yourself.
    ROOT.mkdir(exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > 2000 or sum(m.file_size for m in members) > 100_000_000:
            raise ValueError("Backup exceeds size/file-count limit.")
        root = ROOT.resolve()
        for member in members:
            destination = (ROOT / member.filename).resolve()
            if (root not in destination.parents or member.is_dir()
                    or destination.suffix not in {".json", ".txt", ".png", ".csv"}
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError("Unsafe or unexpected backup member.")
        archive.extractall(ROOT)


def export_backup():
    save_state()
    with zipfile.ZipFile(BACKUP, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in ROOT.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".txt", ".png", ".csv"}:
                archive.write(path, path.relative_to(ROOT))
    print("Learning backup:", BACKUP.resolve())
    if AUTO_DOWNLOAD_BACKUP:
        try:
            from google.colab import files
            files.download(str(BACKUP))
        except ImportError:
            print("Download the ZIP from your file browser.")


def load_bars(market):
    asset = market["asset"].upper()
    endpoint, source = reference_for(market)
    start, end = utc(market["start_date"]), utc(market["end_date"])
    rows, until = [], int(end.timestamp() * 1000) - 1
    for _ in range(HISTORY_PAGES):
        response = requests.get(endpoint, params={"symbol": asset + "USDT", "interval": "1h",
            "limit": 1000, "endTime": until}, timeout=20)
        response.raise_for_status()  # No workaround if the correct feed is restricted.
        page = response.json()
        if not isinstance(page, list) or not page:
            break
        rows.extend(page)
        until = min(int(row[0]) for row in page) - 1
        if len(page) < 1000:
            break
    if not rows:
        raise ValueError("No OHLCV data.")
    frame = pd.DataFrame({"time": [pd.to_datetime(r[0], unit="ms", utc=True) for r in rows],
        "o": [float(r[1]) for r in rows], "h": [float(r[2]) for r in rows],
        "l": [float(r[3]) for r in rows], "c": [float(r[4]) for r in rows],
        "v": [float(r[5]) for r in rows], "q": [float(r[7]) for r in rows],
        "end_ms": [int(r[6]) for r in rows]}).drop_duplicates("time").set_index("time").sort_index()
    anchor = pd.Timestamp(start)
    if anchor not in frame.index:
        raise ValueError("Missing exact contract opening candle.")
    opening = float(frame.loc[anchor, "o"])
    past = frame[frame.end_ms < int(start.timestamp() * 1000)].copy()
    if past.empty or past.index.max() != anchor - pd.Timedelta(hours=1) or opening <= 0:
        raise ValueError("Missing latest completed candle/opening price.")
    past = past.reindex(pd.date_range(past.index.min(), past.index.max(), freq="h"))
    bad = (past[["o", "h", "l", "c"]] <= 0).any(axis=1) | (past.v < 0) | (past.q < 0)
    past.loc[bad, :] = np.nan
    return past, opening, endpoint, source


def features(frame):
    f = frame.copy()
    f["r"] = 100 * (f.c / f.o - 1)
    f["d"] = f.r.diff()
    delta = f.c.diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    f["RSI"] = (100 - 100 / (1 + up / down.replace(0, np.nan))).where(down > 0, np.where(up > 0, 100, 50))
    f["MACD"] = f.c.ewm(span=12, adjust=False).mean() - f.c.ewm(span=26, adjust=False).mean()
    signal = f.MACD.ewm(span=9, adjust=False).mean()
    f["Volume"] = f.v
    f["EMA"] = f.c.ewm(span=20, adjust=False).mean()
    f["VWAP"] = f.q.rolling(24).sum() / f.v.rolling(24).sum().replace(0, np.nan)
    levels = {"RSI": f.RSI / 100, "MACD": 100 * (f.MACD - signal) / f.c,
        "Volume": f.v / f.v.rolling(24).mean().replace(0, np.nan),
        "EMA": 100 * (f.c / f.EMA - 1), "VWAP": 100 * (f.c / f.VWAP - 1)}
    changes = {"RSI": 100 * f.RSI.diff() / f.RSI.shift().abs().clip(lower=1),
        "MACD": 100 * f.MACD.diff() / f.c.shift(),
        "Volume": 100 * f.v.diff() / f.v.shift().replace(0, np.nan),
        "EMA": 100 * f.EMA.pct_change(fill_method=None),
        "VWAP": 100 * f.VWAP.pct_change(fill_method=None)}
    groups = {}
    for name in INDICATORS:
        f[name + "_level"] = levels[name]
        for lag in range(3):
            f[name + f"_chg{lag}"] = changes[name].shift(lag)
        groups[name] = [name + "_level"] + [name + f"_chg{k}" for k in range(3)] + ["r", "r1", "r2", "d"]
    f["r1"], f["r2"] = f.r.shift(), f.r.shift(2)
    f["y"] = f.r.shift(-1)  # NEXT hour's open-to-close return, not a price level.
    return f.replace([np.inf, -np.inf], np.nan), groups


def markov_fit(frame):
    r = 100 * (frame.c / frame.o - 1)
    change = r.diff()
    pairs = pd.DataFrame({"current": change, "next": change.shift(-1)}).dropna()
    if len(pairs) < 500:
        raise ValueError("Insufficient Markov history.")
    cuts = np.quantile(pairs.current, np.arange(1, STATES) / STATES)
    if len(np.unique(cuts)) != STATES - 1:
        raise ValueError("Non-distinct Markov states.")
    src = np.searchsorted(cuts, pairs.current, side="right")
    dst = np.searchsorted(cuts, pairs.next, side="right")
    counts = np.ones((STATES, STATES)) * ALPHA
    np.add.at(counts, (src, dst), 1)
    return dict(cuts=cuts, T=counts / counts.sum(axis=1, keepdims=True), src=src,
                dst=dst, changes=pairs.next.to_numpy())


def markov_predict(model, r, change, scores=()):
    state = int(np.searchsorted(model["cuts"], change, side="right"))
    rows = []
    for score in scores or [0.0]:
        matrix = model["T"] * np.exp(NEWS_STRENGTH * score * np.linspace(-1, 1, STATES))[None, :]
        matrix /= matrix.sum(axis=1, keepdims=True)
        rows.append(matrix[state])
    probabilities, means = [], []
    for j in range(STATES):
        local = model["changes"][(model["src"] == state) & (model["dst"] == j)]
        samples = local if len(local) >= 20 else model["changes"][model["dst"] == j]
        if not len(samples) or (r + samples <= -100).any():
            raise ValueError("Invalid Markov return samples.")
        probabilities.append(np.mean(r + samples >= 0))
        means.append(np.mean(r + samples))
    combined = np.mean(rows, axis=0)
    return float(combined @ probabilities), float(combined @ means)


def residual_probability(prediction, errors):
    return np.mean(np.asarray(prediction)[..., None] + np.asarray(errors) >= 0, axis=-1)


def resolved_errors(asset, endpoint):
    unique = {}
    for record in sorted(S.get("forecasts", []), key=lambda x: x["created"]):
        if (record["asset"] == asset and record.get("resolved") and record.get("forward")
                and record.get("version") == VERSION and record["endpoint"] == endpoint):
            unique.setdefault(record["start"], record["actual_return"] - record["gbdt_raw"])
    return list(unique.values())[-100:]


def resolve_history(asset, frame, endpoint):
    count = 0
    for record in S.get("forecasts", []):
        if record["asset"] != asset or record.get("resolved") or record["endpoint"] != endpoint:
            continue
        if utc(record["end"]) >= datetime.now(timezone.utc):
            continue  # Never learn an outcome before its contract has ended.
        stamp = pd.Timestamp(record["start"])
        if stamp not in frame.index:
            continue  # Missing outcomes stay unresolved, never invented.
        bar = frame.loc[stamp]
        if not np.isfinite([bar.o, bar.c]).all() or bar.o <= 0 or bar.c <= 0:
            continue
        actual_return = 100 * (bar.c / bar.o - 1)
        record.update(resolved=True, actual_close=float(bar.c), actual_return=float(actual_return),
            error_pct=float(100 * (record["pred_close"] / bar.c - 1)),
            return_error_pp=float(record["blend_return"] - actual_return))
        count += 1
    by_id = {r["id"]: r for r in S.get("forecasts", [])}
    for trade in S.get("trades", []):
        forecast = by_id.get(trade.get("forecast_id"), {})
        if forecast.get("resolved"):
            trade.update(actual_close=forecast["actual_close"], price_error_pct=forecast["error_pct"])
    return count


def train_or_load(asset, frame, f, groups, endpoint):
    columns = list(dict.fromkeys(c for group in groups.values() for c in group))
    samples = f.dropna(subset=columns + ["y"])
    if len(samples) < MIN_TRANSITIONS or not np.isfinite(f[columns].iloc[-1]).all():
        raise ValueError("Insufficient clean indicator/target data.")
    folder = ROOT / "models" / asset
    folder.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256((VERSION + endpoint + json.dumps(PARAMS, sort_keys=True)
        + frame.tail(48).to_json()).encode()).hexdigest()
    meta_path = folder / "meta.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        if metadata.get("fingerprint") == fingerprint and all((folder / (n + ".txt")).exists() for n in INDICATORS):
            models = [lgb.Booster(model_file=str(folder / (n + ".txt"))) for n in INDICATORS]
            return metadata, models
    n = len(samples)
    a, b, c, d = [int(n * fraction) for fraction in (0.60, 0.75, 0.825, 0.90)]
    train, validation = samples.iloc[:a-1], samples.iloc[a:b-1]
    score_set, blend_set, test = samples.iloc[b:c-1], samples.iloc[c:d-1], samples.iloc[d:]
    # One-sample gaps; chronological splits, no shuffled validation.
    blocks = [score_set, blend_set, test]
    predictions = [[], [], []]
    best_iterations = []
    for name in INDICATORS:
        model = lgb.LGBMRegressor(**PARAMS)
        model.fit(train[groups[name]], train.y,
            eval_set=[(validation[groups[name]], validation.y)], eval_metric="l1",
            callbacks=[lgb.early_stopping(25, first_metric_only=True, verbose=False)])
        best_iterations.append(max(1, int(model.best_iteration_ or PARAMS["n_estimators"])))
        for store, block in zip(predictions, blocks):
            store.append(model.predict(block[groups[name]]))
    predictions = [np.asarray(p).T for p in predictions]
    mae = np.mean(np.abs(predictions[0] - score_set.y.to_numpy()[:, None]), axis=0)
    baseline_mae = max(float(np.mean(np.abs(score_set.y))), 1e-6)
    scores = 100 / (1 + mae / baseline_mae)  # Closer NEXT-hour return -> higher score.
    inverse = 1 / (mae + 1e-6)
    weights = 0.8 * inverse / inverse.sum() + 0.2 / len(INDICATORS)
    residuals = score_set.y.to_numpy() - predictions[0] @ weights
    historical_bias = float(np.median(residuals))
    markov = markov_fit(frame.loc[:samples.index[b-1]])
    mp, mr = [], []
    for block in (blend_set, test):
        outputs = [markov_predict(markov, row.r, row.d) for _, row in block.iterrows()]
        mp.append(np.array([x[0] for x in outputs])); mr.append(np.array([x[1] for x in outputs]))
    gp = [residual_probability(p @ weights, residuals) for p in predictions[1:]]
    gr = [p @ weights + historical_bias for p in predictions[1:]]
    grid = np.linspace(0, 1, 11)
    # Historical Markov has NO article/news adjustments: no historical news archive exists.
    wp = float(min(grid, key=lambda w: np.mean((w*gp[0] + (1-w)*mp[0] - (blend_set.y >= 0))**2)))
    wr = float(min(grid, key=lambda w: np.mean(np.abs(w*gr[0] + (1-w)*mr[0] - blend_set.y))))
    test_return = wr*gr[1] + (1-wr)*mr[1]
    test_probability = wp*gp[1] + (1-wp)*mp[1]
    metadata = dict(fingerprint=fingerprint, weights=weights.tolist(), scores=scores.tolist(),
        indicator_mae_pp=mae.tolist(), residuals=residuals.tolist(), wp=wp, wr=wr,
        iterations=best_iterations, test_mae_pp=float(np.mean(np.abs(test_return-test.y))),
        test_baseline_mae_pp=float(np.mean(np.abs(test.y))),
        test_brier=float(np.mean((test_probability-(test.y >= 0))**2)),
        test_n=len(test), asof=frame.index[-1].isoformat())
    # Evaluate BEFORE this refit. Final live models use ALL known targets, including new traded hours.
    models = []
    for name, iterations in zip(INDICATORS, best_iterations):
        final = lgb.LGBMRegressor(**{**PARAMS, "n_estimators": iterations})
        final.fit(samples[groups[name]], samples.y)
        final.booster_.save_model(str(folder / (name + ".txt")))
        models.append(final.booster_)
    write_json(meta_path, metadata)
    return metadata, models


def make_ml_candidate(market, articles, data):
    frame, opening, endpoint, source = data
    asset = market["asset"].upper()
    f, groups = features(frame)
    metadata, models = train_or_load(asset, frame, f, groups, endpoint)
    raw = np.array([model.predict(f[groups[name]].iloc[[-1]])[0] for name, model in zip(INDICATORS, models)])
    graw = float(raw @ metadata["weights"])
    residuals = np.array(metadata["residuals"])
    online = resolved_errors(asset, endpoint)
    strength = len(online) / (len(online) + 50) if online else 0.0
    bias = (1-strength)*float(np.median(residuals)) + strength*(float(np.median(online)) if online else 0)
    gp = (1-strength)*float(residual_probability(graw, residuals))
    if online:
        gp += strength*float(residual_probability(graw, online))
    greturn = graw + bias
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(a["title"] + ". " + a["summary"])["compound"] for a in articles]
    mp, mreturn = markov_predict(markov_fit(frame), float(f.r.iloc[-1]), float(f.d.iloc[-1]), sentiments)
    p_up = metadata["wp"]*gp + (1-metadata["wp"])*mp
    predicted_return = metadata["wr"]*greturn + (1-metadata["wr"])*mreturn
    if not np.isfinite([predicted_return, opening, gp, mp]).all() or predicted_return <= -100:
        raise ValueError("Invalid price forecast.")
    labels = market["markets"][0]["outcomes"]
    labels = json.loads(labels) if isinstance(labels, str) else labels
    outcomes = {label.casefold(): label for label in labels}
    if set(outcomes) != {"up", "down"}:
        raise ValueError("Unexpected market outcomes.")
    direction = "up" if p_up >= 0.5 else "down"
    return dict(asset=asset, market=market, start=utc(market["start_date"]), end=utc(market["end_date"]),
        outcome=outcomes[direction], p_up=float(p_up), p_win=float(max(p_up, 1-p_up)),
        news=articles, source=source, endpoint=endpoint, opening=opening, meta=metadata,
        gbdt_raw=graw, gbdt_return=greturn, markov_return=mreturn, blend_return=float(predicted_return),
        pred_close=float(opening*(1+predicted_return/100)), gp=float(gp), mp=float(mp),
        online_n=len(online), frame=f)


def plot_candidate(candidate):
    f = candidate["frame"].tail(100)
    times = f.index + pd.Timedelta(hours=1)
    figure, axes = plt.subplots(3, 2, figsize=(15, 11))
    for axis, name, score, weight in zip(axes.flat, INDICATORS,
            candidate["meta"]["scores"], candidate["meta"]["weights"]):
        axis.plot(times, f.c, color="black", linewidth=1, label="Price")
        other = axis.twinx()
        other.plot(times, f[name], color="tab:blue", alpha=0.7)
        if name in {"EMA", "VWAP"}:
            other.set_ylim(axis.get_ylim())  # Both curves have the same price units.
        axis.set_title(f"{name}: predictive score {score:.1f}/100; weight {weight:.1%}")
        axis.set_ylabel("Price (USDT)"); other.set_ylabel(name)
        axis.tick_params(axis="x", rotation=20)
    axis = axes.flat[-1]
    axis.plot(times[-12:], f.c.iloc[-12:], color="black", label="Completed closes")
    for label, value, color in [("GBDT", candidate["gbdt_return"], "green"),
            ("Markov/news", candidate["markov_return"], "orange"),
            ("Blended", candidate["blend_return"], "purple")]:
        axis.plot([candidate["start"], candidate["end"]],
            [candidate["opening"], candidate["opening"]*(1+value/100)], "o--", label=label, color=color)
    axis.set_title("Next contract close: forecasts, NOT known outcomes")
    axis.legend(fontsize=8); axis.tick_params(axis="x", rotation=20)
    figure.suptitle(candidate["asset"] + " — chronological predictive scores, not causal elasticity")
    figure.tight_layout()
    folder = ROOT / "charts"; folder.mkdir(exist_ok=True)
    figure.savefig(folder / (candidate["asset"] + ".png"), dpi=130)
    if SHOW_CHARTS:
        plt.show()
    plt.close(figure)


def record_forecast(candidate, run_id):
    candidate["forecast_id"] = str(uuid4())
    now = datetime.now(timezone.utc)
    S["forecasts"].append(dict(id=candidate["forecast_id"], run_id=run_id,
        asset=candidate["asset"], start=candidate["start"].isoformat(), end=candidate["end"].isoformat(),
        created=now.isoformat(), forward=now < candidate["end"], endpoint=candidate["endpoint"],
        version=VERSION, opening=candidate["opening"], gbdt_raw=candidate["gbdt_raw"],
        blend_return=candidate["blend_return"], pred_close=candidate["pred_close"], resolved=False))
    save_state()


def place_five(api, candidates, run_id, key):
    reserved, filled, records, seen = Decimal("0"), 0, [], set()
    for candidate in candidates:
        if filled >= TARGET_BETS or reserved + PER_BET > RUN_BUDGET:
            break
        market = candidate["market"]
        cid = market["condition_id"]
        if cid in seen:
            continue
        seen.add(cid)
        now = datetime.now(timezone.utc)
        if (not candidate["start"] <= now < candidate["end"]
                or (now - candidate["start"]).total_seconds() > MAX_ENTRY_MINUTES * 60
                or (candidate["end"] - now).total_seconds() <= 120):
            print(f"{candidate['asset']}: timing safeguard; no order.")
            continue
        # New UUID per run means a deliberate rerun can place NEW bets.
        order_key = str(uuid5(NAMESPACE_URL, run_id + "|" + cid + "|" + candidate["outcome"]))
        record = {"asset": candidate["asset"], "outcome": candidate["outcome"],
                  "order_id": None, "status": "SUBMITTING", "max_notional": str(PER_BET), "forecast_id": candidate["forecast_id"], "run_id": run_id}
        records.append(record)
        S["trades"].append(record)
        save_state()  # Journal intent BEFORE the write; never store the API key.
        reserved += PER_BET  # Reserve BEFORE sending, including ambiguous outcomes.
        print(f"\nPAPER BUY {candidate['asset']} {candidate['outcome']}: up to ${PER_BET} before fees.")
        try:
            response = api.call("place_order", market_id=cid, side="BUY", outcome=candidate["outcome"],
                amount=str(PER_BET), order_type="market", price=str(MAX_SHARE_PRICE),
                time_in_force="FOK", client_order_id=order_key, idempotency_key=order_key)
            print("ORDER RESPONSE:", json.dumps(response, indent=2, default=str))
            status, order_id, final = settle_order(api, response)
            record.update(status=status, order_id=order_id)
            save_state()
            if status == "FILLED":
                filled += 1
                print(f"Confirmed filled contracts: {filled}/{TARGET_BETS}")
            elif status in NO_FILL:
                reserved -= PER_BET
                print("No fill. Trying another distinct coin if available.")
            else:
                print("Unconfirmed: keep its budget reserved; do not count it as filled.")
        except (Exception, KeyboardInterrupt) as error:
            record["status"] = "UNCONFIRMED_ERROR"
            save_state()
            print(safe_error(error, key))
            print("STOP: submission/status may be uncertain. Check order history before rerunning.")
            break  # Never retry a write or spend around an unknown submission error.
    print("\nRUN SUMMARY")
    if records:
        print(pd.DataFrame(records).to_string(index=False))
    print(f"Confirmed filled: {filled}/{TARGET_BETS}; maximum notional reserved: ${reserved} before fees.")
    if filled < TARGET_BETS:
        print("SHORTFALL: five fills were not confirmed. No rejected/unconfirmed request was counted as a bet.")
    return {"filled": filled, "reserved": reserved, "orders": records}


def run_bot():
    global S
    print("GBDT + MARKOV — PAPER ONLY; target 5 fills, $5 each/$25 notional BEFORE fees.")
    key = getpass("PolySimulator API key (hidden): ").strip()
    if not key:
        print("No key; no orders."); return
    run_id, stage = str(uuid4()), "authentication"
    try:
        with PolySimClient(api_key=key, base_url="https://api.polysimulator.com", timeout=20, max_retries=0) as client:
            api = PacedAPI(client)
            account = api.call("me")
            if not set(account.get("permissions") or []).intersection({"trade", "write"}):
                raise ValueError("Effective trading permission not confirmed.")
            stage = "learning storage"
            ROOT.mkdir(exist_ok=True)
            if RESTORE_FROM and not (ROOT / "history.json").exists():
                restore_backup(RESTORE_FROM)
            elif RESTORE_FROM:
                print("Existing history kept; backup restore was not repeated.")
            history_file = ROOT / "history.json"
            S = json.loads(history_file.read_text()) if history_file.exists() else {"forecasts": [], "trades": []}
            if not isinstance(S.get("forecasts"), list) or not isinstance(S.get("trades"), list):
                raise ValueError("Invalid learning-history schema.")
            print("Run:", run_id, "— rerunning creates NEW bets. Keep paper funds for fees too.")
            stage = "waiting"
            wait_for_entry()
            stage = "market discovery"
            markets = select_markets(api.call("list_updown", interval="1H", live=True), datetime.now(timezone.utc))
            if not markets:
                raise ValueError("No eligible hourly markets.")
            stage = "fresh news and OHLCV"
            news = get_news_many(list(markets))
            data = {}
            with ThreadPoolExecutor(max_workers=4) as executor:
                tasks = {executor.submit(load_bars, market): asset for asset, market in markets.items()}
                for future in as_completed(tasks):
                    asset = tasks[future]
                    try:
                        data[asset] = future.result()
                    except Exception as error:
                        print("SKIP", asset, safe_error(error, key))
            resolved = 0
            for asset, item in data.items():
                resolved += resolve_history(asset, item[0], item[2])
                folder = ROOT / "data"; folder.mkdir(exist_ok=True)
                item[0].to_csv(folder / (asset + ".csv"))
            save_state()
            print("Newly resolved forecasts:", resolved, "(unknown outcomes remain pending).")
            stage = "GBDT training and evaluation"
            candidates = []
            with ThreadPoolExecutor(max_workers=2) as executor:
                tasks = {executor.submit(make_ml_candidate, markets[a], news.get(a, []), item): a for a, item in data.items()}
                for future in as_completed(tasks):
                    try:
                        candidates.append(future.result())
                    except Exception as error:
                        print("SKIP", tasks[future], safe_error(error, key))
            if not candidates:
                raise ValueError("No valid model candidates.")
            candidates.sort(key=lambda c: (-c["p_win"], c["asset"]))
            for candidate in candidates:
                record_forecast(candidate, run_id)
                metadata = candidate["meta"]
                print(f"\n{candidate['asset']}: forecast close={candidate['pred_close']:.6f}; "
                      f"GBDT P(Up)={candidate['gp']:.3f}, Markov/news={candidate['mp']:.3f}, blended={candidate['p_up']:.3f}")
                print(f"Price blend GBDT weight={metadata['wr']:.1%}; probability blend={metadata['wp']:.1%}; "
                      f"prior unique-hour online errors={candidate['online_n']}; articles={len(candidate['news'])}")
                print(pd.DataFrame({"Indicator": INDICATORS, "Predictive_score": metadata["scores"],
                    "Weight": metadata["weights"], "Heldout_MAE_pp": metadata["indicator_mae_pp"]}).round(4).to_string(index=False))
                print(f"Pre-refit chronological test: MAE={metadata['test_mae_pp']:.4f} pp; "
                      f"zero-return baseline={metadata['test_baseline_mae_pp']:.4f} pp; Brier={metadata['test_brier']:.4f}.")
                plot_candidate(candidate)
            print("\nScores are predictive diagnostics, not causal elasticity or guaranteed win probabilities.")
            print("News tilts, residual probabilities and live refits remain experimental/uncalibrated.")
            stage = "paper execution"
            return place_five(api, candidates, run_id, key)
    except (Exception, KeyboardInterrupt) as error:
        print("STOP during", stage, safe_error(error, key))
        print("No orders were attempted." if stage != "paper execution" else
              "Check order history before retrying if a submission was interrupted.")
    finally:
        if S:
            try:
                pd.DataFrame(S["forecasts"]).to_csv(ROOT / "forecasts.csv", index=False)
                pd.DataFrame(S["trades"]).to_csv(ROOT / "trades.csv", index=False)
                export_backup()
            except Exception as error:
                print("Backup failed:", safe_error(error, key))


if __name__ == "__main__":
    run_bot()
