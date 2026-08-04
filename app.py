from __future__ import annotations

import io
import re
import time as time_module
from datetime import time
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="日本株 決算トレーダー分析",
    page_icon="📊",
    layout="wide",
)

JQUANTS_URL = "https://api.jquants.com/v2/fins/summary"
JQUANTS_MINUTE_URL = "https://api.jquants.com/v2/equities/bars/minute"
VALID_QUARTERS = ["1Q", "2Q", "3Q", "本決算"]


# -----------------------------
# 共通ユーティリティ
# -----------------------------
def normalize_code(raw: str) -> str:
    code = raw.strip().upper().replace(".T", "")
    if not re.fullmatch(r"[0-9A-Z]{4,6}", code):
        raise ValueError("銘柄コードは半角英数字4〜6文字で入力してください。")
    return code


def ticker_for(code: str) -> str:
    return f"{code}.T"


def jquants_code(code: str) -> str:
    return code if len(code) == 5 else f"{code}0"


def _extract_records(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("data", "statements", "fin_summary", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    for value in payload.values():
        if isinstance(value, list) and (not value or isinstance(value[0], dict)):
            return [x for x in value if isinstance(x, dict)]
    return []


def _pick(record: dict, *names: str) -> object:
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return ""


def quarter_from_doc_type(doc_type: str) -> str:
    value = str(doc_type)
    if value.startswith("1QFinancialStatements_"):
        return "1Q"
    if value.startswith("2QFinancialStatements_"):
        return "2Q"
    if value.startswith("3QFinancialStatements_"):
        return "3Q"
    if value.startswith("FYFinancialStatements_"):
        return "本決算"
    return ""


def document_priority(doc_type: str) -> int:
    value = str(doc_type)
    score = 0
    if "Consolidated" in value and "NonConsolidated" not in value:
        score += 20
    if any(x in value for x in ("IFRS", "JMIS", "JP")):
        score += 5
    return score


def normalize_clock(raw: object) -> str:
    text = str(raw).strip()
    if re.fullmatch(r"(?:[01]?\d|2[0-3]):[0-5]\d:[0-5]\d", text):
        return text[:5]
    if re.fullmatch(r"(?:[01]?\d|2[0-3]):[0-5]\d", text):
        hour, minute = text.split(":")
        return f"{int(hour):02d}:{int(minute):02d}"
    return ""


def close_time_for(day: pd.Timestamp) -> time:
    return time(15, 30) if day.normalize() >= pd.Timestamp("2024-11-05") else time(15, 0)


def classify_announcement(day: pd.Timestamp, clock_text: str, is_trading_day: bool) -> tuple[str, str]:
    if not is_trading_day:
        return "休場日発表", "決算跨ぎ対象"
    if not clock_text:
        return "時刻不明", "判定不能"
    hour, minute = map(int, clock_text.split(":"))
    announced = time(hour, minute)
    close_time = close_time_for(day)
    if announced < time(9, 0):
        return "寄り前", "場中分析対象外"
    if announced < time(11, 30):
        return "前場中", "場中分析対象"
    if announced < time(12, 30):
        return "昼休み", "場中分析対象"
    if announced < close_time:
        return "後場中", "場中分析対象"
    return "引け後", "決算跨ぎ対象"


def reaction_start_timestamp(day: pd.Timestamp, clock_text: str, session: str) -> pd.Timestamp | None:
    if not clock_text:
        return None
    day = pd.Timestamp(day).normalize()
    if session == "昼休み":
        return day + pd.Timedelta(hours=12, minutes=30)
    hour, minute = map(int, clock_text.split(":"))
    return day + pd.Timedelta(hours=hour, minutes=minute)


def confidence_label(n: int) -> str:
    if n >= 8:
        return "高"
    if n >= 5:
        return "中"
    return "低"


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):+.2f}%"


def sample_judgement(n: int) -> tuple[str, str]:
    """サンプル数に応じた表示用の判定と注意文。"""
    if n >= 8:
        return "実用参考", "サンプル8件以上"
    if n >= 5:
        return "参考", "サンプル5〜7件"
    if n >= 3:
        return "暫定", "サンプル3〜4件"
    return "判定保留", "サンプル2件以下"


def api_error_category(status_code: int | None, message: str = "") -> str:
    text = str(message).lower()
    if status_code == 401:
        return "APIキー認証エラー"
    if status_code == 403:
        return "契約権限不足または権限反映待ち"
    if status_code == 429:
        return "APIレート制限"
    if status_code and status_code >= 500:
        return "J-Quants側の一時障害"
    if "timeout" in text:
        return "通信タイムアウト"
    return "通信・応答エラー"


# -----------------------------
# データ取得
# -----------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_daily_prices(ticker: str, years: int) -> pd.DataFrame:
    today = pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None).normalize()
    start = today - pd.DateOffset(years=years, months=9)
    end = today + pd.Timedelta(days=3)
    data = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise RuntimeError("日足株価を取得できませんでした。")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise RuntimeError(f"株価列が不足しています: {', '.join(missing)}")
    out = data[required].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    return out[~out.index.duplicated(keep="last")].sort_index()


@st.cache_data(ttl=21600, show_spinner=False)
def load_jquants_earnings(code: str, api_key: str) -> tuple[pd.DataFrame, str]:
    columns = ["earnings_date", "quarter", "announcement_time", "source", "doc_type"]
    if not api_key:
        return pd.DataFrame(columns=columns), "J-Quants APIキー未設定"

    try:
        response = requests.get(
            JQUANTS_URL,
            params={"code": jquants_code(code)},
            headers={"x-api-key": api_key, "Accept": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        records = _extract_records(response.json())
    except Exception as exc:
        return pd.DataFrame(columns=columns), f"J-Quants取得失敗: {type(exc).__name__}"

    rows: list[dict] = []
    excluded = 0
    for record in records:
        doc_type = str(_pick(record, "DocType", "TypeOfDocument", "document_type")).strip()
        quarter = quarter_from_doc_type(doc_type)
        if not quarter:
            excluded += 1
            continue
        disclosed = pd.to_datetime(
            _pick(record, "DiscDate", "DisclosedDate", "disclosed_date", "Date"),
            errors="coerce",
        )
        if pd.isna(disclosed):
            excluded += 1
            continue
        rows.append(
            {
                "earnings_date": pd.Timestamp(disclosed).normalize(),
                "quarter": quarter,
                "announcement_time": normalize_clock(
                    _pick(record, "DiscTime", "DisclosedTime", "disclosed_time")
                ),
                "source": "J-Quants 財務情報",
                "doc_type": doc_type,
                "_priority": document_priority(doc_type),
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns), f"J-Quants採用0件（API応答{len(records)}件・対象外{excluded}件）"

    raw = pd.DataFrame(rows)
    before = len(raw)
    out = (
        raw.sort_values(
            ["earnings_date", "quarter", "_priority", "announcement_time"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["earnings_date", "quarter"], keep="first")
        .sort_values("earnings_date")
        .reset_index(drop=True)
    )
    duplicates = before - len(out)
    counts = out["quarter"].value_counts().reindex(VALID_QUARTERS, fill_value=0)
    count_text = " / ".join(f"{q}:{int(n)}件" for q, n in counts.items())
    status = (
        f"J-Quants採用{len(out)}件（API応答{len(records)}件・対象外{excluded}件・"
        f"重複統合{duplicates}件）｜{count_text}"
    )
    return out[columns], status


@st.cache_data(ttl=21600, show_spinner=False)
def load_jquants_minute_bars(
    code: str, api_key: str, dates: tuple[str, ...]
) -> tuple[pd.DataFrame, str, bool]:
    """J-Quants分足アドオンから、指定した決算日の1分足を取得する。

    Returns:
        bars: DatetimeIndexのOHLCV
        status: 画面表示用メッセージ
        addon_available: API利用権限があるか
    """
    columns = ["Open", "High", "Low", "Close", "Volume", "Value"]
    if not api_key:
        return pd.DataFrame(columns=columns), "J-Quants APIキー未設定", False
    if not dates:
        return pd.DataFrame(columns=columns), "場中決算候補なし", True

    all_records: list[dict] = []
    failed_dates: list[str] = []
    headers = {"x-api-key": api_key, "Accept": "application/json"}

    for date_text in dates:
        params = {"code": jquants_code(code), "date": date_text}
        pagination_key = ""
        while True:
            if pagination_key:
                params["pagination_key"] = pagination_key
            response = None
            last_exc: Exception | None = None
            for attempt in range(3):
                try:
                    response = requests.get(
                        JQUANTS_MINUTE_URL,
                        params=params,
                        headers=headers,
                        timeout=30,
                    )
                    if response.status_code == 429 or response.status_code >= 500:
                        if attempt < 2:
                            time_module.sleep(1.5 * (2 ** attempt))
                            continue
                    break
                except Exception as exc:
                    last_exc = exc
                    if attempt < 2:
                        time_module.sleep(1.5 * (2 ** attempt))
                        continue

            if response is None:
                category = api_error_category(None, str(last_exc or ""))
                failed_dates.append(f"{date_text}:{category}")
                break

            if response.status_code in (401, 403):
                category = api_error_category(response.status_code)
                message = f"{category}（HTTP {response.status_code}）。契約直後は数分待って再分析してください。"
                return pd.DataFrame(columns=columns), message, False
            if response.status_code == 429:
                failed_dates.append(f"{date_text}:{api_error_category(429)}")
                break
            try:
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                category = api_error_category(response.status_code, str(exc))
                failed_dates.append(f"{date_text}:{category}")
                break

            records = _extract_records(payload)
            all_records.extend(records)
            pagination_key = str(payload.get("pagination_key", "")).strip() if isinstance(payload, dict) else ""
            if not pagination_key:
                break

    if not all_records:
        detail = " / ".join(failed_dates[:3])
        suffix = f"（{detail}）" if detail else ""
        return pd.DataFrame(columns=columns), f"分足API取得0件{suffix}", True

    rows: list[dict] = []
    for record in all_records:
        day = str(_pick(record, "Date", "date")).strip()
        clock = str(_pick(record, "Time", "time")).strip()
        dt = pd.to_datetime(f"{day} {clock}", errors="coerce")
        if pd.isna(dt):
            continue
        rows.append(
            {
                "datetime": pd.Timestamp(dt),
                "Open": pd.to_numeric(_pick(record, "O", "Open"), errors="coerce"),
                "High": pd.to_numeric(_pick(record, "H", "High"), errors="coerce"),
                "Low": pd.to_numeric(_pick(record, "L", "Low"), errors="coerce"),
                "Close": pd.to_numeric(_pick(record, "C", "Close"), errors="coerce"),
                "Volume": pd.to_numeric(_pick(record, "Vo", "Volume"), errors="coerce"),
                "Value": pd.to_numeric(_pick(record, "Va", "Value"), errors="coerce"),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=columns), "分足APIの応答を解析できませんでした", True
    frame = frame.dropna(subset=["datetime", "Open", "High", "Low", "Close"])
    frame = frame.set_index("datetime").sort_index()
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
    frame = frame[~frame.index.duplicated(keep="last")]
    failed = f"・失敗{len(failed_dates)}日" if failed_dates else ""
    return frame[columns], f"J-Quants分足API: {len(frame):,}本・{len(set(frame.index.date))}日{failed}", True


def intraday_candidate_dates(prices: pd.DataFrame, events: pd.DataFrame) -> tuple[str, ...]:
    if events.empty:
        return tuple()
    trading_days = set(prices.index.normalize())
    dates: list[str] = []
    for _, event in events.iterrows():
        day = pd.Timestamp(event["earnings_date"]).normalize()
        clock = normalize_clock(event.get("announcement_time", ""))
        session, strategy = classify_announcement(day, clock, day in trading_days)
        if strategy == "場中分析対象" and session in {"前場中", "昼休み", "後場中"}:
            dates.append(day.strftime("%Y-%m-%d"))
    return tuple(sorted(set(dates)))


# -----------------------------
# 日足・決算跨ぎ分析
# -----------------------------
def analyze_daily_events(prices: pd.DataFrame, events: pd.DataFrame, flat_threshold: float) -> pd.DataFrame:
    rows: list[dict] = []
    index = prices.index
    for _, event in events.iterrows():
        disclosed = pd.Timestamp(event["earnings_date"]).normalize()
        pos = index.searchsorted(disclosed, side="left")
        if pos >= len(index):
            continue
        event_day = index[pos]
        is_trading_day = event_day == disclosed
        if is_trading_day:
            if pos < 1 or pos + 1 >= len(index):
                continue
            prev_day = index[pos - 1]
            next_day = index[pos + 1]
            ref_day = event_day
        else:
            if pos < 1:
                continue
            prev_day = index[pos - 1]
            next_day = event_day
            ref_day = prev_day

        clock = normalize_clock(event.get("announcement_time", ""))
        session, strategy = classify_announcement(disclosed, clock, is_trading_day)
        prev_close = float(prices.loc[prev_day, "Close"])
        event_open = float(prices.loc[event_day, "Open"])
        event_close = float(prices.loc[event_day, "Close"])
        ref_close = float(prices.loc[ref_day, "Close"])
        next_open = float(prices.loc[next_day, "Open"])
        next_close = float(prices.loc[next_day, "Close"])

        same_day_change = (event_close / prev_close - 1) * 100 if is_trading_day else np.nan
        next_gu = (next_open / ref_close - 1) * 100
        next_close_change = (next_close / ref_close - 1) * 100
        next_intraday = (next_close / next_open - 1) * 100
        if strategy == "決算跨ぎ対象":
            result = "上昇" if next_close_change > flat_threshold else (
                "下落" if next_close_change < -flat_threshold else "横ばい"
            )
        else:
            result = "対象外"

        rows.append(
            {
                "決算発表日": disclosed.date(),
                "四半期": event["quarter"],
                "発表時刻": clock or "不明",
                "発表区分": session,
                "分析区分": strategy,
                "データ源": event["source"],
                "前営業日": prev_day.date(),
                "翌営業日": next_day.date(),
                "当日終値騰落率(%)": same_day_change,
                "翌営業日GU率(%)": next_gu,
                "翌営業日寄り後(%)": next_intraday,
                "翌営業日終値騰落率(%)": next_close_change,
                "決算跨ぎ判定": result,
            }
        )
    return pd.DataFrame(rows)


def summarize_daily(detail: pd.DataFrame, target_strategy: str) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    target = detail[detail["分析区分"] == target_strategy].copy()
    if target.empty:
        return pd.DataFrame()
    rows = []
    for quarter, group in target.groupby("四半期"):
        n = len(group)
        rows.append(
            {
                "四半期": quarter,
                "件数": n,
                "勝率(%)": (group["翌営業日終値騰落率(%)"] > 0).mean() * 100,
                "GU率(%)": (group["翌営業日GU率(%)"] > 0).mean() * 100,
                "平均GU(%)": group["翌営業日GU率(%)"].mean(),
                "平均翌日終値(%)": group["翌営業日終値騰落率(%)"].mean(),
                "中央値翌日終値(%)": group["翌営業日終値騰落率(%)"].median(),
                "確信度": confidence_label(n),
            }
        )
    out = pd.DataFrame(rows)
    out["参考スコア"] = (
        out["平均翌日終値(%)"] * 0.5
        + out["中央値翌日終値(%)"] * 0.2
        + out["平均GU(%)"] * 0.2
        + (out["勝率(%)"] - 50) / 10 * 0.1
    ) * np.minimum(out["件数"] / 6, 1)
    out["評価"] = "判定保留"
    mask = out["件数"] >= 2
    ranks = out.loc[mask, "参考スコア"].rank(method="min", ascending=False)
    out.loc[mask, "評価"] = ranks.map(lambda r: "S" if r == 1 else ("A" if r == 2 else ("B" if r == 3 else "C")))
    return out.sort_values(["参考スコア", "件数"], ascending=False).reset_index(drop=True)


# -----------------------------
# 場中CSV分析
# -----------------------------
def read_csv_flexible(raw: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "cp932", "utf-8"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding)
        except Exception as exc:
            last_error = exc
    raise ValueError(f"CSVを読み取れませんでした: {last_error}")


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    normalized = {re.sub(r"[^a-z0-9]", "", str(c).lower()): str(c) for c in columns}
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    return None


def parse_intraday_csv(uploaded_file) -> tuple[pd.DataFrame, str]:
    if uploaded_file is None:
        return pd.DataFrame(), "CSV未選択"
    frame = read_csv_flexible(uploaded_file.getvalue())
    if frame.empty:
        raise ValueError("場中CSVが空です。")

    dt_col = find_column(frame.columns, ["datetime", "date_time", "timestamp", "日時", "時刻"])
    date_col = find_column(frame.columns, ["date", "日付"])
    time_col = find_column(frame.columns, ["time", "時間"])
    open_col = find_column(frame.columns, ["open", "始値"])
    high_col = find_column(frame.columns, ["high", "高値"])
    low_col = find_column(frame.columns, ["low", "安値"])
    close_col = find_column(frame.columns, ["close", "終値"])
    volume_col = find_column(frame.columns, ["volume", "出来高"])

    if dt_col:
        dt = pd.to_datetime(frame[dt_col], errors="coerce")
    elif date_col and time_col:
        dt = pd.to_datetime(frame[date_col].astype(str) + " " + frame[time_col].astype(str), errors="coerce")
    else:
        raise ValueError("CSVには datetime 列、または date と time 列が必要です。")

    required_map = {"Open": open_col, "High": high_col, "Low": low_col, "Close": close_col}
    missing = [name for name, col in required_map.items() if col is None]
    if missing:
        raise ValueError(f"場中CSVに不足している列: {', '.join(missing)}")

    out = pd.DataFrame(index=dt)
    for target, source in required_map.items():
        out[target] = pd.to_numeric(frame[source], errors="coerce").to_numpy()
    out["Volume"] = pd.to_numeric(frame[volume_col], errors="coerce").to_numpy() if volume_col else np.nan
    out = out[~out.index.isna()].dropna(subset=["Open", "High", "Low", "Close"])
    out.index = pd.DatetimeIndex(out.index).tz_localize(None)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if out.empty:
        raise ValueError("有効な場中データがありません。")

    diffs = out.index.to_series().diff().dropna().dt.total_seconds().div(60)
    interval = int(round(diffs[diffs > 0].median())) if (diffs > 0).any() else 0
    return out, f"{len(out):,}本・推定{interval}分足"


def value_at_or_after(day_bars: pd.DataFrame, target: pd.Timestamp, column: str = "Close") -> float | None:
    pos = day_bars.index.searchsorted(target, side="left")
    if pos >= len(day_bars):
        return None
    return float(day_bars.iloc[pos][column])


def classify_intraday_pattern(move5: float, move30: float, close_move: float) -> str:
    if pd.isna(move5) or pd.isna(close_move):
        return "判定不能"
    if move5 > 0 and close_move > 0:
        if close_move >= move5 * 0.8:
            return "素直上昇型"
        return "上昇失速型"
    if move5 < 0 and close_move > 0:
        return "V字回復型"
    if move5 > 0 and close_move <= 0:
        return "行って来い型"
    if move5 < 0 and close_move < 0:
        return "素直下落型"
    if not pd.isna(move30) and move30 > 0 and close_move < 0:
        return "後半失速型"
    return "横ばい型"


def analyze_intraday_events(intraday: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if intraday.empty or events.empty:
        return pd.DataFrame()

    intraday_dates = set(intraday.index.normalize())
    entry_minutes = (0, 5, 10, 15, 30, 60)
    for _, event in events.iterrows():
        day = pd.Timestamp(event["earnings_date"]).normalize()
        clock = normalize_clock(event.get("announcement_time", ""))
        is_trading_day = day in intraday_dates
        session, strategy = classify_announcement(day, clock, is_trading_day)
        if strategy != "場中分析対象" or not clock or not is_trading_day:
            continue

        day_bars = intraday[intraday.index.normalize() == day]
        if day_bars.empty:
            continue
        start = reaction_start_timestamp(day, clock, session)
        if start is None:
            continue

        pre = day_bars[day_bars.index < start]
        post = day_bars[day_bars.index >= start]
        if pre.empty or post.empty:
            continue

        ref_price = float(pre.iloc[-1]["Close"])
        close_price = float(day_bars.iloc[-1]["Close"])
        high_after = float(post["High"].max())
        low_after = float(post["Low"].min())

        horizon_values: dict[int, float | None] = {}
        for minutes in (5, 10, 15, 30, 60):
            horizon_values[minutes] = value_at_or_after(day_bars, start + pd.Timedelta(minutes=minutes))

        def move_from_ref(value: float | None) -> float:
            return (value / ref_price - 1) * 100 if value is not None else np.nan

        row = {
            "決算発表日": day.date(),
            "四半期": event["quarter"],
            "発表時刻": clock,
            "発表区分": session,
            "基準価格": ref_price,
            "5分後(%)": move_from_ref(horizon_values[5]),
            "10分後(%)": move_from_ref(horizon_values[10]),
            "15分後(%)": move_from_ref(horizon_values[15]),
            "30分後(%)": move_from_ref(horizon_values[30]),
            "60分後(%)": move_from_ref(horizon_values[60]),
            "引け時点(%)": move_from_ref(close_price),
            "最大上昇幅MFE(%)": (high_after / ref_price - 1) * 100,
            "最大下落幅MAE(%)": (low_after / ref_price - 1) * 100,
        }

        for minutes in entry_minutes:
            target = start + pd.Timedelta(minutes=minutes)
            pos = post.index.searchsorted(target, side="left")
            if pos >= len(post):
                entry_price = np.nan
                entry_time = "—"
                to_close = np.nan
                entry_mfe = np.nan
                entry_mae = np.nan
            else:
                entry_bar = post.iloc[pos]
                entry_ts = post.index[pos]
                entry_price = float(entry_bar["Open"])
                entry_time = entry_ts.strftime("%H:%M")
                remaining = post.loc[entry_ts:]
                to_close = (close_price / entry_price - 1) * 100
                entry_mfe = (float(remaining["High"].max()) / entry_price - 1) * 100
                entry_mae = (float(remaining["Low"].min()) / entry_price - 1) * 100
            label = "直後" if minutes == 0 else f"{minutes}分後"
            row[f"{label}エントリー時刻"] = entry_time
            row[f"{label}→引け(%)"] = to_close
            row[f"{label}後MFE(%)"] = entry_mfe
            row[f"{label}後MAE(%)"] = entry_mae

        move5 = row["5分後(%)"]
        move30 = row["30分後(%)"]
        close_move = row["引け時点(%)"]
        row["反応パターン"] = classify_intraday_pattern(move5, move30, close_move)
        row["初動プラス"] = bool(move5 > 0) if not pd.isna(move5) else False
        row["引けプラス"] = bool(close_move > 0) if not pd.isna(close_move) else False
        row["初動継続"] = bool(move5 > 0 and close_move > 0) if not pd.isna(move5) else False
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_entry_timings(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for label in ("直後", "5分後", "10分後", "15分後", "30分後", "60分後"):
        ret_col = f"{label}→引け(%)"
        mfe_col = f"{label}後MFE(%)"
        mae_col = f"{label}後MAE(%)"
        if ret_col not in detail.columns:
            continue
        valid = detail.dropna(subset=[ret_col]).copy()
        if valid.empty:
            continue
        n = len(valid)
        judgement, note = sample_judgement(n)
        mean_ret = valid[ret_col].mean()
        median_ret = valid[ret_col].median()
        win_rate = (valid[ret_col] > 0).mean() * 100
        mean_mae = valid[mae_col].mean() if mae_col in valid else np.nan
        mean_mfe = valid[mfe_col].mean() if mfe_col in valid else np.nan
        risk_adjusted = mean_ret - 0.35 * abs(mean_mae if not pd.isna(mean_mae) else 0)
        score = (mean_ret * 0.45 + median_ret * 0.25 + (win_rate - 50) / 10 * 0.15 + risk_adjusted * 0.15) * min(n / 8, 1)
        rows.append({
            "エントリー": label,
            "件数": n,
            "勝率(%)": win_rate,
            "平均→引け(%)": mean_ret,
            "中央値→引け(%)": median_ret,
            "平均MFE(%)": mean_mfe,
            "平均MAE(%)": mean_mae,
            "リスク調整値": risk_adjusted,
            "参考スコア": score,
            "判定": judgement,
            "サンプル注記": note,
        })
    return pd.DataFrame(rows).sort_values(["参考スコア", "件数"], ascending=False).reset_index(drop=True)

def summarize_intraday(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if detail.empty:
        return pd.DataFrame(), pd.DataFrame()

    quarter_rows = []
    for quarter, group in detail.groupby("四半期"):
        n = len(group)
        quarter_rows.append(
            {
                "四半期": quarter,
                "件数": n,
                "5分後上昇率(%)": group["初動プラス"].mean() * 100,
                "引け上昇率(%)": group["引けプラス"].mean() * 100,
                "初動継続率(%)": group["初動継続"].mean() * 100,
                "平均5分後(%)": group["5分後(%)"].mean(),
                "平均30分後(%)": group["30分後(%)"].mean(),
                "平均引け時点(%)": group["引け時点(%)"].mean(),
                "平均飛び乗り→引け(%)": group["発表後初値→引け(%)"].mean(),
                "平均MFE(%)": group["最大上昇幅MFE(%)"].mean(),
                "平均MAE(%)": group["最大下落幅MAE(%)"].mean(),
                "確信度": confidence_label(n),
            }
        )
    quarter_summary = pd.DataFrame(quarter_rows)
    quarter_summary["場中スコア"] = (
        quarter_summary["平均引け時点(%)"] * 0.40
        + quarter_summary["平均飛び乗り→引け(%)"] * 0.25
        + quarter_summary["平均30分後(%)"] * 0.15
        + (quarter_summary["引け上昇率(%)"] - 50) / 10 * 0.10
        + (quarter_summary["初動継続率(%)"] - 50) / 10 * 0.10
    ) * np.minimum(quarter_summary["件数"] / 6, 1)
    quarter_summary = quarter_summary.sort_values(["場中スコア", "件数"], ascending=False).reset_index(drop=True)

    pattern_summary = (
        detail["反応パターン"]
        .value_counts()
        .rename_axis("反応パターン")
        .reset_index(name="件数")
    )
    pattern_summary["構成比(%)"] = pattern_summary["件数"] / len(detail) * 100
    return quarter_summary, pattern_summary


# -----------------------------
# UI
# -----------------------------
st.title("📊 日本株 決算トレーダー分析 Pro")
st.caption("銘柄コードだけで決算日時・日足・分足を自動分析。発表直後〜60分後の各エントリーから引けまでの期待値を比較します。")

with st.sidebar:
    st.header("分析条件")
    raw_code = st.text_input("銘柄コード", value="7203", max_chars=8)
    years = st.slider("決算取得年数", 2, 10, 2)
    flat_threshold = st.number_input("横ばい判定幅（±%）", 0.0, 2.0, 0.2, 0.1)
    st.caption("Freeプランは取得期間・遅延の制限があります。")

st.markdown("## 分析モード")
st.info(
    "通常は銘柄コードを入力して『分析する』だけです。J-Quants分足アドオンが有効なら場中分析まで自動化します。未契約の場合だけ、後からCSV補完欄が表示されます。"
)
run = st.button("分析する", type="primary", use_container_width=True)

if run:
    try:
        code = normalize_code(raw_code)
        ticker = ticker_for(code)
        api_key = str(st.secrets.get("JQUANTS_API_KEY", "")).strip()
        cutoff = pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None).normalize() - pd.DateOffset(years=years)

        with st.spinner("決算日時・日足・分足の利用可否を確認しています…"):
            prices = load_daily_prices(ticker, years)
            events, jq_status = load_jquants_earnings(code, api_key)
            events = events[
                (events["earnings_date"] >= cutoff)
                & (events["earnings_date"] <= prices.index.max())
            ].copy()
            daily_detail = analyze_daily_events(prices, events, flat_threshold)
            carry_summary = summarize_daily(daily_detail, "決算跨ぎ対象")
            candidate_dates = intraday_candidate_dates(prices, events)
            auto_bars, minute_status, addon_available = load_jquants_minute_bars(
                code, api_key, candidate_dates
            )
            auto_intraday_detail = analyze_intraday_events(auto_bars, events) if not auto_bars.empty else pd.DataFrame()
            auto_intraday_summary, auto_pattern_summary = summarize_intraday(auto_intraday_detail)
            auto_entry_summary = summarize_entry_timings(auto_intraday_detail)

        st.session_state["analysis_bundle"] = {
            "code": code,
            "ticker": ticker,
            "events": events,
            "prices": prices,
            "daily_detail": daily_detail,
            "carry_summary": carry_summary,
            "jq_status": jq_status,
            "candidate_dates": candidate_dates,
            "auto_bars": auto_bars,
            "minute_status": minute_status,
            "addon_available": addon_available,
            "auto_intraday_detail": auto_intraday_detail,
            "auto_intraday_summary": auto_intraday_summary,
            "auto_pattern_summary": auto_pattern_summary,
            "auto_entry_summary": auto_entry_summary,
        }
    except Exception as exc:
        st.error(f"処理中にエラーが発生しました: {exc}")

bundle = st.session_state.get("analysis_bundle")
if bundle:
    code = bundle["code"]
    ticker = bundle["ticker"]
    events = bundle["events"]
    prices = bundle["prices"]
    daily_detail = bundle["daily_detail"]
    carry_summary = bundle["carry_summary"]
    candidate_dates = bundle["candidate_dates"]
    auto_bars = bundle["auto_bars"]
    auto_intraday_detail = bundle["auto_intraday_detail"]
    auto_intraday_summary = bundle["auto_intraday_summary"]
    auto_pattern_summary = bundle["auto_pattern_summary"]
    auto_entry_summary = bundle.get("auto_entry_summary", pd.DataFrame())
    addon_available = bundle["addon_available"]

    st.subheader(f"{code}（{ticker}）分析結果")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("採用決算", f"{len(daily_detail)}件")
    c2.metric("引け後・休場日", f"{int((daily_detail['分析区分'] == '決算跨ぎ対象').sum()) if not daily_detail.empty else 0}件")
    c3.metric("場中・昼休み", f"{int((daily_detail['分析区分'] == '場中分析対象').sum()) if not daily_detail.empty else 0}件")
    c4.metric("自動分足一致", f"{len(auto_intraday_detail)}件")

    with st.expander("取得状況", expanded=True):
        st.write(f"- {bundle['jq_status']}")
        st.write(f"- 日足最新日: {prices.index.max().strftime('%Y-%m-%d')}")
        st.write(f"- 場中候補日: {len(candidate_dates)}日")
        st.write(f"- {bundle['minute_status']}")
        if "403" in bundle["minute_status"] or "権限" in bundle["minute_status"]:
            st.info("契約直後は権限反映に時間がかかる場合があります。数分後に再度『分析する』を押してください。")
        elif not auto_bars.empty:
            st.success("分足API：利用可能")

    csv_intraday_detail = pd.DataFrame()
    csv_intraday_summary = pd.DataFrame()
    csv_pattern_summary = pd.DataFrame()
    csv_entry_summary = pd.DataFrame()
    csv_status = "CSV未選択"

    if auto_intraday_detail.empty:
        if addon_available:
            st.warning("分足APIは利用できますが、取得期間内の対象日に分足がありませんでした。必要ならCSVで補完してください。")
        else:
            st.warning("分足アドオン未契約のため、場中5分後・30分後・引け反応はCSV補完で分析できます。引け後分析は完全自動です。")

        with st.expander("📁 場中分足CSVで補完する（任意）", expanded=False):
            uploaded = st.file_uploader(
                "1分足・5分足CSVを選択",
                type=["csv"],
                help="datetime, open, high, low, close, volume の形式を推奨します。",
                key=f"fallback_csv_{code}",
            )
            sample_csv = (
                "datetime,open,high,low,close,volume\n"
                "2025-08-07 13:20:00,2500,2505,2498,2503,120000\n"
                "2025-08-07 13:25:00,2503,2520,2501,2518,450000\n"
            )
            st.download_button(
                "CSVテンプレート",
                data=sample_csv.encode("utf-8-sig"),
                file_name="sample_intraday.csv",
                mime="text/csv",
            )
            if uploaded is not None:
                try:
                    csv_bars, csv_status = parse_intraday_csv(uploaded)
                    csv_intraday_detail = analyze_intraday_events(csv_bars, events)
                    csv_intraday_summary, csv_pattern_summary = summarize_intraday(csv_intraday_detail)
                    csv_entry_summary = summarize_entry_timings(csv_intraday_detail)
                    st.success(f"{uploaded.name} を読み込みました：{csv_status} / 決算一致 {len(csv_intraday_detail)}件")
                except Exception as exc:
                    st.error(f"CSV解析エラー: {exc}")

    intraday_detail = auto_intraday_detail if not auto_intraday_detail.empty else csv_intraday_detail
    intraday_summary = auto_intraday_summary if not auto_intraday_detail.empty else csv_intraday_summary
    pattern_summary = auto_pattern_summary if not auto_intraday_detail.empty else csv_pattern_summary
    entry_summary = auto_entry_summary if not auto_intraday_detail.empty else csv_entry_summary
    intraday_source = "J-Quants分足API（自動）" if not auto_intraday_detail.empty else ("CSV補完" if not csv_intraday_detail.empty else "未取得")

    tab1, tab2, tab3 = st.tabs(["🔵 場中決算分析", "🟢 引け後決算分析", "📋 全決算明細"])

    with tab1:
        st.markdown("## 場中決算：発表直後から引けまで")
        st.caption(f"分足データ源：{intraday_source}")
        if intraday_detail.empty:
            st.info("分足データがないため、場中の5分後・30分後・引け反応は未分析です。引け後分析と発表時刻分類は利用できます。")
        else:
            top = intraday_summary.iloc[0]
            st.success(
                f"場中反応最上位：{top['四半期']}｜件数 {int(top['件数'])}｜"
                f"5分後上昇率 {top['5分後上昇率(%)']:.1f}%｜"
                f"引け上昇率 {top['引け上昇率(%)']:.1f}%｜"
                f"平均引け {top['平均引け時点(%)']:+.2f}%｜確信度 {top['確信度']}"
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("発表5分後 上昇率", f"{intraday_detail['初動プラス'].mean()*100:.1f}%")
            m2.metric("引けまで上昇率", f"{intraday_detail['引けプラス'].mean()*100:.1f}%")
            m3.metric("初動継続率", f"{intraday_detail['初動継続'].mean()*100:.1f}%")
            m4.metric("直後→引け平均", pct(intraday_detail['直後→引け(%)'].mean()))

            st.markdown("### 最適エントリー時間（発表後→引け）")
            if entry_summary.empty:
                st.info("エントリー時間別の比較データがありません。")
            else:
                best_entry = entry_summary.iloc[0]
                if int(best_entry["件数"]) < 3:
                    st.warning(
                        f"参考1位：{best_entry['エントリー']}｜平均 {best_entry['平均→引け(%)']:+.2f}%｜"
                        f"勝率 {best_entry['勝率(%)']:.1f}%｜件数 {int(best_entry['件数'])}｜判定保留"
                    )
                else:
                    st.success(
                        f"参考1位：{best_entry['エントリー']}｜平均 {best_entry['平均→引け(%)']:+.2f}%｜"
                        f"勝率 {best_entry['勝率(%)']:.1f}%｜平均MAE {best_entry['平均MAE(%)']:+.2f}%｜"
                        f"件数 {int(best_entry['件数'])}｜{best_entry['判定']}"
                    )
                edisplay = entry_summary.copy()
                enum = edisplay.select_dtypes(include=[np.number]).columns
                edisplay[enum] = edisplay[enum].round(2)
                st.dataframe(edisplay, use_container_width=True, hide_index=True)
                st.caption("平均MAEはエントリー後の平均最大逆行幅です。損切り-2%の運用では特に確認してください。")

            st.markdown("### 四半期別の場中反応")
            display = intraday_summary.copy()
            num = display.select_dtypes(include=[np.number]).columns
            display[num] = display[num].round(2)
            st.dataframe(display, use_container_width=True, hide_index=True)

            st.markdown("### 反応パターン")
            pdisplay = pattern_summary.copy()
            pdisplay["構成比(%)"] = pdisplay["構成比(%)"].round(1)
            st.dataframe(pdisplay, use_container_width=True, hide_index=True)
            st.bar_chart(pdisplay.set_index("反応パターン")["件数"])

            if not entry_summary.empty:
                st.download_button(
                    "エントリー時間ランキングCSVをダウンロード",
                    entry_summary.to_csv(index=False).encode("utf-8-sig"),
                    f"{code}_entry_timing_ranking.csv",
                    "text/csv",
                    use_container_width=True,
                )

            st.markdown("### 決算ごとの場中明細")
            ddisplay = intraday_detail.sort_values("決算発表日", ascending=False).copy()
            num = ddisplay.select_dtypes(include=[np.number]).columns
            ddisplay[num] = ddisplay[num].round(2)
            st.dataframe(ddisplay, use_container_width=True, hide_index=True)
            st.download_button(
                "場中分析CSVをダウンロード",
                intraday_detail.to_csv(index=False).encode("utf-8-sig"),
                f"{code}_intraday_earnings.csv",
                "text/csv",
                use_container_width=True,
            )

    with tab2:
        st.markdown("## 引け後決算：翌営業日のGU・終値")
        if carry_summary.empty:
            st.warning("取得期間内に引け後・休場日発表がないか、件数が不足しています。")
        else:
            best = carry_summary.iloc[0]
            st.success(
                f"決算跨ぎ最上位：{best['四半期']}｜評価 {best['評価']}｜"
                f"勝率 {best['勝率(%)']:.1f}%｜GU率 {best['GU率(%)']:.1f}%｜"
                f"平均翌日終値 {best['平均翌日終値(%)']:+.2f}%｜確信度 {best['確信度']}"
            )
            display = carry_summary.copy()
            num = display.select_dtypes(include=[np.number]).columns
            display[num] = display[num].round(2)
            st.dataframe(display, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("## 決算日時・日足反応一覧")
        if daily_detail.empty:
            st.error("分析可能な決算データがありません。")
        else:
            display = daily_detail.sort_values("決算発表日", ascending=False).copy()
            num = display.select_dtypes(include=[np.number]).columns
            display[num] = display[num].round(2)
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.download_button(
                "全決算明細CSVをダウンロード",
                daily_detail.to_csv(index=False).encode("utf-8-sig"),
                f"{code}_earnings_all.csv",
                "text/csv",
                use_container_width=True,
            )

with st.expander("場中パターンの定義"):
    st.markdown(
        "- **素直上昇型**：発表5分後がプラスで、引けもプラスを維持\n"
        "- **上昇失速型**：初動は上昇したが、引けにかけて上昇幅を大きく縮小\n"
        "- **V字回復型**：発表5分後はマイナスだが、引けはプラス\n"
        "- **行って来い型**：発表5分後はプラスだが、引けはゼロ以下\n"
        "- **素直下落型**：発表5分後も引けもマイナス\n"
        "- **MFE**：発表前基準価格から引けまでの最大上昇幅\n"
        "- **MAE**：発表前基準価格から引けまでの最大下落幅"
    )
