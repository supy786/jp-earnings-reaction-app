from __future__ import annotations

import io
import re
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
        first_open = float(post.iloc[0]["Open"])
        close_price = float(day_bars.iloc[-1]["Close"])
        high_after = float(post["High"].max())
        low_after = float(post["Low"].min())

        horizon_values: dict[int, float | None] = {}
        for minutes in (5, 15, 30, 60):
            horizon_values[minutes] = value_at_or_after(day_bars, start + pd.Timedelta(minutes=minutes))

        def move(value: float | None) -> float:
            return (value / ref_price - 1) * 100 if value is not None else np.nan

        move5 = move(horizon_values[5])
        move15 = move(horizon_values[15])
        move30 = move(horizon_values[30])
        move60 = move(horizon_values[60])
        close_move = move(close_price)
        entry_to_close = (close_price / first_open - 1) * 100
        mfe = (high_after / ref_price - 1) * 100
        mae = (low_after / ref_price - 1) * 100
        pattern = classify_intraday_pattern(move5, move30, close_move)

        rows.append(
            {
                "決算発表日": day.date(),
                "四半期": event["quarter"],
                "発表時刻": clock,
                "発表区分": session,
                "基準価格": ref_price,
                "発表後初値": first_open,
                "5分後(%)": move5,
                "15分後(%)": move15,
                "30分後(%)": move30,
                "60分後(%)": move60,
                "引け時点(%)": close_move,
                "発表後初値→引け(%)": entry_to_close,
                "最大上昇幅MFE(%)": mfe,
                "最大下落幅MAE(%)": mae,
                "反応パターン": pattern,
                "初動プラス": move5 > 0 if not pd.isna(move5) else False,
                "引けプラス": close_move > 0 if not pd.isna(close_move) else False,
                "初動継続": (move5 > 0 and close_move > 0) if not pd.isna(move5) else False,
            }
        )
    return pd.DataFrame(rows)


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
st.title("📊 日本株 決算トレーダー分析")
st.caption("引け後決算のオーバーナイト分析と、場中決算の発表直後〜引けまでの反応分析を1つに統合します。")

with st.sidebar:
    st.header("分析条件")
    raw_code = st.text_input("銘柄コード", value="7203", max_chars=8)
    years = st.slider("決算取得年数", 2, 10, 2)
    flat_threshold = st.number_input("横ばい判定幅（±%）", 0.0, 2.0, 0.2, 0.1)

st.markdown("## ① 分足CSVを選択（場中分析をする場合）")
st.info(
    "引け後決算の分析だけならCSVなしでも使えます。場中決算の5分後・30分後・引け反応を調べる場合は、1分足または5分足CSVを選択してください。"
)

left_upload, right_help = st.columns([2, 1])
with left_upload:
    intraday_file = st.file_uploader(
        "📁 1分足・5分足CSVをここで選択",
        type=["csv"],
        help="datetime, open, high, low, close, volume の形式を推奨します。",
        key="intraday_csv_main",
    )
    if intraday_file is not None:
        st.success(f"CSVを選択しました：{intraday_file.name}")
    else:
        st.caption("未選択：引け後分析のみ実行できます。")

with right_help:
    sample_csv = (
        "datetime,open,high,low,close,volume\n"
        "2025-08-07 13:20:00,2500,2505,2498,2503,120000\n"
        "2025-08-07 13:25:00,2503,2520,2501,2518,450000\n"
    )
    st.download_button(
        "⬇️ CSVテンプレート",
        data=sample_csv.encode("utf-8-sig"),
        file_name="sample_intraday.csv",
        mime="text/csv",
        use_container_width=True,
    )
    with st.expander("対応する列名"):
        st.markdown(
            "- 英語：`datetime, open, high, low, close, volume`\n"
            "- 日本語：`日時, 始値, 高値, 安値, 終値, 出来高`\n"
            "- 日本時間・時刻の古い順を推奨"
        )

st.markdown("## ② 分析を開始")
run = st.button("分析する", type="primary", use_container_width=True)

if run:
    try:
        code = normalize_code(raw_code)
        ticker = ticker_for(code)
        api_key = str(st.secrets.get("JQUANTS_API_KEY", "")).strip()
        cutoff = pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None).normalize() - pd.DateOffset(years=years)

        with st.spinner("決算日時と株価を集計しています…"):
            prices = load_daily_prices(ticker, years)
            events, jq_status = load_jquants_earnings(code, api_key)
            events = events[
                (events["earnings_date"] >= cutoff)
                & (events["earnings_date"] <= prices.index.max())
            ].copy()
            daily_detail = analyze_daily_events(prices, events, flat_threshold)
            carry_summary = summarize_daily(daily_detail, "決算跨ぎ対象")
            intraday_bars, intraday_status = parse_intraday_csv(intraday_file)
            intraday_detail = analyze_intraday_events(intraday_bars, events) if not intraday_bars.empty else pd.DataFrame()
            intraday_summary, pattern_summary = summarize_intraday(intraday_detail)

        st.subheader(f"{code}（{ticker}）分析結果")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("採用決算", f"{len(daily_detail)}件")
        c2.metric("引け後・休場日", f"{int((daily_detail['分析区分'] == '決算跨ぎ対象').sum()) if not daily_detail.empty else 0}件")
        c3.metric("場中・昼休み", f"{int((daily_detail['分析区分'] == '場中分析対象').sum()) if not daily_detail.empty else 0}件")
        c4.metric("場中CSV一致", f"{len(intraday_detail)}件")

        with st.expander("取得状況"):
            st.write(f"- {jq_status}")
            st.write(f"- 日足最新日: {prices.index.max().strftime('%Y-%m-%d')}")
            st.write(f"- 場中データ: {intraday_status}")

        tab1, tab2, tab3 = st.tabs(["🔵 場中決算分析", "🟢 引け後決算分析", "📋 全決算明細"])

        with tab1:
            st.markdown("## 場中決算：発表直後から引けまで")
            if intraday_file is None:
                st.warning("場中分析には1分足または5分足CSVを選択してください。決算日時は自動取得済みです。")
            elif intraday_detail.empty:
                st.warning(
                    "CSVと決算発表日が一致しませんでした。CSVに対象日の分足が含まれるか、日時列が日本時間か確認してください。"
                )
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
                m4.metric("飛び乗り→引け平均", pct(intraday_detail['発表後初値→引け(%)'].mean()))

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

    except Exception as exc:
        st.error(f"処理中にエラーが発生しました: {exc}")

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
