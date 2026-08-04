from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup

st.set_page_config(page_title="日本株 決算四半期リアクション分析", page_icon="📊", layout="wide")

USER_AGENT = (
    "Mozilla/5.0 (iPad; CPU OS 18_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1"
)
VALID_QUARTERS = {"1Q", "2Q", "3Q", "本決算"}


@dataclass(frozen=True)
class Settings:
    years: int
    flat_threshold: float
    fiscal_year_end_month: int


def normalize_code(raw: str) -> str:
    code = raw.strip().upper().replace(".T", "")
    if not code:
        raise ValueError("銘柄コードを入力してください。")
    if not re.fullmatch(r"[0-9A-Z]{4,6}", code):
        raise ValueError("銘柄コードは半角英数字4〜6文字で入力してください。")
    return code


def to_tse_ticker(code: str) -> str:
    return f"{code}.T"


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(ticker: str, years: int) -> pd.DataFrame:
    today = pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None).normalize()
    start = today - pd.DateOffset(years=years, months=9)
    end = today + pd.Timedelta(days=3)
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise RuntimeError("株価データを取得できませんでした。銘柄コードまたは通信状況を確認してください。")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"株価データの列が不足しています: {', '.join(missing)}")
    out = df[required].copy()
    out.index = pd.to_datetime(out.index).tz_localize(None).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out




def _extract_records(payload):
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ("data", "statements", "fin_summary", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
    for value in payload.values():
        if isinstance(value, list) and (not value or isinstance(value[0], dict)):
            return value
    return []


def _pick(record, *names):
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return ""


def _normalize_jq_quarter(value: str) -> str:
    t = re.sub(r"\s+", "", str(value)).upper()
    if t in {"1Q", "Q1", "1"} or "第1四半期" in t:
        return "1Q"
    if t in {"2Q", "Q2", "2", "HY", "H1"} or "第2四半期" in t or "中間" in t:
        return "2Q"
    if t in {"3Q", "Q3", "3"} or "第3四半期" in t:
        return "3Q"
    if t in {"FY", "4Q", "Q4", "4", "FULLYEAR", "ANNUAL"} or "通期" in t or "本決算" in t:
        return "本決算"
    return ""


@st.cache_data(ttl=21600, show_spinner=False)
def load_jquants_earnings(code: str, api_key: str) -> tuple[pd.DataFrame, str]:
    """J-Quants財務サマリーから「実際の四半期決算」だけを取得する。

    業績予想修正・配当予想修正・その他期間・同日重複を除外する。
    DocTypeを最優先に四半期を判定し、同日同四半期に複数資料がある場合は
    連結資料を優先して1件へ統合する。
    """
    cols = ["earnings_date", "quarter", "announcement_time", "source"]
    if not api_key:
        return pd.DataFrame(columns=cols), "J-Quants: APIキー未設定"

    jq_code = code if len(code) == 5 else f"{code}0"
    url = "https://api.jquants.com/v2/fins/summary"
    try:
        response = requests.get(
            url,
            params={"code": jq_code},
            headers={"x-api-key": api_key, "Accept": "application/json"},
            timeout=30,
        )
        response.raise_for_status()
        records = _extract_records(response.json())
    except Exception as exc:
        return pd.DataFrame(columns=cols), f"J-Quants取得失敗: {type(exc).__name__}"

    def quarter_from_doc_type(doc_type: str) -> str:
        t = str(doc_type).strip()
        if t.startswith("1QFinancialStatements_"):
            return "1Q"
        if t.startswith("2QFinancialStatements_"):
            return "2Q"
        if t.startswith("3QFinancialStatements_"):
            return "3Q"
        if t.startswith("FYFinancialStatements_"):
            return "本決算"
        return ""

    def document_priority(doc_type: str) -> int:
        t = str(doc_type)
        score = 0
        if "Consolidated" in t and "NonConsolidated" not in t:
            score += 20
        if "IFRS" in t or "JP" in t or "JMIS" in t:
            score += 5
        return score

    raw_rows = []
    doc_type_counts: dict[str, int] = {}
    excluded = 0
    for rec in records:
        if not isinstance(rec, dict):
            continue
        doc_type = str(_pick(rec, "DocType", "TypeOfDocument", "document_type")).strip()
        doc_type_counts[doc_type or "(空欄)"] = doc_type_counts.get(doc_type or "(空欄)", 0) + 1

        # 決算短信に対応する4種類だけを採用。修正開示やその他期間は除外。
        quarter = quarter_from_doc_type(doc_type)
        if not quarter:
            excluded += 1
            continue

        dt = pd.to_datetime(
            _pick(rec, "DiscDate", "DisclosedDate", "disclosed_date", "Date"),
            errors="coerce",
        )
        if pd.isna(dt):
            excluded += 1
            continue

        tm = str(_pick(rec, "DiscTime", "DisclosedTime", "disclosed_time")).strip()
        if not re.fullmatch(r"(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?", tm):
            tm = ""
        elif len(tm) == 8:
            tm = tm[:5]

        raw_rows.append({
            "earnings_date": pd.Timestamp(dt).normalize(),
            "quarter": quarter,
            "announcement_time": tm,
            "source": "J-Quants 財務情報",
            "_doc_type": doc_type,
            "_priority": document_priority(doc_type),
        })

    if not raw_rows:
        type_summary = ", ".join(f"{k}:{v}" for k, v in sorted(doc_type_counts.items())[:6])
        suffix = f" / 種別 {type_summary}" if type_summary else ""
        return pd.DataFrame(columns=cols), f"J-Quants: 0件（応答{len(records)}件・除外{excluded}件{suffix}）"

    raw = pd.DataFrame(raw_rows)
    before = len(raw)
    # 同日・同四半期の連結／単体や訂正由来の重複を1件へ統合。
    out = (
        raw.sort_values(
            ["earnings_date", "quarter", "_priority", "announcement_time"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["earnings_date", "quarter"], keep="first")
        .sort_values("earnings_date")
        .reset_index(drop=True)
    )
    duplicate_count = before - len(out)
    out = out[cols]
    q_counts = out["quarter"].value_counts().reindex(["1Q", "2Q", "3Q", "本決算"], fill_value=0)
    q_text = " / ".join(f"{q}:{int(n)}件" for q, n in q_counts.items())
    status = (
        f"J-Quants: 採用{len(out)}件（API応答{len(records)}件 / "
        f"対象外除外{excluded}件 / 重複統合{duplicate_count}件）｜{q_text}"
    )
    return out, status


@st.cache_data(ttl=21600, show_spinner=False)
def load_irbank_earnings(code: str) -> tuple[pd.DataFrame, str]:
    """IRBANKの「決算発表資料」から四半期決算日を取得する。

    Streamlit Cloud等からIRBANK本体へ直接接続できない場合があるため、
    1. IRBANK本体HTML
    2. Jina Reader経由のMarkdown
    の順で取得する。取得できない日付を推測で補完しない。
    """

    cols = ["earnings_date", "quarter", "announcement_time", "source"]
    records: list[dict] = []
    diagnostics: list[str] = []

    def normalize_quarter(value: str) -> str:
        t = re.sub(r"\s+", "", str(value)).upper()
        if t in {"1Q", "第1四半期", "第一四半期"} or "第1四半期" in t:
            return "1Q"
        if t in {"2Q", "第2四半期", "第二四半期", "中間", "中間期"} or "第2四半期" in t:
            return "2Q"
        if t in {"3Q", "第3四半期", "第三四半期"} or "第3四半期" in t:
            return "3Q"
        if t in {"通期", "本決算", "年度決算"}:
            return "本決算"
        return ""

    def add_record(date_text: str, time_text: str, quarter_text: str, source: str) -> None:
        q = normalize_quarter(quarter_text)
        if not q:
            return
        dt = pd.to_datetime(str(date_text).strip(), errors="coerce")
        if pd.isna(dt):
            return
        tm = str(time_text).strip()
        if not re.fullmatch(r"(?:[01]?\d|2[0-3]):[0-5]\d", tm):
            tm = ""
        records.append({
            "earnings_date": pd.Timestamp(dt).normalize(),
            "quarter": q,
            "announcement_time": tm,
            "source": source,
        })

    # 1) IRBANK本体。表の列名を明示的に見る。
    direct_url = f"https://irbank.net/{code}"
    try:
        r = requests.get(
            direct_url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "ja,en-US;q=0.8,en;q=0.6",
                "Cache-Control": "no-cache",
            },
            timeout=25,
        )
        r.raise_for_status()
        diagnostics.append(f"直接取得HTTP {r.status_code}")
        try:
            tables = pd.read_html(io.StringIO(r.text))
        except Exception:
            tables = []
        for table in tables:
            table.columns = [
                " ".join(map(str, c)).strip() if isinstance(c, tuple) else str(c).strip()
                for c in table.columns
            ]
            colmap = {re.sub(r"\s+", "", c): c for c in table.columns}
            date_col = next((orig for key, orig in colmap.items() if key in {"提出日", "発表日", "日付"}), None)
            quarter_col = next((orig for key, orig in colmap.items() if key in {"区分", "四半期"}), None)
            time_col = next((orig for key, orig in colmap.items() if key == "時間"), None)
            if not date_col or not quarter_col:
                continue
            for _, row in table.iterrows():
                add_record(
                    row.get(date_col, ""),
                    row.get(time_col, "") if time_col else "",
                    row.get(quarter_col, ""),
                    "IRBANK 決算発表資料",
                )
    except Exception as exc:
        diagnostics.append(f"直接取得失敗:{type(exc).__name__}")

    # 2) 直接取得が0件ならJina Reader経由。Markdown表を正規表現で読む。
    if not records:
        jina_url = f"https://r.jina.ai/https://irbank.net/{code}"
        try:
            jr = requests.get(
                jina_url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/plain,text/markdown,*/*",
                    "X-Return-Format": "markdown",
                    "X-Timeout": "30",
                },
                timeout=45,
            )
            jr.raise_for_status()
            text = jr.text
            diagnostics.append(f"Jina取得HTTP {jr.status_code}")

            # 決算発表資料セクション以降だけを優先して誤検出を抑える。
            marker = text.find("決算発表資料")
            if marker >= 0:
                text = text[marker:]

            # 例: 2025/11/04 | 11:30 | 2Q | ...
            pattern = re.compile(
                r"(?m)^\s*(20\d{2}[/-]\d{1,2}[/-]\d{1,2})\s*\|\s*"
                r"((?:[01]?\d|2[0-3]):[0-5]\d)?\s*\|\s*"
                r"(1Q|2Q|3Q|通期|本決算)\s*\|"
            )
            for m in pattern.finditer(text):
                add_record(m.group(1), m.group(2), m.group(3), "IRBANK（Jina経由）")

            # Markdown化で縦棒が消えた場合の行単位フォールバック。
            if not records:
                line_pattern = re.compile(
                    r"(20\d{2}[/-]\d{1,2}[/-]\d{1,2}).{0,30}?"
                    r"((?:[01]?\d|2[0-3]):[0-5]\d).{0,20}?"
                    r"(?:\b(1Q|2Q|3Q)\b|(通期|本決算))"
                )
                for line in text.splitlines():
                    m = line_pattern.search(line)
                    if m:
                        add_record(m.group(1), m.group(2), m.group(3) or m.group(4), "IRBANK（Jina経由）")
        except Exception as exc:
            diagnostics.append(f"Jina取得失敗:{type(exc).__name__}")

    if not records:
        return pd.DataFrame(columns=cols), "IRBANK: 0件（" + " / ".join(diagnostics) + "）"

    result = pd.DataFrame(records, columns=cols)
    result = (
        result.drop_duplicates(["earnings_date", "quarter"], keep="first")
        .sort_values("earnings_date")
        .reset_index(drop=True)
    )
    return result, f"IRBANK: 四半期決算{len(result)}件（" + " / ".join(diagnostics) + "）"


@st.cache_data(ttl=21600, show_spinner=False)
def load_yahoo_earnings(ticker: str, limit: int) -> tuple[pd.DataFrame, str]:
    try:
        df = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    except Exception as exc:
        return pd.DataFrame(columns=["earnings_date", "source"]), f"Yahoo取得失敗: {type(exc).__name__}"
    if df is None or df.empty:
        return pd.DataFrame(columns=["earnings_date", "source"]), "Yahoo Finance: 0件"
    idx = pd.to_datetime(df.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Asia/Tokyo").tz_localize(None)
    result = pd.DataFrame({"earnings_date": idx.normalize()}).dropna().drop_duplicates()
    result["source"] = "Yahoo Finance 決算日"
    return result.sort_values("earnings_date").reset_index(drop=True), f"Yahoo Finance: {len(result)}件"


def parse_uploaded_csv(uploaded_file) -> pd.DataFrame:
    cols = ["earnings_date", "quarter", "announcement_time", "source"]
    if uploaded_file is None:
        return pd.DataFrame(columns=cols)
    raw = uploaded_file.getvalue()
    decoded = None
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            decoded = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        raise ValueError("CSVの文字コードを読み取れません。UTF-8またはShift-JISを使用してください。")
    df = pd.read_csv(io.StringIO(decoded))
    if "earnings_date" not in df.columns:
        raise ValueError("CSVには earnings_date 列が必要です。")
    out = pd.DataFrame()
    out["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce").dt.normalize()
    out["quarter"] = df["quarter"].astype(str) if "quarter" in df.columns else ""
    out["announcement_time"] = df["announcement_time"].astype(str) if "announcement_time" in df.columns else ""
    out["source"] = df["source"].astype(str) if "source" in df.columns else "CSV"
    return out.dropna(subset=["earnings_date"])


def parse_manual_dates(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        value = line.strip()
        if not value:
            continue
        parts = [p.strip() for p in value.split(",")]
        dt = pd.to_datetime(parts[0], errors="coerce")
        if pd.isna(dt):
            continue
        rows.append({
            "earnings_date": dt.normalize(),
            "quarter": parts[1] if len(parts) >= 2 else "",
            "announcement_time": parts[2] if len(parts) >= 3 else "",
            "source": "手入力",
        })
    return pd.DataFrame(rows, columns=["earnings_date", "quarter", "announcement_time", "source"])


def infer_quarter(announcement_date: pd.Timestamp, fiscal_year_end_month: int) -> str:
    candidates = []
    quarter_end_months = {
        fiscal_year_end_month,
        ((fiscal_year_end_month - 3 - 1) % 12) + 1,
        ((fiscal_year_end_month - 6 - 1) % 12) + 1,
        ((fiscal_year_end_month - 9 - 1) % 12) + 1,
    }
    for year in range(announcement_date.year - 2, announcement_date.year + 1):
        for month in quarter_end_months:
            candidates.append(pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0))
    valid = [d for d in candidates if timedelta(days=15) <= announcement_date - d <= timedelta(days=155)]
    if not valid:
        return "不明"
    period_end = max(valid)
    months_back = (fiscal_year_end_month - period_end.month) % 12
    return {0: "本決算", 9: "1Q", 6: "2Q", 3: "3Q"}.get(months_back, "不明")


def is_plausible_earnings_date(dt: pd.Timestamp, fy_end_month: int) -> bool:
    """発表月として極端に不自然な日付を除外する緩いフィルタ。"""
    q = infer_quarter(dt, fy_end_month)
    return q in VALID_QUARTERS


def combine_earnings(
    jquants: pd.DataFrame,
    irbank: pd.DataFrame,
    yahoo: pd.DataFrame,
    uploaded: pd.DataFrame,
    manual: pd.DataFrame,
    fy_end_month: int,
    cutoff: pd.Timestamp,
    latest_price_date: pd.Timestamp,
) -> pd.DataFrame:
    frames = []
    for frame in (jquants, irbank, yahoo):
        if not frame.empty:
            f = frame.copy()
            if "quarter" not in f.columns:
                f["quarter"] = ""
            if "announcement_time" not in f.columns:
                f["announcement_time"] = ""
            frames.append(f)
    if not uploaded.empty:
        frames.append(uploaded)
    if not manual.empty:
        frames.append(manual)
    if not frames:
        return pd.DataFrame(columns=["earnings_date", "quarter", "announcement_time", "source"])

    all_dates = pd.concat(frames, ignore_index=True)
    all_dates["earnings_date"] = pd.to_datetime(all_dates["earnings_date"], errors="coerce").dt.normalize()
    all_dates = all_dates.dropna(subset=["earnings_date"])
    all_dates = all_dates[
        (all_dates["earnings_date"] >= cutoff)
        & (all_dates["earnings_date"] <= latest_price_date)
    ]

    # 自動取得日は、決算発表時期として成立する候補だけ残す。
    manual_sources = {"手入力", "CSV"}
    auto_mask = ~all_dates["source"].astype(str).isin(manual_sources)
    all_dates = all_dates[(~auto_mask) | all_dates["earnings_date"].map(lambda x: is_plausible_earnings_date(x, fy_end_month))]

    priority = {
        "手入力": 50,
        "CSV": 40,
        "J-Quants 財務情報": 35,
        "IRBANK 決算発表履歴": 20,
        "Yahoo Finance 決算日": 10,
    }
    all_dates["_priority"] = all_dates["source"].map(priority).fillna(15)
    all_dates = (
        all_dates.sort_values(["earnings_date", "_priority"])
        .drop_duplicates("earnings_date", keep="last")
    )
    all_dates["quarter"] = all_dates.apply(
        lambda r: r["quarter"] if str(r["quarter"]).strip() in VALID_QUARTERS
        else infer_quarter(r["earnings_date"], fy_end_month),
        axis=1,
    )
    return all_dates.drop(columns="_priority").sort_values("earnings_date").reset_index(drop=True)


def trading_day_at_or_after(index: pd.DatetimeIndex, target: pd.Timestamp):
    pos = index.searchsorted(target, side="left")
    return None if pos >= len(index) else index[pos]


def analyze_events(prices: pd.DataFrame, events: pd.DataFrame, flat_threshold: float) -> pd.DataFrame:
    rows = []
    idx = prices.index
    for _, event in events.iterrows():
        announced = pd.Timestamp(event["earnings_date"]).normalize()
        event_day = trading_day_at_or_after(idx, announced)
        if event_day is None:
            continue
        same_calendar_day = event_day == announced
        pos = idx.get_loc(event_day)
        if pos < 1 or pos + 1 >= len(idx):
            continue
        prev_day, next_day = idx[pos - 1], idx[pos + 1]
        prev_close = float(prices.loc[prev_day, "Close"])
        event_open = float(prices.loc[event_day, "Open"])
        event_close = float(prices.loc[event_day, "Close"])
        next_open = float(prices.loc[next_day, "Open"])
        next_close = float(prices.loc[next_day, "Close"])

        event_change = (event_close / prev_close - 1) * 100
        next_change = (next_close / event_close - 1) * 100
        next_total = (next_close / prev_close - 1) * 100
        gu = (next_open / event_close - 1) * 100
        intraday = (next_close / next_open - 1) * 100
        reaction = "上昇" if next_change > flat_threshold else ("下落" if next_change < -flat_threshold else "横ばい")

        rows.append({
            "決算発表日": announced.date(),
            "発表日取引": "通常" if same_calendar_day else "休場日発表",
            "四半期": event["quarter"],
            "発表時刻": str(event.get("announcement_time", "")).strip() or "不明",
            "データ源": event["source"],
            "前営業日": prev_day.date(),
            "当日取引日": event_day.date(),
            "翌営業日": next_day.date(),
            "決算当日騰落率(%)": event_change,
            "翌日GU率(%)": gu,
            "翌日寄り後騰落率(%)": intraday,
            "翌日終値騰落率(%)": next_change,
            "決算前終値→翌日終値(%)": next_total,
            "翌日判定": reaction,
            "当日出来高": int(prices.loc[event_day, "Volume"]),
            "翌日出来高": int(prices.loc[next_day, "Volume"]),
        })
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for q, g in detail.groupby("四半期", dropna=False):
        n = len(g)
        up = int((g["翌日判定"] == "上昇").sum())
        down = int((g["翌日判定"] == "下落").sum())
        flat = int((g["翌日判定"] == "横ばい").sum())
        rows.append({
            "四半期": q,
            "件数": n,
            "上昇": up,
            "下落": down,
            "横ばい": flat,
            "上昇率(%)": up / n * 100,
            "平均・当日騰落率(%)": g["決算当日騰落率(%)"].mean(),
            "平均・翌日GU率(%)": g["翌日GU率(%)"].mean(),
            "GU回数": int((g["翌日GU率(%)"] > 0).sum()),
            "GU率(%)": (g["翌日GU率(%)"] > 0).mean() * 100,
            "平均・翌日寄り後(%)": g["翌日寄り後騰落率(%)"].mean(),
            "平均・翌日終値騰落率(%)": g["翌日終値騰落率(%)"].mean(),
            "中央値・翌日終値騰落率(%)": g["翌日終値騰落率(%)"].median(),
        })
    result = pd.DataFrame(rows)
    order = {"1Q": 0, "2Q": 1, "3Q": 2, "本決算": 3, "不明": 9}
    result["_order"] = result["四半期"].map(order).fillna(99)
    return result.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def score_quarters(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    reliability = np.minimum(out["件数"] / 6.0, 1.0)
    raw = (
        out["平均・翌日終値騰落率(%)"] * 0.40
        + out["中央値・翌日終値騰落率(%)"] * 0.20
        + out["平均・翌日GU率(%)"] * 0.20
        + (out["上昇率(%)"] - 50) / 10 * 0.12
        + (out["GU率(%)"] - 50) / 10 * 0.08
    ) * reliability
    out["参考スコア"] = raw.round(2)

    # 1件だけでS判定しない。2件未満は判定保留とする。
    eligible = out["四半期"].isin(VALID_QUARTERS) & (out["件数"] >= 2)
    ranks = out.loc[eligible, "参考スコア"].rank(method="min", ascending=False)
    out["決算跨ぎ評価"] = "判定保留"
    out.loc[eligible, "決算跨ぎ評価"] = ranks.map(
        lambda r: "S" if r == 1 else ("A" if r == 2 else ("B" if r == 3 else "C"))
    )
    out["確信度"] = out["件数"].map(lambda n: "高" if n >= 8 else ("中" if n >= 5 else "低"))
    return out


def source_counts(events: pd.DataFrame) -> str:
    if events.empty:
        return "なし"
    counts = events["source"].value_counts()
    return " / ".join(f"{name}: {count}件" for name, count in counts.items())


st.title("📊 日本株 決算当日・翌営業日リアクション分析 改良版")
st.caption("J-Quantsの決算短信だけを抽出し、修正開示・配当修正・同日重複を除外して短期反応を集計します。")

with st.sidebar:
    st.header("分析条件")
    raw_code = st.text_input("銘柄コード", value="4203", max_chars=8)
    years = st.slider("分析年数", 2, 10, 8)
    fy_end_month = st.selectbox("決算月", list(range(1, 13)), index=2, format_func=lambda x: f"{x}月")
    flat_threshold = st.number_input("横ばい判定幅（±%）", 0.0, 2.0, 0.2, 0.1)
    st.divider()
    st.subheader("補完データ（任意）")
    uploaded = st.file_uploader("決算日CSV", type=["csv"], help="必須列: earnings_date。任意列: quarter, announcement_time, source")
    manual_text = st.text_area(
        "手入力",
        placeholder="2025-08-05,1Q,15:00\n2025-11-07,2Q,15:00",
        help="日付,四半期,発表時刻。四半期と時刻は省略できます。",
    )
    run = st.button("分析する", type="primary", use_container_width=True)

st.info(
    "決算当日騰落率は前営業日終値→発表日終値です。引け後発表では、決算反応の中心は翌営業日のGU率・終値騰落率です。発表時刻不明データでは因果関係を断定しません。"
)

if run:
    try:
        api_key = str(st.secrets.get("JQUANTS_API_KEY", "")).strip()
        code = normalize_code(raw_code)
        ticker = to_tse_ticker(code)
        cutoff = pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None).normalize() - pd.DateOffset(years=years)

        with st.spinner("株価・決算発表日を取得して集計しています…"):
            prices = load_prices(ticker, years)
            jquants, jquants_status = load_jquants_earnings(code, api_key)
            irbank, irbank_status = load_irbank_earnings(code)
            yahoo, yahoo_status = load_yahoo_earnings(ticker, max(50, years * 6))
            csv_events = parse_uploaded_csv(uploaded)
            manual_events = parse_manual_dates(manual_text)
            events = combine_earnings(
                jquants, irbank, yahoo, csv_events, manual_events,
                fy_end_month, cutoff, prices.index.max(),
            )
            detail = analyze_events(prices, events, flat_threshold)
            summary = score_quarters(summarize(detail))

        st.subheader(f"{code}（{ticker}）分析結果")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("確認できた決算", f"{len(detail)}件")
        c2.metric("決算日候補", f"{len(events)}件")
        c3.metric("分析期間", f"過去{years}年")
        c4.metric("最新株価日", prices.index.max().strftime("%Y-%m-%d"))

        with st.expander("取得状況", expanded=detail.empty):
            st.write(f"- {jquants_status}")
            st.write(f"- {irbank_status}")
            st.write(f"- {yahoo_status}")
            st.write(f"- 採用データ: {source_counts(events)}")
            st.caption("J-Quantsを最優先し、IRBANK・Yahoo Financeは補完に使用します。Freeプランでは取得期間と遅延に制限があります。取得できない日付を推測で追加しません。")

        if detail.empty:
            st.error("分析可能な決算日を自動取得できませんでした。企業IR・TDnetで確認した日付をCSVまたは手入力で追加してください。")
        else:
            eligible = summary[summary["四半期"].isin(VALID_QUARTERS)] if not summary.empty else pd.DataFrame()
            if not eligible.empty:
                best = eligible.sort_values(["参考スコア", "件数"], ascending=False).iloc[0]
                worst = eligible.sort_values(["参考スコア", "件数"], ascending=True).iloc[0]
                st.success(
                    f"最も強い四半期：{best['四半期']}｜評価 {best['決算跨ぎ評価']}｜"
                    f"平均翌日終値 {best['平均・翌日終値騰落率(%)']:+.2f}%｜GU率 {best['GU率(%)']:.1f}%｜確信度 {best['確信度']}"
                )
                st.warning(
                    f"最も弱い四半期：{worst['四半期']}｜平均翌日終値 {worst['平均・翌日終値騰落率(%)']:+.2f}%｜"
                    f"GU率 {worst['GU率(%)']:.1f}%｜確信度 {worst['確信度']}"
                )

            st.markdown("### 四半期別集計")
            display_summary = summary.copy()
            num_cols = display_summary.select_dtypes(include=[np.number]).columns
            display_summary[num_cols] = display_summary[num_cols].round(2)
            st.dataframe(display_summary, use_container_width=True, hide_index=True)

            chart_df = summary[summary["四半期"].isin(VALID_QUARTERS)].set_index("四半期")
            if not chart_df.empty:
                st.markdown("### 平均反応")
                st.bar_chart(chart_df[["平均・翌日GU率(%)", "平均・翌日終値騰落率(%)"]])

            st.markdown("### 決算ごとの明細")
            display = detail.sort_values("決算発表日", ascending=False).copy()
            pct_cols = [c for c in display.columns if "(%)" in c]
            display[pct_cols] = display[pct_cols].round(2)
            st.dataframe(display, use_container_width=True, hide_index=True)

            col1, col2 = st.columns(2)
            col1.download_button(
                "明細CSVをダウンロード",
                detail.to_csv(index=False).encode("utf-8-sig"),
                f"{code}_earnings_reaction_detail.csv",
                "text/csv",
                use_container_width=True,
            )
            col2.download_button(
                "四半期集計CSVをダウンロード",
                summary.to_csv(index=False).encode("utf-8-sig"),
                f"{code}_earnings_reaction_summary.csv",
                "text/csv",
                use_container_width=True,
            )

            st.markdown("### データ品質")
            st.write(f"- 発表時刻不明：{int((detail['発表時刻'] == '不明').sum())}件")
            st.write(f"- 四半期判定不明：{int((detail['四半期'] == '不明').sum())}件")
            st.write("- 1件だけの四半期は決算跨ぎ評価を『判定保留』にします。")
            st.write("- 最終的な決算跨ぎ判断では、企業IR・TDnetで発表日時を照合してください。")

    except Exception as exc:
        st.error(f"処理中にエラーが発生しました: {exc}")

with st.expander("CSV形式を見る"):
    st.code(
        "earnings_date,quarter,announcement_time,source\n"
        "2025-08-05,1Q,15:00,企業IR\n"
        "2025-11-07,2Q,15:00,TDnet",
        language="text",
    )

with st.expander("計算定義"):
    st.markdown(
        "- **決算当日騰落率**：発表日終値 ÷ 前営業日終値 − 1\n"
        "- **翌日GU率**：翌営業日始値 ÷ 発表日終値 − 1\n"
        "- **翌日寄り後騰落率**：翌営業日終値 ÷ 翌営業日始値 − 1\n"
        "- **翌日終値騰落率**：翌営業日終値 ÷ 発表日終値 − 1\n"
        "- 四半期は決算月と発表日から推定し、CSV・手入力の指定を優先します。"
    )
