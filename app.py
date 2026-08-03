from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="日本株 決算翌営業日分析", page_icon="📊", layout="wide")


@dataclass(frozen=True)
class Settings:
    years: int
    flat_threshold: float
    fiscal_year_end_month: int


def normalize_code(raw: str) -> str:
    code = raw.strip().upper().replace(".T", "")
    if not code:
        raise ValueError("銘柄コードを入力してください。")
    if not code.isalnum():
        raise ValueError("銘柄コードは半角英数字で入力してください。")
    return code


def to_tse_ticker(code: str) -> str:
    return f"{code}.T"


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(ticker: str, years: int) -> pd.DataFrame:
    start = pd.Timestamp.today(tz="Asia/Tokyo").tz_localize(None) - pd.DateOffset(years=years, months=8)
    end = pd.Timestamp.today(tz="Asia/Tokyo").tz_localize(None) + pd.Timedelta(days=2)
    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise RuntimeError("株価データを取得できませんでした。コードまたは通信状況を確認してください。")
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


@st.cache_data(ttl=3600, show_spinner=False)
def load_auto_earnings(ticker: str, limit: int) -> pd.DataFrame:
    obj = yf.Ticker(ticker)
    try:
        df = obj.get_earnings_dates(limit=limit)
    except Exception:
        return pd.DataFrame(columns=["earnings_date", "source"])
    if df is None or df.empty:
        return pd.DataFrame(columns=["earnings_date", "source"])
    idx = pd.to_datetime(df.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("Asia/Tokyo").tz_localize(None)
    result = pd.DataFrame({"earnings_date": idx.normalize()})
    result = result.dropna().drop_duplicates("earnings_date").sort_values("earnings_date")
    result["source"] = "Yahoo Finance 自動取得"
    return result.reset_index(drop=True)


def parse_uploaded_csv(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame(columns=["earnings_date", "quarter", "announcement_time", "source"])
    raw = uploaded_file.getvalue()
    decoded = None
    for enc in ("utf-8-sig", "cp932", "utf-8"):
        try:
            decoded = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        raise ValueError("CSVの文字コードを読み取れませんでした。UTF-8またはShift-JISを使用してください。")
    df = pd.read_csv(io.StringIO(decoded))
    if "earnings_date" not in df.columns:
        raise ValueError("CSVには earnings_date 列が必要です。")
    out = pd.DataFrame()
    out["earnings_date"] = pd.to_datetime(df["earnings_date"], errors="coerce").dt.normalize()
    out["quarter"] = df["quarter"].astype(str) if "quarter" in df.columns else ""
    out["announcement_time"] = df["announcement_time"].astype(str) if "announcement_time" in df.columns else ""
    out["source"] = df["source"].astype(str) if "source" in df.columns else "CSV"
    out = out.dropna(subset=["earnings_date"])
    return out


def parse_manual_dates(text: str) -> pd.DataFrame:
    rows: list[dict] = []
    for line in text.splitlines():
        value = line.strip()
        if not value:
            continue
        parts = [p.strip() for p in value.split(",")]
        dt = pd.to_datetime(parts[0], errors="coerce")
        if pd.isna(dt):
            continue
        rows.append(
            {
                "earnings_date": dt.normalize(),
                "quarter": parts[1] if len(parts) >= 2 else "",
                "announcement_time": parts[2] if len(parts) >= 3 else "",
                "source": "手入力",
            }
        )
    return pd.DataFrame(rows, columns=["earnings_date", "quarter", "announcement_time", "source"])


def infer_quarter(announcement_date: pd.Timestamp, fiscal_year_end_month: int) -> str:
    # 発表日より20～150日前にある四半期末候補のうち、最も近い日を対象期間末とみなす。
    candidates: list[pd.Timestamp] = []
    for year in range(announcement_date.year - 2, announcement_date.year + 1):
        for month in range(1, 13):
            if month in {
                fiscal_year_end_month,
                ((fiscal_year_end_month - 3 - 1) % 12) + 1,
                ((fiscal_year_end_month - 6 - 1) % 12) + 1,
                ((fiscal_year_end_month - 9 - 1) % 12) + 1,
            }:
                candidates.append(pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0))
    valid = [d for d in candidates if timedelta(days=20) <= announcement_date - d <= timedelta(days=150)]
    if not valid:
        return "不明"
    period_end = max(valid)
    months_back = (fiscal_year_end_month - period_end.month) % 12
    mapping = {0: "本決算", 9: "1Q", 6: "2Q", 3: "3Q"}
    return mapping.get(months_back, "不明")


def combine_earnings(auto: pd.DataFrame, uploaded: pd.DataFrame, manual: pd.DataFrame,
                      fy_end_month: int, cutoff: pd.Timestamp) -> pd.DataFrame:
    frames = []
    if not auto.empty:
        a = auto.copy()
        a["quarter"] = ""
        a["announcement_time"] = ""
        frames.append(a)
    if not uploaded.empty:
        frames.append(uploaded)
    if not manual.empty:
        frames.append(manual)
    if not frames:
        return pd.DataFrame(columns=["earnings_date", "quarter", "announcement_time", "source"])
    all_dates = pd.concat(frames, ignore_index=True)
    all_dates["earnings_date"] = pd.to_datetime(all_dates["earnings_date"]).dt.normalize()
    all_dates = all_dates[all_dates["earnings_date"] >= cutoff]
    # 同じ日付は手入力 > CSV > 自動取得の順で優先
    priority = {"手入力": 3, "CSV": 2, "Yahoo Finance 自動取得": 1}
    all_dates["_priority"] = all_dates["source"].map(priority).fillna(2)
    all_dates = all_dates.sort_values(["earnings_date", "_priority"]).drop_duplicates("earnings_date", keep="last")
    all_dates["quarter"] = all_dates.apply(
        lambda r: r["quarter"] if str(r["quarter"]).strip() in {"1Q", "2Q", "3Q", "本決算"}
        else infer_quarter(r["earnings_date"], fy_end_month), axis=1
    )
    return all_dates.drop(columns="_priority").sort_values("earnings_date").reset_index(drop=True)


def next_trading_day(index: pd.DatetimeIndex, target: pd.Timestamp, strictly_after: bool = False):
    pos = index.searchsorted(target, side="right" if strictly_after else "left")
    if pos >= len(index):
        return None
    return index[pos]


def analyze_events(prices: pd.DataFrame, events: pd.DataFrame, flat_threshold: float) -> pd.DataFrame:
    rows: list[dict] = []
    idx = prices.index
    for _, event in events.iterrows():
        announced = pd.Timestamp(event["earnings_date"]).normalize()
        event_day = next_trading_day(idx, announced, strictly_after=False)
        if event_day is None:
            continue
        # 土日発表などは、最初の取引日を「当日扱い」にせず注記する。
        is_same_calendar_day = event_day == announced
        prev_pos = idx.get_loc(event_day) - 1
        next_pos = idx.get_loc(event_day) + 1
        if prev_pos < 0 or next_pos >= len(idx):
            continue
        prev_day = idx[prev_pos]
        next_day = idx[next_pos]
        prev_close = float(prices.loc[prev_day, "Close"])
        event_open = float(prices.loc[event_day, "Open"])
        event_close = float(prices.loc[event_day, "Close"])
        next_open = float(prices.loc[next_day, "Open"])
        next_close = float(prices.loc[next_day, "Close"])
        event_change = (event_close / prev_close - 1) * 100
        next_change_vs_event_close = (next_close / event_close - 1) * 100
        next_change_vs_preclose = (next_close / prev_close - 1) * 100
        gu = (next_open / event_close - 1) * 100
        reaction = "上昇" if next_change_vs_event_close > flat_threshold else (
            "下落" if next_change_vs_event_close < -flat_threshold else "横ばい"
        )
        rows.append({
            "決算発表日": announced.date(),
            "発表日取引": "通常" if is_same_calendar_day else "休場日発表",
            "四半期": event["quarter"],
            "発表時刻": event.get("announcement_time", "") or "不明",
            "データ源": event["source"],
            "当日取引日": event_day.date(),
            "翌営業日": next_day.date(),
            "決算当日騰落率(%)": event_change,
            "翌日GU率(%)": gu,
            "翌日騰落率(前日終値比)(%)": next_change_vs_event_close,
            "決算前終値→翌日終値(%)": next_change_vs_preclose,
            "翌日判定": reaction,
            "当日出来高": int(prices.loc[event_day, "Volume"]),
            "翌日出来高": int(prices.loc[next_day, "Volume"]),
        })
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    order = ["1Q", "2Q", "3Q", "本決算", "不明"]
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
            "平均・翌日騰落率(%)": g["翌日騰落率(前日終値比)(%)"].mean(),
            "中央値・翌日騰落率(%)": g["翌日騰落率(前日終値比)(%)"].median(),
        })
    result = pd.DataFrame(rows)
    result["_order"] = result["四半期"].map({v: i for i, v in enumerate(order)}).fillna(99)
    return result.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def score_quarters(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    # 小標本の過大評価を抑えるため、件数による信頼係数を付ける。
    reliability = np.minimum(out["件数"] / 5.0, 1.0)
    raw = (
        out["平均・翌日騰落率(%)"] * 0.45
        + out["中央値・翌日騰落率(%)"] * 0.20
        + (out["上昇率(%)"] - 50) / 10 * 0.20
        + (out["GU率(%)"] - 50) / 10 * 0.15
    ) * reliability
    out["参考スコア"] = raw.round(2)
    rank = raw.rank(method="min", ascending=False)
    out["決算跨ぎ評価"] = rank.map(lambda r: "S" if r == 1 else ("A" if r == 2 else ("B" if r == 3 else "C")))
    out["確信度"] = out["件数"].map(lambda n: "高" if n >= 8 else ("中" if n >= 5 else "低"))
    return out


st.title("📊 日本株 決算当日・翌営業日リアクション分析")
st.caption("銘柄コードを入力すると、過去の決算発表日と日足を突合し、四半期別の短期反応を集計します。")

with st.sidebar:
    st.header("分析条件")
    raw_code = st.text_input("銘柄コード", value="4203", max_chars=8)
    years = st.slider("分析年数", min_value=2, max_value=10, value=8)
    fy_end_month = st.selectbox("決算月", options=list(range(1, 13)), index=2, format_func=lambda x: f"{x}月")
    flat_threshold = st.number_input("横ばい判定幅（±%）", min_value=0.0, max_value=2.0, value=0.2, step=0.1)
    st.divider()
    st.subheader("決算日の補完（任意）")
    uploaded = st.file_uploader("決算日CSV", type=["csv"], help="必須列: earnings_date。任意列: quarter, announcement_time, source")
    manual_text = st.text_area(
        "手入力",
        placeholder="2025-08-05,1Q,15:00\n2025-11-07,2Q,15:00",
        help="1行1件。日付,四半期,発表時刻 の順。四半期と時刻は省略可能です。",
    )
    run = st.button("分析する", type="primary", use_container_width=True)

st.info(
    "重要：『決算当日騰落率』は前営業日終値→決算発表日の終値です。引け後発表の場合、決算内容を反映する主指標は翌営業日のGU率・騰落率です。発表時刻が不明なデータは因果関係を断定しません。"
)

if run:
    try:
        code = normalize_code(raw_code)
        ticker = to_tse_ticker(code)
        settings = Settings(years=years, flat_threshold=flat_threshold, fiscal_year_end_month=fy_end_month)
        cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)

        with st.spinner("株価と決算日を取得・集計しています…"):
            prices = load_prices(ticker, years)
            auto = load_auto_earnings(ticker, limit=max(40, years * 5))
            csv_events = parse_uploaded_csv(uploaded)
            manual_events = parse_manual_dates(manual_text)
            events = combine_earnings(auto, csv_events, manual_events, fy_end_month, cutoff)
            detail = analyze_events(prices, events, flat_threshold)
            summary = score_quarters(summarize(detail))

        st.subheader(f"{code}（{ticker}）分析結果")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("確認できた決算", f"{len(detail)}件")
        c2.metric("自動取得日", f"{len(auto)}件")
        c3.metric("分析期間", f"過去{years}年")
        c4.metric("最新株価日", prices.index.max().strftime("%Y-%m-%d"))

        if detail.empty:
            st.error("分析可能な決算日がありません。CSVまたは手入力で決算日を追加してください。")
        else:
            if summary.empty:
                st.warning("四半期別集計を作成できませんでした。")
            else:
                eligible = summary[summary["四半期"].isin(["1Q", "2Q", "3Q", "本決算"])]
                if not eligible.empty:
                    best = eligible.sort_values(["参考スコア", "件数"], ascending=False).iloc[0]
                    worst = eligible.sort_values(["参考スコア", "件数"], ascending=True).iloc[0]
                    st.success(
                        f"最も強い四半期（過去データ）：{best['四半期']} / 評価 {best['決算跨ぎ評価']} / "
                        f"平均翌日騰落率 {best['平均・翌日騰落率(%)']:+.2f}% / 確信度 {best['確信度']}"
                    )
                    st.warning(
                        f"最も弱い四半期（過去データ）：{worst['四半期']} / "
                        f"平均翌日騰落率 {worst['平均・翌日騰落率(%)']:+.2f}% / 確信度 {worst['確信度']}"
                    )

                st.markdown("### 四半期別集計")
                formatted_summary = summary.copy()
                numeric_cols = formatted_summary.select_dtypes(include=[np.number]).columns
                formatted_summary[numeric_cols] = formatted_summary[numeric_cols].round(2)
                st.dataframe(formatted_summary, use_container_width=True, hide_index=True)

                chart_df = summary[summary["四半期"].isin(["1Q", "2Q", "3Q", "本決算"])].set_index("四半期")
                if not chart_df.empty:
                    st.markdown("### 平均反応")
                    st.bar_chart(chart_df[["平均・翌日GU率(%)", "平均・翌日騰落率(%)"]])

            st.markdown("### 決算ごとの明細")
            display = detail.sort_values("決算発表日", ascending=False).copy()
            pct_cols = [c for c in display.columns if "(%)" in c]
            display[pct_cols] = display[pct_cols].round(2)
            st.dataframe(display, use_container_width=True, hide_index=True)

            st.download_button(
                "明細CSVをダウンロード",
                data=detail.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{code}_earnings_reaction_detail.csv",
                mime="text/csv",
            )
            st.download_button(
                "四半期集計CSVをダウンロード",
                data=summary.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{code}_earnings_reaction_summary.csv",
                mime="text/csv",
            )

            unknown_times = (detail["発表時刻"] == "不明").sum()
            inferred = (detail["四半期"] == "不明").sum()
            st.markdown("### データ品質")
            st.write(f"- 発表時刻不明：{unknown_times}件")
            st.write(f"- 四半期判定不明：{inferred}件")
            st.write("- 自動取得した決算日は必ず企業IR・TDnetと照合してください。特に発表時刻はYahoo Financeだけでは十分に確認できません。")

    except Exception as exc:
        st.error(str(exc))

with st.expander("CSV形式を見る"):
    st.code(
        "earnings_date,quarter,announcement_time,source\n"
        "2025-08-05,1Q,15:00,企業IR\n"
        "2025-11-07,2Q,15:00,TDnet",
        language="text",
    )

with st.expander("計算定義"):
    st.markdown(
        "- **決算当日騰落率**：決算発表日の終値 ÷ 前営業日終値 − 1\n"
        "- **翌日GU率**：翌営業日の始値 ÷ 決算発表日の終値 − 1\n"
        "- **翌日騰落率**：翌営業日の終値 ÷ 決算発表日の終値 − 1\n"
        "- **決算前終値→翌日終値**：翌営業日の終値 ÷ 決算発表日前営業日の終値 − 1\n"
        "- 四半期は決算月から自動推定します。CSV・手入力で指定した値を優先します。"
    )
