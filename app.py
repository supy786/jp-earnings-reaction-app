from __future__ import annotations

import io
import re
from datetime import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="日本株 決算四半期リアクション分析", page_icon="📊", layout="wide")

VALID_QUARTERS = ["1Q", "2Q", "3Q", "本決算"]
JQUANTS_URL = "https://api.jquants.com/v2/fins/summary"


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
        return text.zfill(5)
    return ""


def close_time_for(day: pd.Timestamp) -> time:
    # 東証は2024-11-05から取引終了時刻を15:30へ延長。
    if day.normalize() >= pd.Timestamp("2024-11-05"):
        return time(15, 30)
    return time(15, 0)


def classify_announcement(day: pd.Timestamp, clock_text: str, is_trading_day: bool) -> tuple[str, str, str]:
    """発表時刻を取引時間帯に分類する。

    戻り値: (発表区分, 決算跨ぎ適格, 主反応指標)
    """
    if not is_trading_day:
        return "休場日発表", "対象", "次営業日GU・終値"
    if not clock_text:
        return "時刻不明", "判定不能", "翌営業日反応（参考）"

    hour, minute = map(int, clock_text.split(":"))
    announced = time(hour, minute)
    close_time = close_time_for(day)

    if announced < time(9, 0):
        return "寄り前", "対象外", "当日始値・終値"
    if announced < time(11, 30):
        return "前場中", "対象外", "当日終値（参考）"
    if announced < time(12, 30):
        return "昼休み", "対象外", "当日後場・終値"
    if announced < close_time:
        return "後場中", "対象外", "当日終値（参考）"
    return "引け後", "対象", "翌営業日GU・終値"


def time_reliability(source: str, clock_text: str) -> str:
    if not clock_text:
        return "×"
    source_text = str(source)
    if "J-Quants" in source_text or "TDnet" in source_text:
        return "◎"
    if "企業IR" in source_text:
        return "◎"
    if source_text in {"CSV", "手入力"}:
        return "○"
    return "△"


@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(ticker: str, years: int) -> pd.DataFrame:
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
        raise RuntimeError("株価データを取得できませんでした。")
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


def parse_uploaded_csv(uploaded_file) -> pd.DataFrame:
    columns = ["earnings_date", "quarter", "announcement_time", "source", "doc_type"]
    if uploaded_file is None:
        return pd.DataFrame(columns=columns)
    raw = uploaded_file.getvalue()
    decoded = None
    for encoding in ("utf-8-sig", "cp932", "utf-8"):
        try:
            decoded = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        raise ValueError("CSVの文字コードを読み取れません。")
    frame = pd.read_csv(io.StringIO(decoded))
    if "earnings_date" not in frame.columns:
        raise ValueError("CSVには earnings_date 列が必要です。")
    out = pd.DataFrame()
    out["earnings_date"] = pd.to_datetime(frame["earnings_date"], errors="coerce").dt.normalize()
    out["quarter"] = frame["quarter"].astype(str) if "quarter" in frame.columns else ""
    out["announcement_time"] = (
        frame["announcement_time"].map(normalize_clock) if "announcement_time" in frame.columns else ""
    )
    out["source"] = frame["source"].astype(str) if "source" in frame.columns else "CSV"
    out["doc_type"] = frame["doc_type"].astype(str) if "doc_type" in frame.columns else ""
    return out.dropna(subset=["earnings_date"])


def parse_manual_dates(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if not parts or not parts[0]:
            continue
        disclosed = pd.to_datetime(parts[0], errors="coerce")
        if pd.isna(disclosed):
            continue
        rows.append(
            {
                "earnings_date": pd.Timestamp(disclosed).normalize(),
                "quarter": parts[1] if len(parts) > 1 else "",
                "announcement_time": normalize_clock(parts[2]) if len(parts) > 2 else "",
                "source": parts[3] if len(parts) > 3 and parts[3] else "手入力",
                "doc_type": "",
            }
        )
    return pd.DataFrame(rows, columns=["earnings_date", "quarter", "announcement_time", "source", "doc_type"])


def combine_events(jquants: pd.DataFrame, uploaded: pd.DataFrame, manual: pd.DataFrame,
                   cutoff: pd.Timestamp, latest_price_date: pd.Timestamp) -> pd.DataFrame:
    frames = [x for x in (jquants, uploaded, manual) if not x.empty]
    if not frames:
        return pd.DataFrame(columns=["earnings_date", "quarter", "announcement_time", "source", "doc_type"])
    all_events = pd.concat(frames, ignore_index=True)
    all_events["earnings_date"] = pd.to_datetime(all_events["earnings_date"], errors="coerce").dt.normalize()
    all_events = all_events.dropna(subset=["earnings_date"])
    all_events = all_events[
        (all_events["earnings_date"] >= cutoff) & (all_events["earnings_date"] <= latest_price_date)
    ]
    priority = {"手入力": 50, "CSV": 40, "企業IR": 45, "TDnet": 50, "J-Quants 財務情報": 35}
    all_events["_priority"] = all_events["source"].map(priority).fillna(30)
    return (
        all_events.sort_values(["earnings_date", "quarter", "_priority"])
        .drop_duplicates(["earnings_date", "quarter"], keep="last")
        .drop(columns="_priority")
        .sort_values("earnings_date")
        .reset_index(drop=True)
    )


def analyze_events(prices: pd.DataFrame, events: pd.DataFrame, flat_threshold: float) -> pd.DataFrame:
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
            reference_close_day = event_day
        else:
            # 休場日発表は直前営業日を基準、直後営業日を反応日とする。
            if pos < 1:
                continue
            prev_day = index[pos - 1]
            next_day = event_day
            reference_close_day = prev_day

        clock = normalize_clock(event.get("announcement_time", ""))
        session, carry_eligible, primary_metric = classify_announcement(disclosed, clock, is_trading_day)
        reliability = time_reliability(str(event.get("source", "")), clock)

        prev_close = float(prices.loc[prev_day, "Close"])
        event_open = float(prices.loc[event_day, "Open"])
        event_close = float(prices.loc[event_day, "Close"])
        ref_close = float(prices.loc[reference_close_day, "Close"])
        next_open = float(prices.loc[next_day, "Open"])
        next_close = float(prices.loc[next_day, "Close"])

        same_day_change = (event_close / prev_close - 1) * 100 if is_trading_day else np.nan
        same_day_open_gap = (event_open / prev_close - 1) * 100 if is_trading_day else np.nan
        next_gu = (next_open / ref_close - 1) * 100
        next_close_change = (next_close / ref_close - 1) * 100
        next_intraday = (next_close / next_open - 1) * 100

        if carry_eligible == "対象":
            judged_value = next_close_change
            judgment = "上昇" if judged_value > flat_threshold else (
                "下落" if judged_value < -flat_threshold else "横ばい"
            )
        else:
            judgment = "対象外"

        rows.append(
            {
                "決算発表日": disclosed.date(),
                "四半期": event["quarter"],
                "発表時刻": clock or "不明",
                "時間信頼度": reliability,
                "発表区分": session,
                "引け買い決算跨ぎ": carry_eligible,
                "主反応指標": primary_metric,
                "データ源": event["source"],
                "前営業日": prev_day.date(),
                "発表日取引日": event_day.date(),
                "翌営業日": next_day.date(),
                "当日寄りGU率(%)": same_day_open_gap,
                "当日終値騰落率(%)": same_day_change,
                "翌営業日GU率(%)": next_gu,
                "翌営業日寄り後(%)": next_intraday,
                "翌営業日終値騰落率(%)": next_close_change,
                "決算跨ぎ判定": judgment,
            }
        )
    return pd.DataFrame(rows)


def summarize_all(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for quarter, group in detail.groupby("四半期"):
        rows.append(
            {
                "四半期": quarter,
                "全件数": len(group),
                "引け後・休場日件数": int((group["引け買い決算跨ぎ"] == "対象").sum()),
                "昼休み・場中件数": int(group["発表区分"].isin(["前場中", "昼休み", "後場中"]).sum()),
                "平均当日終値(%)": group["当日終値騰落率(%)"].mean(),
                "平均翌営業日GU(%)": group["翌営業日GU率(%)"].mean(),
                "平均翌営業日終値(%)": group["翌営業日終値騰落率(%)"].mean(),
            }
        )
    result = pd.DataFrame(rows)
    order = {q: i for i, q in enumerate(VALID_QUARTERS)}
    result["_order"] = result["四半期"].map(order).fillna(99)
    return result.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def summarize_carry(detail: pd.DataFrame) -> pd.DataFrame:
    eligible = detail[detail["引け買い決算跨ぎ"] == "対象"].copy()
    if eligible.empty:
        return pd.DataFrame()
    rows = []
    for quarter, group in eligible.groupby("四半期"):
        n = len(group)
        up = int((group["決算跨ぎ判定"] == "上昇").sum())
        rows.append(
            {
                "四半期": quarter,
                "件数": n,
                "上昇": up,
                "下落": int((group["決算跨ぎ判定"] == "下落").sum()),
                "横ばい": int((group["決算跨ぎ判定"] == "横ばい").sum()),
                "勝率(%)": up / n * 100,
                "GU率(%)": (group["翌営業日GU率(%)"] > 0).mean() * 100,
                "平均GU(%)": group["翌営業日GU率(%)"].mean(),
                "平均翌日終値(%)": group["翌営業日終値騰落率(%)"].mean(),
                "中央値翌日終値(%)": group["翌営業日終値騰落率(%)"].median(),
            }
        )
    result = pd.DataFrame(rows)
    result["参考スコア"] = (
        result["平均翌日終値(%)"] * 0.45
        + result["中央値翌日終値(%)"] * 0.20
        + result["平均GU(%)"] * 0.20
        + (result["勝率(%)"] - 50) / 10 * 0.10
        + (result["GU率(%)"] - 50) / 10 * 0.05
    ) * np.minimum(result["件数"] / 6, 1)
    result["確信度"] = result["件数"].map(lambda n: "高" if n >= 8 else ("中" if n >= 5 else "低"))
    result["評価"] = "判定保留"
    eligible_mask = result["件数"] >= 2
    ranks = result.loc[eligible_mask, "参考スコア"].rank(method="min", ascending=False)
    result.loc[eligible_mask, "評価"] = ranks.map(lambda r: "S" if r == 1 else ("A" if r == 2 else ("B" if r == 3 else "C")))
    order = {q: i for i, q in enumerate(VALID_QUARTERS)}
    result["_order"] = result["四半期"].map(order).fillna(99)
    return result.sort_values("_order").drop(columns="_order").reset_index(drop=True)


st.title("📊 日本株 決算四半期リアクション分析 — 時刻判定版")
st.caption("発表時刻に応じて『昼休み・場中』と『引け後』を分離し、引け買い→翌営業日売却の決算跨ぎだけを別集計します。")

with st.sidebar:
    st.header("分析条件")
    raw_code = st.text_input("銘柄コード", value="4203", max_chars=8)
    years = st.slider("分析年数", 2, 10, 2)
    flat_threshold = st.number_input("横ばい判定幅（±%）", 0.0, 2.0, 0.2, 0.1)
    st.divider()
    uploaded = st.file_uploader("補完CSV（任意）", type=["csv"])
    manual = st.text_area(
        "手入力（任意）",
        placeholder="2025-08-04,1Q,11:30,TDnet",
        help="日付,四半期,時刻,情報源",
    )
    run = st.button("分析する", type="primary", use_container_width=True)

st.info(
    "重要：昼休み・場中発表は、引け時点ですでに決算が公表済みです。したがって『引け前〜引け成り買い→翌日売却』の決算跨ぎ対象には含めません。"
)

if run:
    try:
        code = normalize_code(raw_code)
        ticker = ticker_for(code)
        api_key = str(st.secrets.get("JQUANTS_API_KEY", "")).strip()
        cutoff = pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None).normalize() - pd.DateOffset(years=years)

        with st.spinner("決算と株価を集計しています…"):
            prices = load_prices(ticker, years)
            jq_events, jq_status = load_jquants_earnings(code, api_key)
            csv_events = parse_uploaded_csv(uploaded)
            manual_events = parse_manual_dates(manual)
            events = combine_events(jq_events, csv_events, manual_events, cutoff, prices.index.max())
            detail = analyze_events(prices, events, flat_threshold)
            all_summary = summarize_all(detail) if not detail.empty else pd.DataFrame()
            carry_summary = summarize_carry(detail) if not detail.empty else pd.DataFrame()

        st.subheader(f"{code}（{ticker}）分析結果")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("採用決算", f"{len(detail)}件")
        c2.metric("決算跨ぎ対象", f"{int((detail['引け買い決算跨ぎ'] == '対象').sum()) if not detail.empty else 0}件")
        c3.metric("昼休み・場中", f"{int(detail['発表区分'].isin(['前場中','昼休み','後場中']).sum()) if not detail.empty else 0}件")
        c4.metric("最新株価日", prices.index.max().strftime("%Y-%m-%d"))

        with st.expander("取得状況", expanded=False):
            st.write(f"- {jq_status}")
            st.write("- 発表時刻：J-Quantsの開示時刻を使用")
            st.write("- 時間信頼度：J-Quants・TDnetは◎、企業IRは◎、手入力・CSVは○")

        if detail.empty:
            st.error("分析可能な決算データがありません。")
        else:
            st.markdown("### ① 決算跨ぎ判定")
            carry_count = int((detail["引け買い決算跨ぎ"] == "対象").sum())
            if carry_count == 0:
                st.warning(
                    "この取得期間では、引け後または休場日発表の決算がありません。"
                    "引け買いによる『決算発表前の持ち越し』分析は対象外です。"
                )
            elif carry_summary.empty:
                st.warning("決算跨ぎ対象はありますが、集計できる件数が不足しています。")
            else:
                ranked = carry_summary.sort_values(["参考スコア", "件数"], ascending=False)
                best = ranked.iloc[0]
                st.success(
                    f"決算跨ぎ最上位：{best['四半期']}｜評価 {best['評価']}｜"
                    f"勝率 {best['勝率(%)']:.1f}%｜GU率 {best['GU率(%)']:.1f}%｜"
                    f"平均翌日終値 {best['平均翌日終値(%)']:+.2f}%｜確信度 {best['確信度']}"
                )
                display_carry = carry_summary.copy()
                numeric = display_carry.select_dtypes(include=[np.number]).columns
                display_carry[numeric] = display_carry[numeric].round(2)
                st.dataframe(display_carry, use_container_width=True, hide_index=True)

            st.markdown("### ② 全決算の参考反応")
            st.caption("昼休み・場中発表を含みます。翌営業日反応は『決算発表前からの持ち越し』ではありません。")
            display_all = all_summary.copy()
            numeric = display_all.select_dtypes(include=[np.number]).columns
            display_all[numeric] = display_all[numeric].round(2)
            st.dataframe(display_all, use_container_width=True, hide_index=True)

            st.markdown("### ③ 決算明細")
            display_detail = detail.sort_values("決算発表日", ascending=False).copy()
            percent_columns = [c for c in display_detail.columns if "(%)" in c]
            display_detail[percent_columns] = display_detail[percent_columns].round(2)
            st.dataframe(display_detail, use_container_width=True, hide_index=True)

            st.markdown("### ④ 発表時刻チェック")
            time_table = (
                detail.groupby(["発表時刻", "時間信頼度", "発表区分", "引け買い決算跨ぎ"])
                .size()
                .reset_index(name="件数")
                .sort_values("件数", ascending=False)
            )
            st.dataframe(time_table, use_container_width=True, hide_index=True)

            col1, col2, col3 = st.columns(3)
            col1.download_button(
                "明細CSV",
                detail.to_csv(index=False).encode("utf-8-sig"),
                f"{code}_earnings_detail.csv",
                "text/csv",
                use_container_width=True,
            )
            col2.download_button(
                "全決算集計CSV",
                all_summary.to_csv(index=False).encode("utf-8-sig"),
                f"{code}_all_summary.csv",
                "text/csv",
                use_container_width=True,
            )
            col3.download_button(
                "決算跨ぎ集計CSV",
                carry_summary.to_csv(index=False).encode("utf-8-sig"),
                f"{code}_carry_summary.csv",
                "text/csv",
                use_container_width=True,
                disabled=carry_summary.empty,
            )

    except Exception as exc:
        st.error(f"処理中にエラーが発生しました: {exc}")

with st.expander("判定ルール"):
    st.markdown(
        "- **寄り前・前場中・昼休み・後場中**：引け買い決算跨ぎの対象外\n"
        "- **引け後・休場日発表**：引け買い決算跨ぎの対象\n"
        "- 東証の引けは2024年11月5日以降15:30、それ以前は15:00として判定\n"
        "- 11:30発表は昼休み発表として扱い、当日後場に決算反応が出る可能性があります\n"
        "- 発表時刻不明は決算跨ぎの対象可否を判定しません"
    )
