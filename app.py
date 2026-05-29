#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台股/美股 AI 實戰交易決策工具 — Streamlit 網頁版 v5.0
用法：streamlit run app.py
"""

from datetime import datetime
import streamlit as st

from stock_ai_analyzer import (
    fetch_raw_hist,
    fetch_fundamental_data,
    fetch_news,
    calculate_technical_from_hist,
    build_analysis_context,
    analyze_with_claude,
)


# ── 頁面設定（必須第一行 st 呼叫）────────────────────────────────────────────
st.set_page_config(
    page_title="AI 股票分析工具",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("📈 台股 / 美股 AI 實戰交易決策工具")
st.caption(
    "動能投資策略 × 防洗盤三重濾網 × 即時技術面插補 × Claude AI  |  v5.0 Web 版"
)
st.divider()


# ── Session State 初始化 ──────────────────────────────────────────────────────
_DEFAULTS = {
    "fetched":       False,   # 是否已撈取數據
    "hist":          None,    # 原始 K 線 DataFrame
    "fundamental":   None,    # 基本面 dict
    "news":          None,    # 新聞 list
    "yahoo_price":   None,    # Yahoo 延遲收盤價
    "last_ticker":   "",      # 上次撈取的代號（換股時重設）
    "run_analysis":  False,   # 觸發 AI 分析的旗標
    "technical":     None,    # 技術指標 dict（含即時修正）
    "position":      None,    # 持倉資訊 dict
    "analysis_done": False,   # 是否已完成 AI 分析
    "analysis_text": "",      # AI 輸出完整文字（快取，供重新渲染 & 下載）
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ════════════════════════════════════════════════════════════════════════════
# ① STEP 1：API Key ＋ 股票代號 ＋ 撈取數據
# ════════════════════════════════════════════════════════════════════════════
st.markdown("### 🔑 Step 1：輸入 API Key 與股票代號")

col_key, col_tick = st.columns([3, 2])
with col_key:
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-api03-...",
        help="Key 僅在記憶體暫存，關閉頁面即清除，程式碼不儲存任何 Key。",
    )
with col_tick:
    ticker_input = st.text_input(
        "股票代號",
        placeholder="台股：2330.TW　美股：NVDA",
        help="台股需加 .TW 後綴（例：2330.TW）",
    ).strip().upper()

# 換股 → 自動重設所有狀態
if ticker_input and ticker_input != st.session_state.last_ticker:
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v
    st.session_state.last_ticker = ticker_input

fetch_ready = bool(api_key and ticker_input)

if st.button("📡  開始撈取數據", type="primary", disabled=not fetch_ready):
    with st.spinner(f"正在撈取 {ticker_input} 數據，請稍候…"):
        try:
            hist        = fetch_raw_hist(ticker_input)
            fundamental = fetch_fundamental_data(ticker_input)
            news        = fetch_news(ticker_input, max_news=3)
            yahoo_price = round(float(hist["Close"].iloc[-1]), 2)

            st.session_state.hist        = hist
            st.session_state.fundamental = fundamental
            st.session_state.news        = news
            st.session_state.yahoo_price = yahoo_price
            st.session_state.fetched     = True
            st.session_state.last_ticker = ticker_input
            # 換股時清除舊分析結果
            st.session_state.run_analysis  = False
            st.session_state.technical     = None
            st.session_state.position      = None
            st.session_state.analysis_done = False
            st.session_state.analysis_text = ""
        except Exception as e:
            st.error(f"❌  撈取失敗：{e}\n\n請確認股票代號格式正確，或稍後再試。")
            st.stop()


# ════════════════════════════════════════════════════════════════════════════
# ② STEP 2：即時股價 ＋ 持倉設定（只在撈取成功後顯示）
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.fetched:
    fundamental = st.session_state.fundamental
    cs          = "NT$" if fundamental.get("currency") == "TWD" else "$"
    yahoo_price = st.session_state.yahoo_price
    company     = fundamental.get("company_name", ticker_input)

    st.divider()
    st.markdown(
        f"### 📊 Step 2：確認即時股價 ＋ 持倉設定　— {company}（{ticker_input}）"
    )

    # ── 即時價輸入 ──────────────────────────────────────────────────────────
    with st.expander("⚡ 即時股價校正（選填，預設沿用 Yahoo 延遲價）", expanded=True):
        st.info(
            f"Yahoo Finance 延遲收盤價：**{cs}{yahoo_price}**　"
            f"（通常落後市場 10-15 分鐘）",
            icon="ℹ️",
        )
        realtime_input = st.number_input(
            f"輸入您券商 App 的即時股價（{cs}）",
            min_value=0.01,
            value=float(yahoo_price),
            step=0.01,
            format="%.2f",
            key="realtime_price_input",
            help="不知道即時價？維持預設值即可，不影響其他欄位。",
        )

    # ── 持倉資訊 ────────────────────────────────────────────────────────────
    st.markdown("#### 💼 持倉資訊")
    holds_radio = st.radio(
        "您目前是否持有此股票？",
        options=["否（未持有）", "是（已持有）"],
        index=0,
        horizontal=True,
        key="holds_radio",
    )
    holds_bool = holds_radio.startswith("是")

    cost_basis_val: float | None = None
    shares_val: int | None = None

    if holds_bool:
        col_cost, col_shares = st.columns(2)
        with col_cost:
            cost_basis_val = st.number_input(
                f"持倉成本價（{cs} / 股）",
                min_value=0.01,
                step=0.01,
                format="%.2f",
                key="cost_basis_input",
            )
        with col_shares:
            shares_unit = st.radio(
                "股數單位",
                options=["股（零股）", "張（1 張 = 1000 股）"],
                horizontal=True,
                key="shares_unit_radio",
            )
            if shares_unit.startswith("股"):
                shares_val = int(
                    st.number_input("持有股數", min_value=1, step=1,
                                   value=100, key="shares_input_raw")
                )
            else:
                lots    = int(st.number_input("持有張數", min_value=1, step=1,
                                              value=1, key="lots_input"))
                leftover = int(st.number_input("加零股（股）", min_value=0, step=1,
                                               value=0, key="leftover_input"))
                shares_val = lots * 1000 + leftover

    # 按鈕可用條件：若持有，則成本必須 > 0
    analyze_ready = not holds_bool or (holds_bool and bool(cost_basis_val and cost_basis_val > 0))

    st.divider()
    if st.button("🤖  執行 AI 實戰分析", type="primary", disabled=not analyze_ready):
        # ── 計算技術指標（Patch → Calculate）───────────────────────────────
        with st.spinner("計算技術指標中…"):
            technical = calculate_technical_from_hist(
                st.session_state.hist.copy(),  # copy 避免修改 session state 的 DataFrame
                float(realtime_input),
            )

        # ── 組裝持倉 dict ─────────────────────────────────────────────────
        current_price = technical["current_price"]
        if holds_bool and cost_basis_val and shares_val:
            pnl     = round((current_price - cost_basis_val) * shares_val, 2)
            pnl_pct = round((current_price - cost_basis_val) / cost_basis_val * 100, 2)
            position = {
                "holds":            True,
                "cost_basis":       cost_basis_val,
                "shares":           shares_val,
                "unrealized_pnl":   pnl,
                "unrealized_pnl_pct": pnl_pct,
                "is_zero_share":    shares_val < 1000,
            }
        else:
            position = {
                "holds":            False,
                "cost_basis":       None,
                "shares":           None,
                "unrealized_pnl":   None,
                "unrealized_pnl_pct": None,
                "is_zero_share":    False,
            }

        # ── 儲存到 session state，設旗標觸發 STEP 3 ────────────────────────
        st.session_state.technical     = technical
        st.session_state.position      = position
        st.session_state.run_analysis  = True
        st.session_state.analysis_done = False
        st.session_state.analysis_text = ""


# ════════════════════════════════════════════════════════════════════════════
# ③ STEP 3：AI 串流輸出 ＋ 快取渲染
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.run_analysis or st.session_state.analysis_done:
    st.divider()
    st.markdown("### 🤖 Step 3：AI 實戰交易決策報告")

    technical   = st.session_state.technical
    fundamental = st.session_state.fundamental
    news        = st.session_state.news
    position    = st.session_state.position
    cs          = "NT$" if fundamental.get("currency") == "TWD" else "$"

    # ── 數據摘要卡片 ─────────────────────────────────────────────────────────
    # ── 計算乖離率與型態標籤（與 build_analysis_context 邏輯一致）──────────
    _p   = technical["current_price"]
    _m5  = technical.get("ma5",  technical["ma20"])
    _m20 = technical["ma20"]
    _m60 = technical["ma60"]
    _is_us_hp_app = (fundamental.get("currency") != "TWD") and _p >= 50
    _dev_app = round((_p - _m20) / _m20 * 100, 2)
    _ra_th   = 8.0 if _is_us_hp_app else 12.0
    if _p > _m5 > _m20 > _m60 and _dev_app > _ra_th:
        _regime_tag = "🚨 型態 A"
    elif _p > _m20 and _m20 > _m60:
        _regime_tag = "📈 型態 B"
    else:
        _regime_tag = "🔍 型態 C"

    c1, c2, c3, c4, c5 = st.columns(5)
    price_tag = (
        f"⚡ {cs}{technical['current_price']}（即時）"
        if technical["price_overridden"]
        else f"{cs}{technical['current_price']}（延遲）"
    )
    c1.metric("當前股價",          price_tag)
    c2.metric("MA5 / MA20 / MA60", f"{cs}{_m5} / {cs}{_m20} / {cs}{_m60}")
    c3.metric("MA20 乖離率",       f"{_dev_app:+.2f}%")
    c4.metric("RSI / 量能倍數",    f"{technical['rsi']} / {technical.get('volume_ratio', 'N/A')}×")
    c5.metric("AI 判定型態",       _regime_tag)

    if position["holds"]:
        pnl     = position["unrealized_pnl"]
        pnl_pct = position["unrealized_pnl_pct"]
        cost    = position["cost_basis"]
        p1, p2, p3 = st.columns(3)
        p1.metric("持倉成本",   f"{cs}{cost}")
        p2.metric("未實現損益", f"{cs}{pnl:+,.0f}",  delta=f"{pnl_pct:+.2f}%")
        p3.metric("持股數量",   f"{position['shares']} 股")

    st.markdown("---")

    # ── 串流輸出（首次）或渲染快取（後續重跑）──────────────────────────────
    if st.session_state.run_analysis:
        # 第一次：組 context → 串流 → 快取
        with st.spinner("組建分析數據中…"):
            context = build_analysis_context(
                ticker_input, technical, fundamental, news, position
            )

        try:
            stream_gen = analyze_with_claude(
                context=context,
                ticker=ticker_input,
                position=position,
                fundamental=fundamental,
                technical=technical,
                api_key=api_key,
                stream=True,
            )
            # st.write_stream 逐字顯示，結束後回傳完整文字
            full_text = st.write_stream(stream_gen)
            st.session_state.analysis_text = full_text
            st.session_state.analysis_done = True
        except Exception as e:
            st.error(f"❌  Claude API 呼叫失敗：{e}")
        finally:
            # 清除旗標，下次重跑走快取路徑
            st.session_state.run_analysis = False

    else:
        # 後續重跑：直接渲染快取文字（不重複呼叫 API）
        st.markdown(st.session_state.analysis_text)

    # ── 下載 ＋ 重新分析按鈕 ──────────────────────────────────────────────
    if st.session_state.analysis_done and st.session_state.analysis_text:
        st.divider()
        col_dl, col_new = st.columns([2, 1])
        with col_dl:
            filename = (
                f"{ticker_input}_AI分析報告_"
                f"{datetime.now().strftime('%Y%m%d_%H%M')}.md"
            )
            st.download_button(
                label="📥  下載完整分析報告（Markdown）",
                data=st.session_state.analysis_text,
                file_name=filename,
                mime="text/markdown",
            )
        with col_new:
            if st.button("🔄  分析另一支股票"):
                for k, v in _DEFAULTS.items():
                    st.session_state[k] = v
                st.rerun()


# ── 側欄說明 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📖 使用說明")
    st.markdown("""
**Step 1** — 輸入 Anthropic API Key 與股票代號，點「開始撈取數據」。

**Step 2** — 確認 Yahoo 延遲價，可輸入券商 App 即時價覆蓋；設定持倉資訊。

**Step 3** — 點「執行 AI 實戰分析」，Claude 即時串流輸出報告。

---

**股票代號格式**

| 市場 | 範例 |
|------|------|
| 台灣上市 | 2330.TW |
| 台灣 ETF | 0050.TW |
| 美股 | NVDA、AAPL |
| 港股 | 0700.HK |

---

**API Key 安全說明**

- Key 僅在記憶體中暫存，用於呼叫 Claude API
- 關閉或重新整理頁面即清除
- 程式碼不存、不上傳任何 Key

---

> ⚠️ **免責聲明**
> 本工具輸出僅供參考，不構成投資建議。
> 投資一定有風險，請依自身風險承受能力決策。
""")
