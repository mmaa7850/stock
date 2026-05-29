#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
台股/美股 AI 實戰交易決策工具 v5.0
動能投資策略 × 防洗盤心理學 × 真·即時技術面插補 × Claude AI

v5.0 核心升級（在 v4.0 防洗盤基礎上解決指標時空錯亂問題）：
  【真·即時技術面插補機制】
  舊版問題：yfinance 的 K 線數據落後市場 10-15 分鐘，導致 MA20/MA60/RSI
            用延遲快照計算，AI 在防洗盤與移動停利判斷時發生時空錯亂。
  v5.0 解法：三段式數據流（Fetch → Patch → Calculate）
    Step A. fetch_raw_hist()：先撈取原始 OHLCV DataFrame，不做任何計算
    Step B. ask_realtime_price()：顯示 Yahoo 延遲價，詢問使用者券商 App 即時價
    Step C. calculate_technical_from_hist()：
            先將 DataFrame 最後一列的 Close（與可能的 High）強行覆蓋為即時價，
            再用覆蓋後的數據重算 MA20、MA60、RSI(14)，
            確保所有技術指標 100% 同步當前市況。

  保留 v4.0 全部功能：
  - 防洗盤三重濾網（條件 A/B/C）
  - 移動停利保本防守線（獲利單防守位不得低於成本）
  - 連續 MA20 跌破偵測
  - Forward P/E / PEG 高成長股估值框架
  - 股息率異常清洗

用法：python stock_ai_analyzer.py
"""

import os
import sys
import getpass
import anthropic
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime


# ── 全域設定 ──────────────────────────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-6"
DIVIDEND_YIELD_ANOMALY_THRESHOLD = 0.20  # 股息率超過 20% 視為 yfinance 資料異常
MAX_TOKENS = 2500                         # 確保報告完整不截斷


# ==================== Step 0：API Key 取得（安全版）===========================

def get_api_key() -> str:
    """
    優先讀取環境變數 ANTHROPIC_API_KEY，
    若無則在終端機以隱藏輸入方式向使用者索取（不顯示字元，安全如密碼欄）。
    程式碼內不存放任何 Key，可安全分享給朋友。
    """
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        print("  ✅  已從環境變數讀取 API Key。")
        return key

    print("\n  請輸入您的 Anthropic API Key（輸入時字元不顯示）：")
    print("  取得金鑰：https://console.anthropic.com/settings/keys\n")
    key = getpass.getpass("  API Key：").strip()

    if not key:
        print("❌  API Key 不能為空，程式結束。")
        sys.exit(1)

    return key


# ==================== Step 1：持倉互動取得 =====================================

def get_position_info(ticker: str, current_price: float, currency: str) -> dict:
    """
    詢問使用者是否持有該股票，若持有則繼續詢問成本與股數。
    在 Python 端計算未實現損益，傳入 Context 讓 AI 給出個人化建議。

    設計原則：損益金額在 Python 算好再傳給 AI，
    比讓 AI 自行計算更準確，也節省 token。
    """
    cs = "NT$" if currency == "TWD" else "$"
    print(f"\n  ──────────────────────────────────────")
    print(f"  當前股價：{cs}{current_price}")

    holds_raw = input("\n  👉 請問您目前是否持有該股票？(Y/N)：").strip().upper()
    holds = holds_raw == "Y"

    if not holds:
        return {
            "holds": False,
            "cost_basis": None,
            "shares": None,
            "unrealized_pnl": None,
            "unrealized_pnl_pct": None,
            "is_zero_share": False,
        }

    # ── 問題 B：每股平均成本 ──────────────────────────────────────────────
    while True:
        try:
            cost_basis = float(
                input("  👉 請問您的每股平均成本是多少？：").strip()
            )
            if cost_basis <= 0:
                raise ValueError
            break
        except ValueError:
            print("  ⚠  請輸入有效正數（例如：1250 或 1250.50）")

    # ── 問題 C：持有股數 ──────────────────────────────────────────────────
    while True:
        try:
            shares = int(
                input("  👉 請問您目前持有多少股？（1 張 = 1000 股）：").strip()
            )
            if shares <= 0:
                raise ValueError
            break
        except ValueError:
            print("  ⚠  請輸入有效正整數（例如：1000 代表 1 張；500 代表零股）")

    # ── 計算未實現損益 ─────────────────────────────────────────────────────
    unrealized_pnl = round((current_price - cost_basis) * shares, 2)
    unrealized_pnl_pct = round((current_price - cost_basis) / cost_basis * 100, 2)

    return {
        "holds": True,
        "cost_basis": cost_basis,
        "shares": shares,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pnl_pct": unrealized_pnl_pct,
        "is_zero_share": shares < 1000,  # 未滿一張視為零股投資人
    }


# ==================== 技術指標計算 ============================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """RSI 計算：Wilder EWM 平滑法，與 TradingView 結果一致。"""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return round(float((100 - 100 / (1 + rs)).iloc[-1]), 2)


# ==================== 技術面數據三段式流程（v5.0 核心升級）======================
#
# 舊版 fetch_technical_data() 的致命問題：
#   它把「撈取數據」和「計算指標」合併在同一個函式裡，
#   導致 MA20/MA60/RSI 永遠基於 Yahoo 的延遲快照（落後 10-15 分鐘），
#   在盤中劇烈波動時，AI 的防洗盤判斷會發生時空錯亂。
#
# v5.0 解法：Fetch → Patch → Calculate 三段式分離
#   fetch_raw_hist()            只負責撈取原始 DataFrame
#   ask_realtime_price()        只負責向使用者詢問即時價格
#   calculate_technical_from_hist()  先覆蓋再計算（核心邏輯）
# ──────────────────────────────────────────────────────────────────────────────

def fetch_raw_hist(ticker: str) -> pd.DataFrame:
    """
    第一段：純粹撈取原始 OHLCV DataFrame，不做任何計算。

    為何分離：確保即時價格覆蓋動作（Patch）發生在所有計算之前，
    徹底杜絕「用延遲數據算出來再補修正」的邏輯順序錯誤。
    抓取 90 天（而非 60 天）是為確保 MA60 計算有足夠的基底數據。
    """
    print(f"  [1/3] 正在從 Yahoo Finance 撈取 {ticker} 歷史 K 線（90 日）...")

    stock = yf.Ticker(ticker)
    hist = stock.history(period="90d")

    if hist.empty:
        raise ValueError(
            f"無法取得 {ticker} 的歷史數據。\n"
            "  台股格式：2330.TW  |  美股格式：AAPL"
        )
    return hist


def ask_realtime_price(delayed_price: float, currency: str) -> float:
    """
    第二段：告知使用者 Yahoo 的延遲股價，詢問是否輸入券商 App 的即時股價。

    回傳邏輯：
    - 使用者直接按 Enter → 回傳 delayed_price（維持 Yahoo 數據）
    - 使用者輸入有效正數  → 回傳使用者輸入的即時股價
    - 使用者輸入無效內容  → 警告並回傳 delayed_price（容錯降級）

    回傳值一定是 float，後續流程不需要再判斷 None。
    """
    cs = "NT$" if currency == "TWD" else "$"

    print(f"\n  ⚠   Yahoo Finance 延遲股價：{cs}{delayed_price}")
    print(f"       （通常落後市場約 10~15 分鐘，盤中波動時可能嚴重失真）")

    raw = input(
        f"\n  👉  請輸入您券商 App 的即時股價"
        f"（直接按 Enter 則採用 Yahoo 延遲數據）："
    ).strip()

    if not raw:
        print(f"  ✅  採用 Yahoo 延遲數據 → {cs}{delayed_price}")
        return delayed_price

    try:
        realtime = float(raw)
        if realtime <= 0:
            raise ValueError
        print(f"  ✅  即時股價修正：{cs}{delayed_price} → {cs}{realtime}")
        print(f"       所有技術指標（MA20 / MA60 / RSI）將基於此即時價重算。")
        return realtime
    except ValueError:
        print(f"  ⚠   無效輸入，降級採用 Yahoo 延遲數據 → {cs}{delayed_price}")
        return delayed_price


def calculate_technical_from_hist(hist: pd.DataFrame, realtime_price: float) -> dict:
    """
    第三段：真·即時技術面計算（v5.0 核心）。

    執行順序絕對不能顛倒：
    ┌─────────────────────────────────────────────────────────────┐
    │  Step 1. 覆蓋（Patch）：                                    │
    │    把 hist 最後一列的 Close 強行替換為 realtime_price。       │
    │    若 realtime_price > 原本 High，同步更新 High，            │
    │    確保「最高價 ≥ 收盤價」的 K 線邏輯不被破壞。              │
    │                                                             │
    │  Step 2. 計算（Calculate）：                                │
    │    用已覆蓋的 DataFrame 計算 MA20、MA60、RSI(14)。           │
    │    最後一根 K 線的 Close 已是即時價，                        │
    │    因此所有滾動計算的最末值都 100% 同步當前市況。            │
    └─────────────────────────────────────────────────────────────┘

    注意：若 realtime_price == Yahoo 延遲價（使用者按 Enter），
    覆蓋動作等同原值賦值，計算結果與舊版完全一致，不影響準確性。
    """
    # ── Step 1：即時價格覆蓋（先覆蓋，後計算，順序不可逆）──────────────────
    last_idx = hist.index[-1]
    yahoo_delayed_price = round(float(hist.at[last_idx, "Close"]), 2)
    price_overridden = (realtime_price != yahoo_delayed_price)

    hist.at[last_idx, "Close"] = realtime_price
    if realtime_price > hist.at[last_idx, "High"]:
        # 即時價突破了 Yahoo 記錄的最高價，同步更新 High
        hist.at[last_idx, "High"] = realtime_price

    # ── Step 2：基於覆蓋後的 DataFrame 重算所有技術指標 ─────────────────────
    hist_60 = hist.tail(60)
    close_all = hist["Close"]   # 最後一筆已是即時價
    close_60  = hist_60["Close"]

    # 移動平均線（rolling window 計算，最末值使用即時收盤價）
    ma20 = round(float(close_all.rolling(20).mean().iloc[-1]), 2)
    ma60 = round(float(close_all.rolling(60).mean().iloc[-1]), 2)

    # RSI(14)（EWM 法，最末差值 delta 使用即時價 - 昨日收盤）
    rsi = calculate_rsi(close_all, period=14)

    current_price  = round(float(close_60.iloc[-1]), 2)   # 即 realtime_price
    current_volume = int(hist_60["Volume"].iloc[-1])
    avg_volume_60d = int(hist_60["Volume"].mean())
    high_60d       = round(float(hist_60["High"].max()), 2)
    low_60d        = round(float(hist_60["Low"].min()), 2)

    # 連續 MA20 跌破偵測（v4.0 保留）
    today_below_ma20  = current_price < ma20
    prev_close        = round(float(close_60.iloc[-2]), 2) if len(close_60) >= 2 else None
    prev_below_ma20   = (prev_close is not None and prev_close < ma20)
    consecutive_break = today_below_ma20 and prev_below_ma20

    change_5d = (
        round((close_60.iloc[-1] / close_60.iloc[-6]  - 1) * 100, 2)
        if len(close_60) >= 6  else None
    )
    change_20d = (
        round((close_60.iloc[-1] / close_60.iloc[-21] - 1) * 100, 2)
        if len(close_60) >= 21 else None
    )

    return {
        "current_price":         current_price,
        "yahoo_delayed_price":   yahoo_delayed_price,   # 原始延遲價，供 Context 標注
        "price_overridden":      price_overridden,       # 是否套用了即時修正
        "ma20":                  ma20,
        "ma60":                  ma60,
        "rsi":                   rsi,
        "high_60d":              high_60d,
        "low_60d":               low_60d,
        "current_volume":        current_volume,
        "avg_volume_60d":        avg_volume_60d,
        "change_5d_pct":         change_5d,
        "change_20d_pct":        change_20d,
        "last_date":             str(hist_60.index[-1].date()),
        # v4.0 防洗盤欄位
        "today_below_ma20":      today_below_ma20,
        "prev_close":            prev_close,
        "consecutive_ma20_break": consecutive_break,
    }


# ==================== 基本面數據撈取（含異常清洗）================================

def fetch_fundamental_data(ticker: str) -> dict:
    """
    撈取基本面數據，兩項核心防禦：

    【異常清洗】dividendYield > 20% → 標為「數據異常，不納入評估」
      原因：yfinance 有時以單季配息×4計算年化率，造成台股顯示 100%+ 的錯誤。

    【估值升級】同時抓取 Forward P/E 與 PEG Ratio
      高成長股（如聯發科、NVDA）的估值必須結合未來獲利成長來判斷，
      單看 Trailing P/E 會造成嚴重踏空。
    """
    print(f"  [2/3] 正在撈取 {ticker} 基本面數據...")

    stock = yf.Ticker(ticker)
    info = stock.info

    def safe_get(key, default="N/A", round_digits=None):
        val = info.get(key)
        if val is None or val == "None":
            return default
        if round_digits is not None and isinstance(val, (int, float)):
            return round(float(val), round_digits)
        return val

    # ── 市值格式化 ────────────────────────────────────────────────────────
    cap = info.get("marketCap")
    if isinstance(cap, (int, float)) and cap > 0:
        if cap >= 1e12:
            market_cap_str = f"{cap / 1e12:.2f} 兆"
        elif cap >= 1e8:
            market_cap_str = f"{cap / 1e8:.2f} 億"
        else:
            market_cap_str = f"{cap:,.0f}"
    else:
        market_cap_str = "N/A"

    # ── 股息率：異常清洗核心邏輯 ───────────────────────────────────────────
    div_raw = info.get("dividendYield")
    if not isinstance(div_raw, (int, float)) or div_raw <= 0:
        div_yield_str = "N/A（無配息）"
    elif div_raw > DIVIDEND_YIELD_ANOMALY_THRESHOLD:
        # 超過 20% 極可能是 yfinance 資料基準錯誤，清洗掉，不傳給 AI
        div_yield_str = (
            f"數據異常（原始值 {div_raw * 100:.1f}%，超過 20% 合理上限），"
            f"不納入評估"
        )
    else:
        div_yield_str = f"{div_raw * 100:.2f}%"

    # ── EPS 成長率與高成長判斷 ────────────────────────────────────────────
    eg = info.get("earningsGrowth")
    earnings_growth_str = f"{eg * 100:.1f}%" if isinstance(eg, float) else "N/A"
    is_high_growth = isinstance(eg, float) and eg > 0.20

    # ── PEG Ratio（本益成長比）────────────────────────────────────────────
    peg = info.get("pegRatio")
    if isinstance(peg, (int, float)) and peg > 0:
        peg_val = round(float(peg), 2)
        if peg_val < 1.0:
            peg_str = f"{peg_val}（< 1，成長速度快於估值，相對低估）"
        elif peg_val < 2.0:
            peg_str = f"{peg_val}（1-2，估值合理）"
        else:
            peg_str = f"{peg_val}（> 2，成長溢價偏高）"
    else:
        peg_str = "N/A"

    # ── Forward P/E 壓縮程度分析 ──────────────────────────────────────────
    trailing_pe = safe_get("trailingPE", round_digits=2)
    forward_pe = safe_get("forwardPE", round_digits=2)
    pe_compression_note = ""
    pe_compression_pct = None

    if (isinstance(trailing_pe, (int, float)) and
            isinstance(forward_pe, (int, float)) and
            trailing_pe > 0 and forward_pe > 0):
        drop = (trailing_pe - forward_pe) / trailing_pe * 100
        pe_compression_pct = round(drop, 1)
        if drop >= 15:
            pe_compression_note = (
                f"⚡ Forward P/E（{forward_pe}）比 Trailing P/E（{trailing_pe}）"
                f"低 {drop:.1f}%，市場已定價強勁的未來獲利成長，"
                f"高靜態 P/E ≠ 股票貴。"
            )
        elif drop < 0:
            pe_compression_note = (
                f"⚠ Forward P/E（{forward_pe}）高於 Trailing P/E（{trailing_pe}），"
                f"市場預期未來獲利下滑，需謹慎。"
            )

    def fmt_pct(key):
        val = info.get(key)
        return f"{val * 100:.1f}%" if isinstance(val, float) else "N/A"

    return {
        "company_name": safe_get("longName", safe_get("shortName", ticker)),
        "sector": safe_get("sector"),
        "industry": safe_get("industry"),
        "currency": safe_get("currency", "USD"),
        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "pe_compression_note": pe_compression_note,
        "pe_compression_pct": pe_compression_pct,
        "peg_ratio": peg_str,
        "pb_ratio": safe_get("priceToBook", round_digits=2),
        "dividend_yield": div_yield_str,
        "eps": safe_get("trailingEps", round_digits=2),
        "earnings_growth": earnings_growth_str,
        "is_high_growth": is_high_growth,
        "market_cap": market_cap_str,
        "week52_high": safe_get("fiftyTwoWeekHigh", round_digits=2),
        "week52_low": safe_get("fiftyTwoWeekLow", round_digits=2),
        "beta": safe_get("beta", round_digits=2),
        "revenue_growth": fmt_pct("revenueGrowth"),
        "profit_margins": fmt_pct("profitMargins"),
        "analyst_target": safe_get("targetMeanPrice", round_digits=2),
        "recommendation": safe_get("recommendationKey", "N/A"),
    }


# ==================== 新聞撈取 ================================================

def fetch_news(ticker: str, max_news: int = 3) -> list:
    """
    撈取最新新聞，同時相容 yfinance 新版（content 嵌套）與舊版（直接欄位）格式。
    """
    print(f"  [3/3] 正在撈取 {ticker} 最新新聞...")

    stock = yf.Ticker(ticker)
    news_result = []

    try:
        news_list = stock.news
        if not news_list:
            return [{"title": "目前無相關新聞", "date": "", "link": ""}]

        for item in news_list[:max_news]:
            content = item.get("content", {})
            if content:
                title = content.get("title", "無標題")
                link = content.get("canonicalUrl", {}).get("url", "")
                pub_date = content.get("pubDate", "")
            else:
                title = item.get("title", "無標題")
                link = item.get("link", "")
                ts = item.get("providerPublishTime")
                pub_date = (
                    datetime.fromtimestamp(ts).strftime("%Y-%m-%d") if ts else ""
                )

            if pub_date and "T" in str(pub_date):
                pub_date = str(pub_date)[:10]

            news_result.append({
                "title": title,
                "date": pub_date or "日期未知",
                "link": link,
            })

    except Exception as e:
        print(f"  ⚠  新聞撈取失敗（{e}），以空白新聞繼續。")
        return [{"title": "新聞暫時無法取得", "date": "", "link": ""}]

    return news_result or [{"title": "目前無相關新聞", "date": "", "link": ""}]


# ==================== 建構傳給 Claude 的 Context ================================

def build_analysis_context(
    ticker: str,
    technical: dict,
    fundamental: dict,
    news: list,
    position: dict,
) -> str:
    """
    將技術面、基本面、新聞、持倉資訊整合為結構化文字 Context。

    v4.0 新增三個關鍵區塊：
    ① 洗盤偵測分析（Python 端預判，給 AI 明確的量價結合訊號）
       - 量能萎縮 + 跌破均線 = 大概率洗盤，不賣
       - 量能放大 + 跌破均線 = 主力出貨警告
       - 連續 2 日跌破 MA20 = 技術面條件 B 初步成立
    ② 保本防守線（僅獲利時計算）
       - 防止 AI 把短線防守位設在成本以下，把賺錢的單放到賠錢
       - 規則：防守位最低 = 成本 × 1.005（至少留 0.5% 獲利空間）
    ③ 損益分類標籤（讓 AI 不必自行判斷，直接走對應策略路徑）
    """
    currency = fundamental.get("currency", "USD")
    cs = "NT$" if currency == "TWD" else "$"

    price = technical["current_price"]

    # ── 市場類型 & 股價區間偵測（決定保本防守線寬容度）──────────────────────
    # 台股：台股有 10% 漲跌幅限制，盤中震盪幅度可控，可用精準保本邏輯
    # 美股低價股（< $50）：波動幅度相對小，防守線貼近成本
    # 美股高價股（≥ $50）：無漲跌幅限制，高價股日內正常震盪輕易達 1~3%，
    #   若把防守線卡在成本 ±0.2%，開盤前幾分鐘的隨機插針就會誤觸停損，
    #   因此必須給予 1.5~2% 的波動寬容度，並要求以收盤價而非盤中價執行
    is_tw_stock      = ".TW" in ticker.upper()                 # 台股（含 .TW / .TWO）
    is_us_high_price = (not is_tw_stock) and (price >= 50)     # 美股高價股

    ma20 = technical["ma20"]
    ma60 = technical["ma60"]
    rsi = technical["rsi"]

    # ── 量能比計算（後續洗盤偵測要用）──────────────────────────────────────
    vol_ratio = (
        technical["current_volume"] / technical["avg_volume_60d"]
        if technical["avg_volume_60d"] > 0 else 1.0
    )

    # ── 均線綜合研判 ─────────────────────────────────────────────────────────
    if price > ma20 > ma60:
        ma_summary = f"黃金排列，股價（{cs}{price}）站上 MA20（{cs}{ma20}）與 MA60（{cs}{ma60}），強勢多頭格局"
    elif price > ma60 and ma20 <= ma60:
        ma_summary = f"死亡排列但股價（{cs}{price}）守住 MA60（{cs}{ma60}），多空交戰中"
    elif ma20 > ma60 and price <= ma20:
        ma_summary = f"黃金排列但股價（{cs}{price}）回跌至 MA20（{cs}{ma20}）下方，短線走弱但中線偏多"
    else:
        ma_summary = f"死亡排列，股價（{cs}{price}）低於 MA20（{cs}{ma20}）與 MA60（{cs}{ma60}），偏空"

    # ── RSI 研判 ─────────────────────────────────────────────────────────────
    if rsi >= 80:
        rsi_signal = f"RSI {rsi}，嚴重超買，短線回調壓力極大"
    elif rsi >= 70:
        rsi_signal = f"RSI {rsi}，超買警戒（70-80），需留意獲利了結賣壓"
    elif rsi <= 20:
        rsi_signal = f"RSI {rsi}，嚴重超賣，恐慌殺低後留意技術反彈"
    elif rsi <= 30:
        rsi_signal = f"RSI {rsi}，超賣區（20-30），具技術性反彈潛力"
    else:
        rsi_signal = f"RSI {rsi}，中性區間（30-70），無明顯超買超賣訊號"

    # ── 量能研判 ─────────────────────────────────────────────────────────────
    if vol_ratio >= 2.0:
        vol_label = "大幅放量，需辨別攻擊量或出逃量"
    elif vol_ratio >= 1.5:
        vol_label = "溫和放量，動能尚可"
    elif vol_ratio <= 0.5:
        vol_label = "大幅縮量，觀望氣氛濃厚"
    else:
        vol_label = "量能正常"
    vol_signal = f"今日量能為 60 日均量的 {vol_ratio:.1f} 倍（{vol_label}）"

    # ── 高成長股 / AI 題材標記 ───────────────────────────────────────────────
    growth_flag = ""
    pe_note = fundamental.get("pe_compression_note", "")
    if fundamental.get("is_high_growth") or pe_note:
        growth_flag = (
            "\n⚡ 【動能題材標記】：本股屬高成長或 AI 題材強勢股。"
            "AI 分析師必須優先參考 Forward P/E 與 PEG Ratio 評估估值，"
            "不得單憑高 Trailing P/E 結論「估值昂貴、觀望」。"
        )

    # ── 持倉區塊 + v4.0 洗盤偵測 + 保本防守線 ───────────────────────────────
    if position["holds"]:
        cost = position["cost_basis"]
        shares = position["shares"]
        pnl = position["unrealized_pnl"]
        pnl_pct = position["unrealized_pnl_pct"]
        is_zero = position["is_zero_share"]

        # 損益分類（讓 AI 直接走對應策略路徑，不需自行判斷）
        if pnl_pct >= 20:
            pnl_tier = "大幅獲利（>20%）→ 必須採用移動停利，鎖住大部分利潤"
        elif pnl_pct >= 5:
            pnl_tier = "中幅獲利（5-20%）→ 設保本防守線，往上加碼評估題材強度"
        elif pnl_pct >= 0:
            pnl_tier = "小幅獲利（0-5%）→ 謹慎守住本金，防守位不得低於成本"
        elif pnl_pct >= -10:
            pnl_tier = "輕度套牢（0 到 -10%）→ 優先啟動防洗盤三重濾網判斷，避免洗出場"
        else:
            pnl_tier = "深度套牢（<-10%）→ 防洗盤三重濾網；若三條件皆成立則果斷止損"

        # ── v5.1 核心：自適應保本防守線計算 ──────────────────────────────
        #
        # 台股（.TW）：保留精準保本邏輯
        #   ≥5% 獲利 → 成本 +0.5%；0~5% 獲利 → 成本 +0.2%
        #
        # 美股低價股（非 .TW，股價 < $50）：防守線貼近成本
        #   波動幅度相對小，成本價即可作為合理防守底線
        #
        # 美股高價股（非 .TW，股價 ≥ $50）：給予 1.5~2% 波動寬容度
        #   無漲跌幅限制 + 高價股開盤 30 分鐘震盪輕易達 1~3%，
        #   防守線必須夠寬，且一律以【當日收盤價】為執行依據，而非盤中插針
        if pnl_pct >= 5:
            if is_tw_stock:
                breakeven_line   = round(cost * 1.005, 2)
                profit_lock_line = round(cost + (price - cost) * 0.5, 2)
                breakeven_note   = (
                    f"\n  ★ 保本防守線（台股）：{cs}{breakeven_line}"
                    f"（成本 +0.5%，跌破即保本出場，禁止轉盈為虧）"
                    f"\n  ★ 獲利保護線：{cs}{profit_lock_line}"
                    f"（當前獲利的 50%，建議設為移動停利最低防守位）"
                )
            elif is_us_high_price:
                breakeven_line   = round(cost * 0.985, 2)          # 成本 -1.5%
                profit_lock_line = round(cost + (price - cost) * 0.5, 2)
                breakeven_note   = (
                    f"\n  ★ 保本防守線（美股高價股，波動容忍版）：{cs}{breakeven_line}"
                    f"（成本 -1.5%，已含美股高價股正常日內波動緩衝；"
                    f"以【當日收盤價】跌破為執行依據，禁止依盤中插針出場）"
                    f"\n  ★ 獲利保護線：{cs}{profit_lock_line}"
                    f"（當前獲利的 50%，移動停利最低防守位，同樣以收盤價判斷）"
                )
            else:                                                   # 美股低價股 < $50
                breakeven_line   = round(cost, 2)
                profit_lock_line = round(cost + (price - cost) * 0.5, 2)
                breakeven_note   = (
                    f"\n  ★ 保本防守線（美股低價股）：{cs}{breakeven_line}"
                    f"（成本價，以【當日收盤價】跌破為執行依據）"
                    f"\n  ★ 獲利保護線：{cs}{profit_lock_line}"
                    f"（當前獲利的 50%，建議設為移動停利最低防守位）"
                )
        elif pnl_pct > 0:
            if is_tw_stock:
                breakeven_line   = round(cost * 1.002, 2)
                profit_lock_line = None
                breakeven_note   = (
                    f"\n  ★ 保本防守線（台股）：{cs}{breakeven_line}"
                    f"（成本 +0.2%，獲利空間小，防守線必須在此之上）"
                )
            elif is_us_high_price:
                breakeven_line   = round(cost * 0.98, 2)            # 成本 -2%
                profit_lock_line = None
                breakeven_note   = (
                    f"\n  ★ 保本防守線（美股高價股，波動容忍版）：{cs}{breakeven_line}"
                    f"（成本 -2%，小幅獲利時給予更大寬容度，防止開盤 30 分鐘震盪誤觸；"
                    f"以【當日收盤價】跌破為執行依據，禁止依盤中插針出場）"
                )
            else:                                                   # 美股低價股 < $50
                breakeven_line   = round(cost, 2)
                profit_lock_line = None
                breakeven_note   = (
                    f"\n  ★ 保本防守線（美股低價股）：{cs}{breakeven_line}"
                    f"（成本價，以【當日收盤價】跌破為執行依據）"
                )
        else:
            breakeven_line   = None
            profit_lock_line = None
            breakeven_note   = ""

        # ── v4.0 核心：洗盤偵測分析 ──────────────────────────────────────
        # 整合「股價位置 vs MA20」+「量能比」+「連續跌破天數」
        # 給 AI 一個預判結論，讓它不需要自己推算
        today_below_ma20 = technical.get("today_below_ma20", price < ma20)
        consecutive_break = technical.get("consecutive_ma20_break", False)

        if today_below_ma20:
            if vol_ratio <= 0.7 and not consecutive_break:
                washout_verdict = (
                    "🟡 初步判定：大概率為主力洗盤（縮量跌破 MA20 + 僅 1 日，非連續）。"
                    "AI 分析師應給予「忍受震盪、守住 MA60 不破則續抱」的建議，"
                    "禁止因短暫跌破成本就叫使用者認賠。"
                )
            elif vol_ratio <= 0.7 and consecutive_break:
                washout_verdict = (
                    "🟡 縮量連續跌破 MA20（已 2 日），仍偏向洗盤，"
                    "但需密切觀察第 3 日是否回收 MA20，若持續縮量則繼續持守。"
                )
            elif vol_ratio >= 1.5 and consecutive_break:
                washout_verdict = (
                    "🔴 高風險警示：放量 + 連續 2 日跌破 MA20，"
                    "防洗盤三重濾網條件 B + C 初步成立，"
                    "需結合新聞判斷條件 A（題材是否破壞），若三條件全中應啟動止損程序。"
                )
            elif vol_ratio >= 1.5 and not consecutive_break:
                washout_verdict = (
                    "🟠 今日放量跌破 MA20（條件 C 成立），但尚未連續 2 日（條件 B 未完全成立）。"
                    "明日若收盤回收 MA20 = 假跌破，繼續持有；若再度跌破 = 條件 B 成立，需提升警戒。"
                )
            else:
                washout_verdict = (
                    "🟡 量能正常跌破 MA20，尚無明確洗盤或出貨訊號，"
                    "需觀察後續量能變化（縮量→洗盤；放量→出貨警告）。"
                )
        else:
            washout_verdict = "🟢 股價站上 MA20，無洗盤疑慮，正常追蹤均線支撐即可。"

        shares_label = (
            f"{shares} 股零股" if is_zero
            else f"{shares // 1000} 張" + (f" {shares % 1000} 零股" if shares % 1000 else "")
        )

        position_block = f"""
▌【使用者持倉資訊】（個人化分析核心數據）
  持倉狀態：持有中
  每股平均成本：{cs}{cost}
  持有數量：{shares_label}（共 {shares} 股）
  當前股價：{cs}{price}
  未實現損益：{'+' if pnl >= 0 else '-'}{cs}{abs(pnl):,.0f}（{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%）
  損益情境分類：{pnl_tier}
  {'⚠ 零股投資人：心理壓力較整張小，防守線應抓得更精準，以保護實質獲利為優先' if is_zero else ''}
{breakeven_note}

  成本與技術均線關係：
  • 成本 {cs}{cost} vs MA20 {cs}{ma20}：{'成本低於 MA20，具安全墊' if cost < ma20 else '成本高於 MA20，套在短均線上方'}
  • 成本 {cs}{cost} vs MA60 {cs}{ma60}：{'成本低於 MA60，中長線安全' if cost < ma60 else '成本套在 MA60 上方，中線承壓'}

▌【v4.0 洗盤偵測分析】（防止被主力洗出場）
  連續 MA20 跌破狀態：{'是（今日 + 昨日收盤皆在 MA20 以下，條件 B 初步成立）' if consecutive_break else '否（今日' + ('跌破但昨日未破' if today_below_ma20 else '未跌破') + '）'}
  今日量能倍數：{vol_ratio:.1f}x 均量（條件 C 門檻：≥ 1.5 倍）
  綜合洗盤研判：{washout_verdict}

  ⚙ AI 分析師核心任務（持有者）：
  ① 必須先通過防洗盤三重濾網（條件 A/B/C），才能建議認賠出場
  ② 若建議續抱，必須給出保本防守線（不得低於 {cs}{breakeven_line if breakeven_line else cost}{'；以【當日收盤價】為觸發依據，禁止依盤中插針出場' if is_us_high_price else ''}）
  ③ 所有防守價格必須是具體數字，並附上「執行前提條件」（縮量則不執行{'；美股高價股另加「以收盤價為準，勿在開盤 30 分鐘內執行」' if is_us_high_price else ''}）"""

    else:
        position_block = f"""
▌【使用者持倉資訊】
  持倉狀態：未持有，正在評估是否進場

  ⚙ AI 分析師任務（未持有者情境）：
  給出「踏空蹲點接刀策略」，必須包含：
  • 積極型進場點：MA20（{cs}{ma20}）附近，說明觸發條件（縮量回測才入）
  • 保守型進場點：MA60（{cs}{ma60}）或近期低點（{cs}{technical['low_60d']}）
  • 若 RSI 目前 > 70，明確說明不追高，等哪個回踩位再進場
  • 若為強勢題材股且動能未衰退，可建議少量零股試單蹲點"""

    # ── 新聞 ─────────────────────────────────────────────────────────────────
    news_lines = "\n".join([
        f"  [{i + 1}] {n['date']}  {n['title']}"
        for i, n in enumerate(news)
    ])

    # ── v5.0：即時修正標注 ────────────────────────────────────────────────────
    # 讓 AI 明確知道「技術指標是否已套用即時價格」，
    # 防止它因為誤以為數據延遲而在分析中保留過多不確定性語氣。
    if technical.get("price_overridden"):
        price_data_note = (
            f"⚡ 【即時股價插補】：Yahoo 延遲價 {cs}{technical['yahoo_delayed_price']} "
            f"→ 使用者輸入即時價 {cs}{price}，"
            f"MA20 / MA60 / RSI 已基於即時價重算，指標 100% 同步當前市況。"
        )
    else:
        price_data_note = (
            f"⏱ 【Yahoo 延遲數據】：使用者未輸入即時修正，"
            f"技術指標基於 Yahoo Finance 延遲快照（約落後市場 10-15 分鐘）。"
        )

    # ── 組裝完整 Context ──────────────────────────────────────────────────────
    context = f"""
════════════════════════════════════════════════════
  AI 實戰分析數據包  ｜  {ticker}  ｜  {technical['last_date']}
════════════════════════════════════════════════════
  {price_data_note}
{growth_flag}
{position_block}

▌技術面數據（{('即時修正後重算' if technical.get('price_overridden') else 'Yahoo 延遲快照')}）
  當前股價：{cs}{price}
  MA20：{cs}{ma20}  ｜  MA60：{cs}{ma60}  ｜  RSI(14)：{rsi}
  均線研判：{ma_summary}
  RSI 研判：{rsi_signal}
  量能研判：{vol_signal}
  近 5 日漲跌：{technical['change_5d_pct']}%  ｜  近 20 日漲跌：{technical['change_20d_pct']}%
  60 日最高：{cs}{technical['high_60d']}  ｜  60 日最低：{cs}{technical['low_60d']}

▌基本面數據（估值參考）
  公司：{fundamental['company_name']}（{fundamental['sector']} / {fundamental['industry']}）
  Trailing P/E：{fundamental['trailing_pe']}  ｜  Forward P/E：{fundamental['forward_pe']}
  {pe_note}
  PEG Ratio：{fundamental['peg_ratio']}
  EPS 年成長率：{fundamental['earnings_growth']}
  股息率：{fundamental['dividend_yield']}
  52 週區間：{cs}{fundamental['week52_low']} ～ {cs}{fundamental['week52_high']}
  分析師目標價：{cs}{fundamental['analyst_target']}  ｜  建議：{fundamental['recommendation']}
  Beta：{fundamental['beta']}  ｜  市值：{fundamental['market_cap']}

▌最新新聞（判斷題材是否仍在的關鍵依據）
{news_lines}

════════════════════════════════════════════════════
""".strip()

    return context


# ==================== 呼叫 Claude API ==========================================

def analyze_with_claude(
    context: str,
    ticker: str,
    position: dict,
    fundamental: dict,
    technical: dict,
    api_key: str,
    stream: bool = False,
):
    """
    傳送 Context 給 Claude API，取得持倉感知的實戰交易決策報告。

    v4.0 System Prompt 四大核心升級：
    1. 防洗盤三重濾網：明確定義條件 A/B/C，三條全中才叫使用者認賠
    2. 縮量 = 禁止執行停損：若縮量跌破均線或成本，禁止叫使用者賣出
    3. 保本防守線鐵律：獲利持倉的防守位絕對不能低於保本防守線（Context 已算好數字）
    4. 精簡輸出格式：每個防守線都必須附「執行前提條件」（有道理才執行）
    """
    client = anthropic.Anthropic(api_key=api_key)

    cs = "NT$" if fundamental.get("currency") == "TWD" else "$"
    company_name = fundamental.get("company_name", ticker)

    # ── 準備嵌入 System Prompt 的持倉描述字串 ─────────────────────────────────
    if position["holds"]:
        pnl = position["unrealized_pnl"]
        pnl_pct = position["unrealized_pnl_pct"]
        shares = position["shares"]
        cost = position["cost_basis"]
        is_zero = position["is_zero_share"]

        pnl_emoji = "📈" if pnl >= 0 else "📉"
        pnl_display = (
            f"{pnl_emoji} 獲利中 {cs}{abs(pnl):,.0f} (+{pnl_pct:.2f}%)"
            if pnl >= 0
            else f"{pnl_emoji} 套牢中 -{cs}{abs(pnl):,.0f} ({pnl_pct:.2f}%)"
        )
        shares_label = (
            f"{shares} 股零股" if is_zero
            else f"{shares // 1000} 張" + (f" {shares % 1000} 零股" if shares % 1000 else "")
        )
        holding_display = f"持有 {shares_label}，成本 {cs}{cost} 元"
        scenario = "HOLDER"

    else:
        pnl_display = ""
        holding_display = "未持有"
        is_zero = False
        cost = None
        pnl_pct = 0
        scenario = "NON_HOLDER"

    # ── System Prompt：v4.0 防洗盤大腦 ───────────────────────────────────────
    #
    # 設計哲學：
    # - 舊版的問題：AI 看到「跌破成本」就叫使用者賣，忽略量能與題材，
    #   導致使用者被主力洗盤洗出場，隨後看著股票題材還在繼續漲。
    # - v4.0 解法：把「主力洗盤判斷」的邏輯硬編碼進 System Prompt，
    #   讓 AI 在叫使用者賣之前，必須先通過三重濾網。
    # - 「執行前提條件」嵌入輸出格式本身，讓每個防守線都帶有量能條件，
    #   從根本上防止「縮量跌破就停損」的教科書式錯誤。
    # - v5.1 新增：美股高價股（≥ $50）保本防守線自適應 + 開盤震盪保護條款

    # ── 美股高價股補充規則（動態插入 System Prompt）─────────────────────────
    _is_us_hp = fundamental.get("currency") != "TWD" and technical["current_price"] >= 50
    _few_shares = (
        position["holds"]
        and position.get("shares") is not None
        and position["shares"] <= 10
    ) if _is_us_hp else False

    if _is_us_hp:
        _few_share_note = (
            f"\n\n【持股極少特別提醒（本次持股 {position.get('shares', '?')} 股）】\n"
            f"持股數量極少，每股損益對心理影響大。請在報告中明確說明：\n"
            f"「股價單日 {technical['current_price'] * 0.01:.2f} 美元（~1%）的波動完全正常，\n"
            f" 不代表趨勢反轉。請以收盤價做判斷，不要因盤中報價起伏而恐慌亂砍。」"
        ) if _few_shares else ""

        _us_hp_section = f"""
══════════════════════════════════════════════
補充：美股高價股執行規則（本次強制適用）
══════════════════════════════════════════════
當前標的股價 ≥ $50（美股高價股），以下規則強制覆蓋一般規則：

【執行時機：強制避開開盤震盪窗口】
美股無漲跌幅限制，高價股在開盤後前 30 分鐘（美東時間 09:30~10:00，
台灣時間 22:30~23:00）往往出現 1%~3% 的隨機劇烈波動（「插針」），
盤中暫時觸及防守線後立刻反彈是常態，不代表趨勢真的反轉。

「執行前提條件」中，必須逐字寫出以下內容：
  ▶ 請勿在開盤後 30 分鐘內（台灣時間 22:30-23:00）執行
  ▶ 以【當日收盤價】跌破為最終執行依據，而非盤中即時報價

【保本防守線說明】
Context 中的保本防守線已依美股高價股特性計算（含 1.5%~2% 波動寬容度）。
AI 分析師直接引用 Context 中的數字，
不得自行縮窄為台股標準（即禁止改回成本 ±0.2% 或 ±0.5%）。{_few_share_note}

"""
    else:
        _us_hp_section = ""

    system_prompt = f"""你是一位在頂級對沖基金執行動能交易（Momentum Trading）15 年的機構交易員，
專精於 AI 半導體題材股的主升浪操盤，深諳主力洗盤心理學。

你的核心信條：
「認賠必須賠得有道理。縮量跌破成本不是理由，題材破壞才是理由。」
「獲利的單子，絕對不能因為短暫震盪而讓它變成虧損的單子。」

══════════════════════════════════════════════
一、動能交易估值準則（禁止違反）
══════════════════════════════════════════════
禁止：只因 Trailing P/E 高就結論「估值昂貴，建議觀望」。
禁止：在 AI / 半導體題材股的主升段中給出「等估值合理再進場」的廢話。

必須：若 Context 標記「⚡ 動能題材」或 Forward P/E 比 Trailing P/E 低 ≥ 15%，
      以 Forward P/E + PEG Ratio 為主要估值框架，不得單憑靜態 P/E 論貴賤。

══════════════════════════════════════════════
二、防洗盤三重濾網（核心鐵律，持有者適用）
══════════════════════════════════════════════
叫使用者「認賠出場」之前，必須確認以下三個條件都同時成立。
只要有任何一個條件不成立，就不得建議認賠，改為「忍受震盪、守住均線」。

【條件 A：題材與籌碼實質變壞】
  ✓ 成立標準：近期新聞出現重大基本面利空（大客戶轉單、競爭對手搶市、
    財報大幅不如預期），或 Forward P/E 大幅上修（未來獲利預期被下修）。
  ✗ 不成立：新聞只是技術面的「股價下跌」報導，或一般市場雜音。
    此時題材仍在，條件 A 不成立，禁止因此叫使用者認賠。

【條件 B：技術面連續實質破壞】
  ✓ 成立標準：連續 2 個交易日收盤價都跌破 MA20（Context 中有「連續 MA20 跌破狀態」）。
    或放量跌破前波起漲點的關鍵 K 線低點（60 日最低點以下）。
  ✗ 不成立：僅盤中跌破，或只有 1 日收盤低於 MA20，次日有可能回收。
    單日跌破可能是洗盤，禁止因此叫使用者認賠。

【條件 C：量能異常放大（主力實質出貨訊號）】
  ✓ 成立標準：下跌當天成交量 ≥ 60 日均量的 1.5 倍（Context 中「今日量能倍數」）。
    此為主力機構大量出貨的量能特徵，而非散戶自然震盪。
  ✗ 不成立：量能萎縮（< 0.7 倍均量）或量能正常（0.7-1.5 倍均量）。
    縮量下跌是主力惡意洗盤的典型手法，禁止因縮量下跌叫使用者認賠。

★ 反向鐵律（題材在 + 縮量 = 洗盤）：
  若新聞顯示 AI / 產業題材仍然強勁，且下跌時量能萎縮（< 均量 0.7 倍），
  AI 必須明確判定「大概率為主力健康洗盤」，
  強烈建議使用者「忍受短期震盪，守住 MA60 不破則續抱」，
  即使股價短暫跌破成本，也不叫使用者認賠。

══════════════════════════════════════════════
三、保本防守線鐵律（獲利持倉適用）
══════════════════════════════════════════════
若使用者目前持有的部位處於獲利狀態（損益為正）：

規則一：短線防守位的設定，絕對不能低於 Context 中的「保本防守線」。
        （保本防守線已在 Context 中依市場類型自動計算完畢，請直接引用數字）
        違反此規則 = 把賺錢的單子放到賠錢，是最不可饒恕的交易錯誤。

規則二：若 Context 中有「獲利保護線」，優先將此線設為移動停利的防守位。
        （獲利保護線 = 鎖住一半獲利的價格，確保最差也是賺 50% 的利潤出場）

規則三：零股投資人（< 1000 股）的防守線應設得更緊，
        因為他們的槓桿和心理壓力都較小，更容易執行精準的停利。

{_us_hp_section}══════════════════════════════════════════════
四、使用者持倉情境
══════════════════════════════════════════════
情境：{"【HOLDER — 持有者策略】" if scenario == "HOLDER" else "【NON_HOLDER — 踏空接刀策略】"}
持倉：{holding_display}
{"損益：" + pnl_display if scenario == "HOLDER" else ""}

{"持有者分析要點：" if scenario == "HOLDER" else "未持有者分析要點："}
{"① 先做防洗盤三重濾網判斷，再決定是否建議出場" if scenario == "HOLDER" else "① 給出積極型（MA20）與保守型（MA60）兩個具體進場方案"}
{"② 依損益分類走對應策略（Context 中已標好分類）" if scenario == "HOLDER" else "② 若 RSI > 70 禁止追高，明確說明等哪個回踩位才進場"}
{"③ 獲利持倉：防守位不得低於保本防守線（Context 中已給數字，直接引用）" if scenario == "HOLDER" else "③ 若題材動能強且 RSI < 70，可建議少量零股試單蹲點"}
{"④ 零股投資人：防守線抓得更精準，以保護實質利潤為優先" if is_zero else ""}

══════════════════════════════════════════════
五、輸出格式（嚴格照抄，不得刪減任何區塊）
══════════════════════════════════════════════
輸出純 Markdown，全程繁體中文，語氣直接犀利，不說廢話。
所有價格必須是具體數字（禁止只寫「附近」「左右」而不附數字）。
「風險防守與認賠執行線」區塊中，每一條防守線都必須附上「執行前提條件」，
明確說明在什麼量能條件下才執行（縮量則不執行，是最重要的保護）。
從 `## 🎯` 標題直接開始輸出，完整輸出至免責聲明結束，中途不得截斷。

---

## 🎯 AI 實戰交易決策：[填入公司名稱（股票代號）]

**當前股價：** [填入 {cs}X.XX] | **您的持倉：** [填入 {holding_display}]
{"**目前損益：** [填入 " + pnl_display + "]" if scenario == "HOLDER" else ""}

---

### 📌 【核心行動決策】：【填入決策標題，例如：動能續抱抗洗盤 / 題材破壞果斷停損 / 守住保本線觀察】

[2-3 句說明核心邏輯。持有者：必須說明「防洗盤三重濾網判斷結果」是否叫出場。
未持有者：說明當前題材熱度與最佳進場時機。不說廢話，直接切入結論。]

---

### 📈 您的實戰操作指南

- [第一步：具體行動 + 觸發條件或價格]
- [第二步：具體行動 + 觸發條件或價格]
- [第三步：具體行動 + 觸發條件或價格]

---

### 🛑 風險防守與認賠執行線（有道理才執行）

* **關鍵防守價**：{cs}X.XX
* **執行前提條件**：[例如：必須連續 2 日收盤跌破此位 + 當日量能 ≥ 均量 1.5 倍，才執行停損；若縮量跌破，判定為洗盤，不執行]
* **決策理由**：[1-2 句說明為何選此價位，依據 MA / 保本防守線 / 成本位 / 60 日低點]

---

> ⚠️ **免責聲明**：本報告由 AI 自動生成，僅供參考，不構成投資建議。投資一定有風險，請依自身風險承受能力決策。"""

    # ── User Message：注入數據 + 四重確認規則 ─────────────────────────────────
    user_message = f"""以下是 {ticker}（{company_name}）的完整即時數據，請輸出實戰交易決策報告：

{context}

輸出前四重確認：
1. 若股息率標示「數據異常」，基本面分析完全不引用此數字
2. 若有「⚡ 動能題材標記」，以 Forward P/E + PEG 為主要估值框架，不得以高 P/E 喊貴
3. 風險防守線的「執行前提條件」必須包含量能條件
   （縮量跌破 = 洗盤 = 不執行；放量 ≥ 1.5 倍均量 + 連續 2 日跌破 = 才考慮執行）
4. 若使用者持倉為獲利狀態，防守價格不得低於 Context 中的「保本防守線」
   （參考：MA20={cs}{technical['ma20']}，MA60={cs}{technical['ma60']}，60日低點={cs}{technical['low_60d']}）
5. 從 `## 🎯` 直接開始輸出，完整輸出至免責聲明結束，不得截斷或添加前後語。"""

    if stream:
        # ── Streamlit 串流模式：回傳 generator 給 st.write_stream() ──────────
        def _stream_generator():
            with client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            ) as s:
                for text in s.text_stream:
                    yield text
        return _stream_generator()
    else:
        # ── CLI 模式：同步等待，印進度提示後回傳完整文字 ─────────────────────
        print(f"\n  正在呼叫 Claude API（{CLAUDE_MODEL}）進行分析...")
        print("  預計需要 20-50 秒，請耐心等候...\n")

        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        return message.content[0].text


# ==================== 主程式入口 =============================================

def main():
    print("\n" + "═" * 62)
    print("    台股 / 美股  AI 實戰交易決策工具  v5.0")
    print("    真·即時插補 × 防洗盤三重濾網 × 移動停利保本")
    print("═" * 62)

    # Step 0：取得 API Key（執行時輸入，程式碼不存 Key）
    api_key = get_api_key()

    # Step 1：輸入股票代號
    print("\n  台股：2454.TW（聯發科）、5274.TW（信驊）、2330.TW（台積電）")
    print("  美股：NVDA（輝達）、AMD、AAPL（蘋果）、MSFT（微軟）\n")
    ticker = input("  請輸入股票代號：").strip().upper()

    if not ticker:
        print("❌  股票代號不能為空！")
        sys.exit(1)

    print(f"\n  開始撈取 {ticker} 數據...\n" + "─" * 62)

    try:
        # ── Step 2a：撈取原始 K 線 DataFrame（不計算任何指標）────────────────
        hist = fetch_raw_hist(ticker)

        # ── Step 2b：撈取基本面與新聞（與 K 線並行完成）─────────────────────
        fundamental_data = fetch_fundamental_data(ticker)
        news_data = fetch_news(ticker, max_news=3)

        cs = "NT$" if fundamental_data["currency"] == "TWD" else "$"

        # ── Step 2c：即時股價校正（v5.0 核心）──────────────────────────────
        # 先取 Yahoo 延遲收盤價顯示給使用者，再詢問是否輸入即時價
        yahoo_delayed = round(float(hist["Close"].iloc[-1]), 2)
        realtime_price = ask_realtime_price(yahoo_delayed, fundamental_data["currency"])

        # ── Step 2d：先覆蓋 DataFrame，再計算所有技術指標 ────────────────────
        # 這是 v5.0 的核心流程：Patch → Calculate，順序絕對不能顛倒
        technical_data = calculate_technical_from_hist(hist, realtime_price)

        # ── Step 3：詢問持倉狀態（使用修正後的即時價計算損益）──────────────
        position = get_position_info(
            ticker,
            technical_data["current_price"],   # 已是即時修正後的價格
            fundamental_data["currency"],
        )

        # ── 顯示數據摘要 ──────────────────────────────────────────────────
        print("\n" + "─" * 62)
        print("  ✅  數據就緒，摘要如下：")

        # v5.0：顯示是否套用即時修正
        if technical_data["price_overridden"]:
            print(f"  ⚡  股價：Yahoo 延遲 {cs}{technical_data['yahoo_delayed_price']}"
                  f" → 即時修正 {cs}{technical_data['current_price']}")
        else:
            print(f"     股價：{cs}{technical_data['current_price']}（Yahoo 延遲數據）")

        print(f"     MA20 {cs}{technical_data['ma20']}"
              f"  ｜  MA60 {cs}{technical_data['ma60']}"
              f"  ｜  RSI {technical_data['rsi']}"
              f"  {'← 已即時重算' if technical_data['price_overridden'] else ''}")
        print(f"     Trailing P/E {fundamental_data['trailing_pe']}"
              f"  ｜  Forward P/E {fundamental_data['forward_pe']}"
              f"  ｜  PEG {fundamental_data['peg_ratio'][:3] if fundamental_data['peg_ratio'] != 'N/A' else 'N/A'}")

        if position["holds"]:
            pnl     = position["unrealized_pnl"]
            pnl_pct = position["unrealized_pnl_pct"]
            status  = "獲利" if pnl >= 0 else "套牢"
            print(f"     持倉損益：{status} "
                  f"{'+' if pnl >= 0 else '-'}{cs}{abs(pnl):,.0f}"
                  f"（{'+' if pnl_pct >= 0 else ''}{pnl_pct:.2f}%）")
            # 連續 MA20 跌破偵測結果（v4.0 保留）
            if technical_data.get("consecutive_ma20_break"):
                print("  🔴  偵測到連續 2 日跌破 MA20（條件 B 初步成立），洗盤分析啟動")
            elif technical_data.get("today_below_ma20"):
                print("  🟡  今日跌破 MA20 但昨日未破（單日，可能為洗盤），持續觀察")
            else:
                print("  🟢  股價仍在 MA20 之上，無洗盤疑慮")
        else:
            print("     持倉狀態：未持有，將提供接刀進場策略")

        if fundamental_data.get("is_high_growth") or fundamental_data.get("pe_compression_note"):
            print("  ⚡  偵測到高成長 / AI 題材股，啟用動能交易估值框架")

        print("─" * 62)

        # Step 4：打包 Context
        context = build_analysis_context(
            ticker, technical_data, fundamental_data, news_data, position
        )

        # Step 5：呼叫 Claude API
        analysis_report = analyze_with_claude(
            context, ticker, position, fundamental_data, technical_data, api_key
        )

        # Step 6：印出報告
        print(analysis_report)
        print()

        # Step 7：儲存為 .md 檔（方便用 Markdown 閱讀器查看）
        save_choice = input("  是否儲存報告為 .md 文字檔？(y/n)：").strip().lower()
        if save_choice == "y":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{ticker}_實戰報告_{timestamp}.md"
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(analysis_report)
            print(f"\n  ✅  報告已儲存至：{filepath}")

    except ValueError as e:
        print(f"\n❌  數據錯誤：{e}")
        sys.exit(1)
    except anthropic.AuthenticationError:
        print("\n❌  API Key 驗證失敗，請確認金鑰是否正確且有效。")
        sys.exit(1)
    except anthropic.RateLimitError:
        print("\n❌  API 呼叫頻率超限，請稍後再試。")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n  使用者中止。")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌  未預期錯誤：{type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
