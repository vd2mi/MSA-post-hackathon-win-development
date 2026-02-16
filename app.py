"""
MSA — Market Sentiment Analyzer
FastAPI backend for stock analysis deployed on Hugging Face Spaces.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx
import numpy as np
import yfinance as yf
from bs4 import BeautifulSoup
from cachetools import TTLCache
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

load_dotenv()

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
HF_API_TOKEN: str = os.environ.get("HF_API_TOKEN", "")
CACHE_TTL_SECONDS: int = 300
FINBERT_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("msa")

app = FastAPI(
    title="MSA — Market Sentiment Analyzer",
    description=(
        "Stock analysis API: historical data, sentiment, news, and GPT-4 insights. "
        "Results are cached for 5 minutes per ticker."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_analysis_cache: TTLCache[str, dict[str, Any]] = TTLCache(
    maxsize=256, ttl=CACHE_TTL_SECONDS
)


class MovingAverages(BaseModel):
    sma_50: float | None = Field(None, description="50-day Simple Moving Average")
    sma_200: float | None = Field(None, description="200-day Simple Moving Average")
    signal: str | None = Field(
        None,
        description="'Golden Cross' if SMA-50 > SMA-200, 'Death Cross' otherwise",
    )


class FearGreed(BaseModel):
    value: int = Field(50, description="Fear & Greed index (0-100)")
    label: str = Field("Neutral", description="Human-readable label, e.g. 'Greed'")


class NewsHeadline(BaseModel):
    title: str
    publisher: str | None = None
    link: str | None = None
    published: str | None = None
    sentiment: str | None = Field(None, description="FinBERT: positive / negative / neutral")
    sentiment_score: float | None = Field(None, description="FinBERT confidence 0-1")


class NewsSentiment(BaseModel):
    positive: int = Field(0, description="Number of positive headlines")
    negative: int = Field(0, description="Number of negative headlines")
    neutral: int = Field(0, description="Number of neutral headlines")
    avg_score: float = Field(0.5, description="Average sentiment score (0=bearish, 1=bullish)")
    label: str = Field("Neutral", description="Overall: Bullish / Bearish / Neutral / Mixed")


class TechnicalIndicators(BaseModel):
    rsi_14: float | None = Field(None, description="14-day RSI")
    rsi_signal: str | None = Field(None, description="Oversold / Neutral / Overbought")
    macd: float | None = Field(None, description="MACD line value")
    macd_signal: float | None = Field(None, description="MACD signal line value")
    macd_histogram: float | None = Field(None, description="MACD histogram")
    macd_trend: str | None = Field(None, description="Bullish / Bearish")
    overall_signal: str | None = Field(None, description="Buy / Sell / Neutral")
    score: int = Field(50, description="0-100 gauge score: 0=Strong Sell, 100=Strong Buy")


class AnalystRating(BaseModel):
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0
    total: int = 0
    recommendation: str | None = Field(None, description="Overall recommendation label")
    score: int = Field(50, description="0-100 gauge score: 0=Strong Sell, 100=Strong Buy")


class PriceTarget(BaseModel):
    current: float | None = None
    daily_change: float | None = Field(None, description="Today's $ change")
    daily_change_pct: float | None = Field(None, description="Today's % change")
    target_mean: float | None = None
    target_high: float | None = None
    target_low: float | None = None
    upside_pct: float | None = Field(None, description="% upside to mean target")


class BollingerBands(BaseModel):
    upper: float | None = Field(None, description="Upper Bollinger Band (SMA20 + 2*stddev)")
    middle: float | None = Field(None, description="Middle band (SMA 20)")
    lower: float | None = Field(None, description="Lower Bollinger Band (SMA20 - 2*stddev)")
    bandwidth: float | None = Field(None, description="Band width as % of middle band")
    position: str | None = Field(None, description="Price position: Upper Band / Mid Band / Lower Band")
    squeeze: bool = Field(False, description="True if bandwidth is historically low (squeeze)")


class OBVAnalysis(BaseModel):
    obv_current: float | None = Field(None, description="Current On-Balance Volume")
    obv_trend: str | None = Field(None, description="Rising / Falling / Flat")
    price_trend: str | None = Field(None, description="Rising / Falling / Flat (last 20 days)")
    divergence: str | None = Field(None, description="Accumulation / Distribution / Confirmation / None")


class VolumeProfile(BaseModel):
    avg_volume_20d: int | None = Field(None, description="20-day average volume")
    latest_volume: int | None = Field(None, description="Most recent day's volume")
    volume_ratio: float | None = Field(None, description="Latest volume / 20-day avg")
    spike: bool = Field(False, description="True if volume > 1.5x average")
    spike_days: int = Field(0, description="Number of spike days in last 20")


class DenoisedTrend(BaseModel):
    slope: float | None = Field(None, description="Denoised price velocity (% per day)")
    slope_direction: str | None = Field(None, description="Rising / Falling / Flat")
    acceleration: float | None = Field(None, description="Rate of velocity change (% per day²)")
    momentum_exhaustion: bool = Field(False, description="True when denoised slope diverges from RSI")
    exhaustion_type: str | None = Field(None, description="Bullish Exhaustion / Bearish Exhaustion")
    denoised_prices: list[float] = Field(default_factory=list, description="Last 30 denoised closing prices (oldest first)")


class ZScoreAnalysis(BaseModel):
    zscore: float | None = Field(None, description="Current 20-day rolling Z-Score")
    mean_20d: float | None = Field(None, description="20-day rolling mean price")
    stddev_20d: float | None = Field(None, description="20-day rolling standard deviation")
    signal: str | None = Field(None, description="Statistical signal")
    reversal_probability: float | None = Field(None, description="Probability of mean reversion (0-100%)")


class InferenceWeights(BaseModel):
    technical_score: float = Field(50, description="Technical component (0-100) — 35% weight")
    sentiment_score: float = Field(50, description="Sentiment component (0-100) — 25% weight")
    analyst_score: float = Field(50, description="Analyst component (0-100) — 20% weight")
    volume_score: float = Field(50, description="Volume component (0-100) — 20% weight")
    composite_score: float = Field(50, description="Final weighted composite (0-100)")
    composite_signal: str = Field("Neutral", description="Buy / Sell / Hold / Watch based on composite")


class GPTInsight(BaseModel):
    actionable_insight: str = Field(..., description="GPT-4 actionable recommendation")
    confidence_score: int = Field(
        ..., ge=0, le=100, description="Confidence score 0-100"
    )
    reasoning: str = Field(..., description="Brief reasoning behind the recommendation")


class PricePoint(BaseModel):
    date: str
    close: float


class AnalysisResponse(BaseModel):
    ticker: str
    timestamp: str
    cached: bool = Field(False, description="True if this result came from cache")
    moving_averages: MovingAverages
    technicals: TechnicalIndicators | None = None
    bollinger_bands: BollingerBands | None = None
    obv_analysis: OBVAnalysis | None = None
    volume_profile: VolumeProfile | None = None
    denoised_trend: DenoisedTrend | None = None
    zscore_analysis: ZScoreAnalysis | None = None
    inference_weights: InferenceWeights | None = None
    analyst_ratings: AnalystRating | None = None
    price_target: PriceTarget | None = None
    fear_greed: FearGreed
    news: list[NewsHeadline]
    news_sentiment: NewsSentiment | None = None
    gpt_analysis: GPTInsight | None = None
    price_history: list[PricePoint] = Field(default_factory=list, description="Last 30 days of closing prices (oldest first)")


def _yf_fetch_history(ticker: str, period: str = "1y") -> list[dict[str, Any]]:
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)

    if df.empty:
        raise ValueError(f"yfinance returned no data for {ticker}")

    rows: list[dict[str, Any]] = []
    for date, row in df.iterrows():
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": int(row["Volume"]),
            }
        )

    rows.reverse()
    return rows


async def fetch_daily_ohlcv(ticker: str) -> list[dict[str, Any]]:
    try:
        return await asyncio.to_thread(_yf_fetch_history, ticker.upper())
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch historical data for {ticker}: {exc}",
        ) from exc


def calculate_smas(daily_rows: list[dict[str, Any]]) -> MovingAverages:
    closes = [r["close"] for r in daily_rows]

    sma_50: float | None = None
    sma_200: float | None = None

    if len(closes) >= 50:
        sma_50 = round(sum(closes[:50]) / 50, 4)
    if len(closes) >= 200:
        sma_200 = round(sum(closes[:200]) / 200, 4)

    signal: str | None = None
    if sma_50 is not None and sma_200 is not None:
        signal = "Golden Cross" if sma_50 > sma_200 else "Death Cross"

    return MovingAverages(sma_50=sma_50, sma_200=sma_200, signal=signal)


def calculate_technicals(daily_rows: list[dict[str, Any]]) -> TechnicalIndicators:
    closes = [r["close"] for r in daily_rows]

    # RSI-14
    rsi_14: float | None = None
    rsi_signal: str | None = None
    if len(closes) >= 15:
        gains, losses = [], []
        for i in range(1, 15):
            delta = closes[i - 1] - closes[i]  # rows are newest-first
            gains.append(max(delta, 0))
            losses.append(max(-delta, 0))
        avg_gain = sum(gains) / 14
        avg_loss = sum(losses) / 14
        if avg_loss == 0:
            rsi_14 = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_14 = round(100 - (100 / (1 + rs)), 2)
        if rsi_14 <= 30:
            rsi_signal = "Oversold"
        elif rsi_14 >= 70:
            rsi_signal = "Overbought"
        else:
            rsi_signal = "Neutral"

    # MACD (12, 26, 9) — we need at least ~35 days of data
    macd_val: float | None = None
    macd_sig: float | None = None
    macd_hist: float | None = None
    macd_trend: str | None = None

    if len(closes) >= 35:
        ordered = list(reversed(closes))  # oldest-first for EMA calc

        def ema(data: list[float], span: int) -> list[float]:
            k = 2 / (span + 1)
            result = [data[0]]
            for price in data[1:]:
                result.append(price * k + result[-1] * (1 - k))
            return result

        ema12 = ema(ordered, 12)
        ema26 = ema(ordered, 26)
        macd_line = [a - b for a, b in zip(ema12[25:], ema26[25:])]

        if len(macd_line) >= 9:
            signal_line = ema(macd_line, 9)
            macd_val = round(macd_line[-1], 4)
            macd_sig = round(signal_line[-1], 4)
            macd_hist = round(macd_val - macd_sig, 4)
            macd_trend = "Bullish" if macd_hist > 0 else "Bearish"

    # gauge score: map RSI + MACD into 0-100 (0=Strong Sell, 100=Strong Buy)
    # RSI contributes: <20 → +2, 20-30 → +1, 30-70 → 0, 70-80 → -1, >80 → -2
    rsi_pts = 0
    if rsi_14 is not None:
        if rsi_14 < 20: rsi_pts = 2
        elif rsi_14 < 30: rsi_pts = 1
        elif rsi_14 > 80: rsi_pts = -2
        elif rsi_14 > 70: rsi_pts = -1

    macd_pts = 0
    if macd_hist is not None:
        if macd_hist > 0: macd_pts = 1
        else: macd_pts = -1

    raw = rsi_pts + macd_pts  # range [-3, +3]
    gauge = int(((raw + 3) / 6) * 100)
    gauge = max(0, min(100, gauge))

    if gauge >= 70:
        overall = "Buy" if gauge < 85 else "Strong Buy"
    elif gauge <= 30:
        overall = "Sell" if gauge > 15 else "Strong Sell"
    else:
        overall = "Neutral"

    return TechnicalIndicators(
        rsi_14=rsi_14,
        rsi_signal=rsi_signal,
        macd=macd_val,
        macd_signal=macd_sig,
        macd_histogram=macd_hist,
        macd_trend=macd_trend,
        overall_signal=overall,
        score=gauge,
    )


def calculate_bollinger(daily_rows: list[dict[str, Any]]) -> BollingerBands:
    """Bollinger Bands: 20-period SMA +/- 2 standard deviations. daily_rows[0] = newest."""
    try:
        ordered = [r["close"] for r in reversed(daily_rows)]  # oldest first
        if len(ordered) < 20:
            return BollingerBands()

        window = ordered[-20:]
        sma20 = sum(window) / 20
        variance = sum((x - sma20) ** 2 for x in window) / 20
        stddev = variance ** 0.5

        upper = round(sma20 + 2 * stddev, 2)
        lower = round(sma20 - 2 * stddev, 2)
        middle = round(sma20, 2)
        bw = round((upper - lower) / middle * 100, 2) if middle else None

        # detect squeeze: compare current bandwidth to avg of last 100 days' bandwidths
        squeeze = False
        if len(ordered) >= 100:
            bws = []
            for i in range(20, min(len(ordered) + 1, 101)):
                w = ordered[i - 20:i]
                m = sum(w) / 20
                sd = (sum((x - m) ** 2 for x in w) / 20) ** 0.5
                if m > 0:
                    bws.append((m + 2 * sd - (m - 2 * sd)) / m * 100)
            if bws and bw is not None:
                avg_bw = sum(bws) / len(bws)
                squeeze = bw < avg_bw * 0.5

        current_price = ordered[-1]
        if current_price >= upper * 0.98:
            position = "Upper Band"
        elif current_price <= lower * 1.02:
            position = "Lower Band"
        else:
            position = "Mid Band"

        return BollingerBands(
            upper=upper, middle=middle, lower=lower,
            bandwidth=bw, position=position, squeeze=squeeze,
        )
    except Exception as exc:
        logger.warning("Bollinger Bands calculation failed: %s", exc)
        return BollingerBands()


def calculate_obv(daily_rows: list[dict[str, Any]]) -> OBVAnalysis:
    """On-Balance Volume with trend and divergence detection. daily_rows[0] = newest."""
    try:
        ordered = list(reversed(daily_rows))  # oldest first
        if len(ordered) < 20:
            return OBVAnalysis()

        # build OBV series
        obv = [0.0]
        for i in range(1, len(ordered)):
            if ordered[i]["close"] > ordered[i - 1]["close"]:
                obv.append(obv[-1] + ordered[i]["volume"])
            elif ordered[i]["close"] < ordered[i - 1]["close"]:
                obv.append(obv[-1] - ordered[i]["volume"])
            else:
                obv.append(obv[-1])

        obv_now = obv[-1]

        # OBV trend over last 20 days: linear regression slope direction
        obv_20 = obv[-20:]
        obv_slope = obv_20[-1] - obv_20[0]
        if obv_slope > 0:
            obv_trend = "Rising"
        elif obv_slope < 0:
            obv_trend = "Falling"
        else:
            obv_trend = "Flat"

        # price trend over last 20 days
        closes_20 = [r["close"] for r in ordered[-20:]]
        price_slope = closes_20[-1] - closes_20[0]
        pct_change = abs(price_slope) / closes_20[0] * 100 if closes_20[0] else 0
        if pct_change < 2:
            price_trend = "Flat"
        elif price_slope > 0:
            price_trend = "Rising"
        else:
            price_trend = "Falling"

        # divergence detection
        if obv_trend == "Rising" and price_trend == "Flat":
            divergence = "Accumulation"
        elif obv_trend == "Rising" and price_trend == "Falling":
            divergence = "Accumulation"
        elif obv_trend == "Falling" and price_trend == "Rising":
            divergence = "Distribution"
        elif obv_trend == "Falling" and price_trend == "Flat":
            divergence = "Distribution"
        elif obv_trend == price_trend:
            divergence = "Confirmation"
        else:
            divergence = "None"

        return OBVAnalysis(
            obv_current=round(obv_now),
            obv_trend=obv_trend,
            price_trend=price_trend,
            divergence=divergence,
        )
    except Exception as exc:
        logger.warning("OBV calculation failed: %s", exc)
        return OBVAnalysis()


def calculate_volume_profile(daily_rows: list[dict[str, Any]]) -> VolumeProfile:
    """Volume analysis: 20-day average, spikes, ratio. daily_rows[0] = newest."""
    try:
        if len(daily_rows) < 20:
            return VolumeProfile()

        last_20 = daily_rows[:20]  # newest 20 days
        volumes = [r["volume"] for r in last_20]
        avg_vol = int(sum(volumes) / 20)
        latest_vol = volumes[0]
        ratio = round(latest_vol / avg_vol, 2) if avg_vol > 0 else 0
        spike = ratio > 1.5
        spike_days = sum(1 for v in volumes if avg_vol > 0 and v > avg_vol * 1.5)

        return VolumeProfile(
            avg_volume_20d=avg_vol,
            latest_volume=latest_vol,
            volume_ratio=ratio,
            spike=spike,
            spike_days=spike_days,
        )
    except Exception as exc:
        logger.warning("Volume profile calculation failed: %s", exc)
        return VolumeProfile()


BULL_KEYWORDS = {
    "surge", "soar", "rally", "gain", "beat", "upgrade", "rise", "jump",
    "bull", "record", "high", "buy", "outperform", "breakout", "growth",
}
BEAR_KEYWORDS = {
    "crash", "fall", "drop", "plunge", "sell", "downgrade", "cut", "loss",
    "bear", "low", "miss", "warn", "risk", "decline", "slump", "fear",
}


def _savgol_smooth(prices: np.ndarray, window: int = 21, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay polynomial smoothing via numpy (no scipy dependency).
    Computes convolution coefficients from the Vandermonde pseudo-inverse."""
    n = len(prices)
    if n < window:
        window = n if n % 2 == 1 else n - 1
    if window < polyorder + 2:
        return prices.copy()
    half = window // 2
    x = np.arange(-half, half + 1, dtype=float)
    A = np.vander(x, N=polyorder + 1, increasing=True)
    smooth_coeffs = np.linalg.pinv(A)[0]
    padded = np.pad(prices, half, mode="edge")
    return np.convolve(padded, smooth_coeffs[::-1], mode="valid")


def calculate_denoised_trend(
    daily_rows: list[dict[str, Any]], rsi: float | None,
) -> DenoisedTrend:
    """Savitzky-Golay denoising to extract price velocity and momentum exhaustion."""
    try:
        ordered = [r["close"] for r in reversed(daily_rows)]
        n = len(ordered)
        if n < 30:
            return DenoisedTrend()

        prices = np.array(ordered, dtype=float)
        smoothed = _savgol_smooth(prices)

        recent = smoothed[-10:]
        x10 = np.arange(10, dtype=float)
        fit = np.polyfit(x10, recent, 1)
        slope = float(fit[0])
        slope_pct = (slope / float(prices[-1])) * 100 if prices[-1] > 0 else 0.0

        accel_pct = 0.0
        if n >= 30:
            prev = smoothed[-20:-10]
            prev_fit = np.polyfit(np.arange(10, dtype=float), prev, 1)
            accel = slope - float(prev_fit[0])
            accel_pct = (accel / float(prices[-1])) * 100 if prices[-1] > 0 else 0.0

        if abs(slope_pct) < 0.05:
            direction = "Flat"
        elif slope_pct > 0:
            direction = "Rising"
        else:
            direction = "Falling"

        exhaustion = False
        exhaustion_type = None
        if rsi is not None:
            if slope_pct > 0.1 and rsi > 65 and accel_pct < -0.01:
                exhaustion = True
                exhaustion_type = "Bullish Exhaustion"
            elif slope_pct < -0.1 and rsi < 35 and accel_pct > 0.01:
                exhaustion = True
                exhaustion_type = "Bearish Exhaustion"

        hist_len = min(30, n)
        denoised_last = smoothed[-hist_len:].tolist()

        return DenoisedTrend(
            slope=round(slope_pct, 4),
            slope_direction=direction,
            acceleration=round(accel_pct, 4),
            momentum_exhaustion=exhaustion,
            exhaustion_type=exhaustion_type,
            denoised_prices=[round(float(p), 2) for p in denoised_last],
        )
    except Exception as exc:
        logger.warning("Denoised trend calculation failed: %s", exc)
        return DenoisedTrend()


def calculate_zscore(daily_rows: list[dict[str, Any]]) -> ZScoreAnalysis:
    """20-day rolling Z-Score for statistical mean reversion signals."""
    try:
        ordered = [r["close"] for r in reversed(daily_rows)]
        if len(ordered) < 20:
            return ZScoreAnalysis()

        window = ordered[-20:]
        mean = sum(window) / 20
        variance = sum((x - mean) ** 2 for x in window) / 20
        stddev = variance ** 0.5

        current = ordered[-1]
        zscore = (current - mean) / stddev if stddev > 0 else 0.0

        abs_z = abs(zscore)
        reversal_prob = round(math.erf(abs_z / math.sqrt(2)) * 100, 1)

        if zscore > 2.0:
            signal = "Overbought — Pullback Likely (>2σ)"
        elif zscore > 1.5:
            signal = "Stretched — Watch for Reversal"
        elif zscore < -2.0:
            signal = "Oversold — Bounce Likely (>2σ)"
        elif zscore < -1.5:
            signal = "Compressed — Watch for Bounce"
        else:
            signal = "Normal Range"

        return ZScoreAnalysis(
            zscore=round(zscore, 3),
            mean_20d=round(mean, 2),
            stddev_20d=round(stddev, 2),
            signal=signal,
            reversal_probability=reversal_prob,
        )
    except Exception as exc:
        logger.warning("Z-Score calculation failed: %s", exc)
        return ZScoreAnalysis()


def calculate_inference_weights(
    technicals: TechnicalIndicators,
    denoised: DenoisedTrend,
    zscore: ZScoreAnalysis,
    fear_greed: FearGreed,
    news_sent: NewsSentiment,
    analyst: AnalystRating,
    obv: OBVAnalysis,
    volume: VolumeProfile,
) -> InferenceWeights:
    """Pre-calculate the 4-pillar weighted inference scores."""
    rsi_macd = technicals.score if technicals else 50
    slope_score = 50.0
    if denoised.slope is not None:
        slope_score = max(0.0, min(100.0, 50 + denoised.slope * 25))

    z_component = 50.0
    if zscore.zscore is not None:
        z_component = max(0.0, min(100.0, 50 - zscore.zscore * 20))

    technical_score = rsi_macd * 0.40 + slope_score * 0.35 + z_component * 0.25

    contrarian_fg = 100 - fear_greed.value
    finbert_score = news_sent.avg_score * 100
    sentiment_score = contrarian_fg * 0.60 + finbert_score * 0.40

    analyst_score_val = float(analyst.score) if analyst else 50.0

    vol_base = 50.0
    if obv.divergence == "Accumulation":
        vol_base = 80.0
    elif obv.divergence == "Distribution":
        vol_base = 20.0
    elif obv.divergence == "Confirmation":
        vol_base = 65.0 if obv.obv_trend == "Rising" else 35.0

    if volume and volume.volume_ratio:
        if volume.volume_ratio > 1.5 and obv.obv_trend == "Rising":
            vol_base = min(100, vol_base + 12)
        elif volume.volume_ratio > 1.5 and obv.obv_trend == "Falling":
            vol_base = max(0, vol_base - 12)

    composite = (
        technical_score * 0.35
        + sentiment_score * 0.25
        + analyst_score_val * 0.20
        + vol_base * 0.20
    )

    if composite >= 70:
        sig = "Buy"
    elif composite >= 55:
        sig = "Hold"
    elif composite >= 40:
        sig = "Watch"
    else:
        sig = "Sell"

    return InferenceWeights(
        technical_score=round(technical_score, 1),
        sentiment_score=round(sentiment_score, 1),
        analyst_score=round(analyst_score_val, 1),
        volume_score=round(vol_base, 1),
        composite_score=round(composite, 1),
        composite_signal=sig,
    )


def _yf_fetch_analyst_ratings(ticker: str) -> AnalystRating:
    try:
        stock = yf.Ticker(ticker.upper())
        rec = stock.recommendations
        if rec is None or rec.empty:
            return AnalystRating()

        latest = rec.iloc[-1]
        sb = int(latest.get("strongBuy", 0))
        b = int(latest.get("buy", 0))
        h = int(latest.get("hold", 0))
        s = int(latest.get("sell", 0))
        ss = int(latest.get("strongSell", 0))
        total = sb + b + h + s + ss

        if total == 0:
            label = "No Data"
        else:
            buy_pct = (sb + b) / total
            sell_pct = (s + ss) / total
            if buy_pct >= 0.6:
                label = "Strong Buy" if sb > b else "Buy"
            elif sell_pct >= 0.6:
                label = "Strong Sell" if ss > s else "Sell"
            else:
                label = "Hold"

        # weighted score: strong_buy=5, buy=4, hold=3, sell=2, strong_sell=1
        if total > 0:
            weighted = (sb * 5 + b * 4 + h * 3 + s * 2 + ss * 1) / total
            a_score = int(((weighted - 1) / 4) * 100)
            a_score = max(0, min(100, a_score))
        else:
            a_score = 50

        return AnalystRating(
            strong_buy=sb, buy=b, hold=h, sell=s, strong_sell=ss,
            total=total, recommendation=label, score=a_score,
        )
    except Exception as exc:
        logger.warning("Analyst ratings fetch failed for %s: %s", ticker, exc)
        return AnalystRating()


def _yf_fetch_price_target(ticker: str) -> PriceTarget:
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info or {}
        
        current = (
            info.get("currentPrice") 
            or info.get("regularMarketPrice") 
            or info.get("previousClose")
        )
        
        daily_chg = info.get("regularMarketChange")
        daily_chg_pct = info.get("regularMarketChangePercent")
        
        target_mean = (
            info.get("targetMeanPrice")
            or info.get("targetMedianPrice")
            or info.get("recommendationMean")
        )
        target_high = (
            info.get("targetHighPrice")
            or info.get("targetMaxPrice")
        )
        target_low = (
            info.get("targetLowPrice")
            or info.get("targetMinPrice")
        )

        upside = None
        if current and target_mean:
            try:
                upside = round(((target_mean - current) / current) * 100, 2)
            except (TypeError, ZeroDivisionError):
                pass

        return PriceTarget(
            current=round(float(current), 2) if current else None,
            daily_change=round(float(daily_chg), 2) if daily_chg is not None else None,
            daily_change_pct=round(float(daily_chg_pct), 2) if daily_chg_pct is not None else None,
            target_mean=round(float(target_mean), 2) if target_mean else None,
            target_high=round(float(target_high), 2) if target_high else None,
            target_low=round(float(target_low), 2) if target_low else None,
            upside_pct=upside,
        )
    except Exception as exc:
        logger.warning("Price target fetch failed for %s: %s", ticker, exc)
        return PriceTarget()


async def fetch_analyst_ratings(ticker: str) -> AnalystRating:
    return await asyncio.to_thread(_yf_fetch_analyst_ratings, ticker.upper())


async def fetch_price_target(ticker: str) -> PriceTarget:
    return await asyncio.to_thread(_yf_fetch_price_target, ticker.upper())


FEAR_GREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
_NEUTRAL_FALLBACK = FearGreed(value=50, label="Neutral")


def _fg_label(value: int) -> str:
    if value <= 25:
        return "Extreme Fear"
    if value <= 45:
        return "Fear"
    if value <= 55:
        return "Neutral"
    if value <= 75:
        return "Greed"
    return "Extreme Greed"


async def fetch_fear_greed() -> FearGreed:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(FEAR_GREED_URL, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        fg = data.get("fear_and_greed", {})
        score = fg.get("score")
        rating = fg.get("rating")

        if score is not None:
            return FearGreed(value=int(round(score)), label=rating or _fg_label(int(round(score))))
    except Exception as exc:
        logger.warning("Fear & Greed primary fetch failed: %s", exc)

    try:
        fallback_url = "https://money.cnn.com/data/fear-and-greed/"
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(fallback_url, headers=headers)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        needle = soup.find("div", id="needleChart")
        if needle:
            score_text = needle.find("li")
            if score_text:
                val = int("".join(c for c in score_text.text if c.isdigit())[:3])
                return FearGreed(value=val, label=_fg_label(val))
    except Exception as exc2:
        logger.warning("Fear & Greed fallback failed: %s", exc2)

    logger.warning("Returning Neutral (50) default for Fear & Greed")
    return _NEUTRAL_FALLBACK


def _yf_fetch_news(ticker: str, max_items: int = 10) -> list[NewsHeadline]:
    try:
        stock = yf.Ticker(ticker.upper())
        raw_news = stock.news or []
    except Exception as exc:
        logger.warning("yfinance news error for %s: %s", ticker, exc)
        return []

    headlines: list[NewsHeadline] = []
    for item in raw_news[:max_items]:
        content = item.get("content", item)

        title = content.get("title", "")
        publisher = None
        provider = content.get("provider")
        if isinstance(provider, dict):
            publisher = provider.get("displayName")

        link = None
        canonical = content.get("canonicalUrl") or content.get("clickThroughUrl")
        if isinstance(canonical, dict):
            link = canonical.get("url")

        published_iso = content.get("pubDate")

        if title:
            headlines.append(
                NewsHeadline(
                    title=title,
                    publisher=publisher,
                    link=link,
                    published=published_iso,
                )
            )
    return headlines


async def fetch_news(ticker: str, max_items: int = 10) -> list[NewsHeadline]:
    return await asyncio.to_thread(_yf_fetch_news, ticker.upper(), max_items)


async def finbert_sentiment(headlines: list[NewsHeadline]) -> NewsSentiment:
    """Run headlines through ProsusAI/finbert via HF Inference API."""
    if not headlines:
        return NewsSentiment()

    titles = [h.title for h in headlines if h.title]
    if not titles:
        return NewsSentiment()

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                FINBERT_URL,
                json={"inputs": titles, "options": {"wait_for_model": True}},
                headers=headers,
            )
            resp.raise_for_status()
            results = resp.json()

        pos, neg, neu = 0, 0, 0
        score_sum = 0.0

        for i, result in enumerate(results):
            if not isinstance(result, list) or not result:
                continue

            top = max(result, key=lambda x: x.get("score", 0))
            label = top.get("label", "neutral").lower()
            conf = top.get("score", 0)

            if label == "positive":
                pos += 1
                score_sum += 0.5 + conf * 0.5
            elif label == "negative":
                neg += 1
                score_sum += 0.5 - conf * 0.5
            else:
                neu += 1
                score_sum += 0.5

            if i < len(headlines):
                headlines[i].sentiment = label
                headlines[i].sentiment_score = round(conf, 3)

        total = pos + neg + neu
        avg = score_sum / total if total > 0 else 0.5

        if pos > neg * 2:
            overall = "Bullish"
        elif neg > pos * 2:
            overall = "Bearish"
        elif pos > neg:
            overall = "Slightly Bullish"
        elif neg > pos:
            overall = "Slightly Bearish"
        elif pos == 0 and neg == 0:
            overall = "Neutral"
        else:
            overall = "Mixed"

        return NewsSentiment(
            positive=pos, negative=neg, neutral=neu,
            avg_score=round(avg, 3), label=overall,
        )
    except Exception as exc:
        logger.warning("FinBERT sentiment analysis failed: %s — falling back to keyword method", exc)
        return _keyword_sentiment_fallback(headlines)


def _keyword_sentiment_fallback(headlines: list[NewsHeadline]) -> NewsSentiment:
    """Fallback to keyword matching if FinBERT API is unavailable."""
    pos, neg, neu = 0, 0, 0
    for h in headlines:
        title = h.title.lower()
        if any(k in title for k in BULL_KEYWORDS):
            pos += 1
            h.sentiment = "positive"
            h.sentiment_score = 0.7
        elif any(k in title for k in BEAR_KEYWORDS):
            neg += 1
            h.sentiment = "negative"
            h.sentiment_score = 0.7
        else:
            neu += 1
            h.sentiment = "neutral"
            h.sentiment_score = 0.5

    total = max(pos + neg + neu, 1)
    avg = (pos * 0.8 + neu * 0.5 + neg * 0.2) / total

    if pos > neg * 2:
        overall = "Bullish"
    elif neg > pos * 2:
        overall = "Bearish"
    elif pos > neg:
        overall = "Slightly Bullish"
    elif neg > pos:
        overall = "Slightly Bearish"
    else:
        overall = "Neutral"

    return NewsSentiment(
        positive=pos, negative=neg, neutral=neu,
        avg_score=round(avg, 3), label=overall,
    )


async def fetch_stock_data(
    ticker: str,
) -> tuple[list[dict[str, Any]], list[NewsHeadline]]:
    history, news = await asyncio.gather(
        fetch_daily_ohlcv(ticker),
        fetch_news(ticker),
    )
    return history, news


async def gpt_analysis(
    ticker: str,
    weights: InferenceWeights,
    denoised: DenoisedTrend,
    zscore: ZScoreAnalysis,
    smas: MovingAverages,
    technicals: TechnicalIndicators,
    bollinger: BollingerBands,
    obv: OBVAnalysis,
    volume: VolumeProfile,
    analyst: AnalystRating,
    fear_greed: FearGreed,
    news_sent: NewsSentiment,
    headlines: list[NewsHeadline],
) -> GPTInsight:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured.")

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    news_text = "\n".join(
        f"- [{h.sentiment or 'unknown'} {h.sentiment_score or 0:.0%}] {h.title} ({h.publisher})"
        for h in headlines
    ) or "No recent headlines available."

    payload = {
        "ticker": ticker.upper(),
        "inference_weights": {
            "technical_score": weights.technical_score,
            "sentiment_score": weights.sentiment_score,
            "analyst_score": weights.analyst_score,
            "volume_score": weights.volume_score,
            "composite_score": weights.composite_score,
            "composite_signal": weights.composite_signal,
            "formula": "35% Technical + 25% Sentiment + 20% Analyst + 20% Volume",
        },
        "denoised_trend": {
            "savitzky_golay_slope_pct_per_day": denoised.slope,
            "slope_direction": denoised.slope_direction,
            "acceleration_pct_per_day2": denoised.acceleration,
            "momentum_exhaustion": denoised.momentum_exhaustion,
            "exhaustion_type": denoised.exhaustion_type,
        },
        "zscore_20d": {
            "value": zscore.zscore,
            "mean": zscore.mean_20d,
            "stddev": zscore.stddev_20d,
            "signal": zscore.signal,
            "reversal_probability_pct": zscore.reversal_probability,
        },
        "moving_averages": {
            "sma_50": smas.sma_50,
            "sma_200": smas.sma_200,
            "signal": smas.signal,
        },
        "technical_indicators": {
            "rsi_14": technicals.rsi_14,
            "rsi_signal": technicals.rsi_signal,
            "macd_histogram": technicals.macd_histogram,
            "macd_trend": technicals.macd_trend,
        },
        "bollinger_bands": {
            "upper": bollinger.upper,
            "lower": bollinger.lower,
            "bandwidth_pct": bollinger.bandwidth,
            "price_position": bollinger.position,
            "squeeze_detected": bollinger.squeeze,
        },
        "on_balance_volume": {
            "obv_trend": obv.obv_trend,
            "price_trend_20d": obv.price_trend,
            "divergence": obv.divergence,
        },
        "volume_profile": {
            "volume_ratio": volume.volume_ratio,
            "volume_spike": volume.spike,
        },
        "analyst_consensus": {
            "recommendation": analyst.recommendation,
            "score": analyst.score,
        },
        "fear_and_greed": {
            "value": fear_greed.value,
            "label": fear_greed.label,
        },
        "finbert_news_sentiment": {
            "positive_headlines": news_sent.positive,
            "negative_headlines": news_sent.negative,
            "neutral_headlines": news_sent.neutral,
            "avg_score": news_sent.avg_score,
            "overall_label": news_sent.label,
        },
        "recent_news_headlines": news_text,
    }

    system_prompt = (
        "You are a JSON-only response engine. Do not include markdown formatting "
        "or prose outside the JSON object.\n\n"
        "Role: Hardline Mathematical Quantitative Analyst. You operate on "
        "statistically significant signals, not opinions. Every claim must be "
        "anchored to a numerical threshold or mathematical relationship.\n\n"
        "You will receive a pre-calculated inference payload with weighted scores:\n"
        "1. INFERENCE WEIGHTS (pre-computed): Technical (35%), Sentiment (25%), "
        "Analyst (20%), Volume (20%). The composite_score is your baseline — "
        "adjust ±10 points max based on qualitative headline analysis.\n"
        "2. DENOISED PRICE VELOCITY: Savitzky-Golay polynomial regression "
        "(window=21, order=3) removes market noise. The slope (% per day) "
        "represents true price velocity. If slope is positive but acceleration "
        "is negative, the trend is mathematically decelerating.\n"
        "3. 20-DAY Z-SCORE: Statistical distance from the rolling mean in σ. "
        "|Z| > 2.0 = 95.4% statistical significance for mean reversion. "
        "|Z| > 1.5 = 86.6% probability. This is your primary reversal signal.\n"
        "4. Classical technicals: RSI-14, MACD, SMA crossovers, Bollinger Bands.\n"
        "5. OBV divergence + volume confirmation.\n"
        "6. FinBERT NLP Sentiment: Per-headline sentiment labels (positive/negative/neutral) "
        "from ProsusAI/finbert transformer model. Aggregate avg_score 0-1 (0=bearish, 1=bullish).\n"
        "7. Analyst consensus + Fear & Greed (contrarian-adjusted).\n\n"
        "DECISION FRAMEWORK:\n"
        "- Z-Score > +2.0 AND denoised slope decelerating → SELL "
        "(mean reversion imminent, statistically significant).\n"
        "- Z-Score < -2.0 AND OBV rising → BUY "
        "(oversold with accumulation, 95% reversion probability).\n"
        "- Denoised slope positive + acceleration positive + Z < 1.5 → BUY "
        "(healthy trend with statistical room to run).\n"
        "- Momentum Exhaustion detected → reversal is mathematically likely, "
        "WATCH or counter-trade.\n"
        "- Bollinger Squeeze + Z near 0 + dry volume → WATCH "
        "(breakout imminent, wait for directional confirmation).\n"
        "- OBV divergence overrides price action: Accumulation = stealth buying, "
        "Distribution = smart money exiting.\n\n"
        "Your reasoning MUST reference: 'Denoised Price Velocity', "
        "'Statistical Significance', and specific Z-Score / slope values.\n\n"
        "Respond with ONLY this JSON:\n"
        "{\n"
        '  "actionable_insight": "<Buy|Hold|Sell|Watch> — explanation citing '
        'denoised velocity and Z-Score thresholds",\n'
        '  "confidence_score": <int 0-100>,\n'
        '  "reasoning": "<2-3 sentences. MUST reference denoised slope, Z-Score, '
        'statistical significance, and weight contributions>"\n'
        "}"
    )

    user_message = (
        "Analyze the following market data and produce your recommendation:\n\n"
        + json.dumps(payload, indent=2)
    )

    completion = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
        max_tokens=600,
    )

    raw = (completion.choices[0].message.content or "{}").strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("GPT-4 returned unparseable JSON: %s", raw)
        parsed = {
            "actionable_insight": raw[:500],
            "confidence_score": 0,
            "reasoning": "Failed to parse structured response from GPT-4.",
        }

    return GPTInsight(
        actionable_insight=parsed.get("actionable_insight", "N/A"),
        confidence_score=max(0, min(100, int(parsed.get("confidence_score", 0)))),
        reasoning=parsed.get("reasoning", ""),
    )


@app.get("/")
async def root():
    return {
        "service": "MSA — Market Sentiment Analyzer",
        "version": "3.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "ok", "cache_size": len(_analysis_cache)}


@app.get("/analyze", response_model=AnalysisResponse)
async def analyze(
    ticker: str = Query(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol, e.g. AAPL",
    ),
):
    ticker = ticker.upper().strip()

    if ticker in _analysis_cache:
        logger.info("Cache HIT for %s", ticker)
        cached_data = _analysis_cache[ticker].copy()
        cached_data["cached"] = True
        return AnalysisResponse(**cached_data)

    logger.info("Cache MISS — starting analysis for %s", ticker)
    t0 = time.perf_counter()

    (daily_rows, news), fg, analyst, pt = await asyncio.gather(
        fetch_stock_data(ticker),
        fetch_fear_greed(),
        fetch_analyst_ratings(ticker),
        fetch_price_target(ticker),
    )

    news_sent = await finbert_sentiment(news)

    smas = calculate_smas(daily_rows)
    technicals = calculate_technicals(daily_rows)
    bb = calculate_bollinger(daily_rows)
    obv = calculate_obv(daily_rows)
    vol_profile = calculate_volume_profile(daily_rows)
    denoised = calculate_denoised_trend(daily_rows, technicals.rsi_14)
    zsc = calculate_zscore(daily_rows)
    weights = calculate_inference_weights(
        technicals, denoised, zsc, fg, news_sent, analyst, obv, vol_profile,
    )

    gpt_result: GPTInsight | None = None
    try:
        gpt_result = await gpt_analysis(
            ticker, weights, denoised, zsc,
            smas, technicals, bb, obv, vol_profile, analyst, fg, news_sent, news,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("GPT-4 analysis failed: %s", exc)

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info("Analysis for %s completed in %.2fs", ticker, elapsed)

    # last 30 days of closing prices, oldest first for charting
    recent = daily_rows[:30]
    recent.reverse()
    history = [PricePoint(date=r["date"], close=round(r["close"], 2)) for r in recent]

    result = AnalysisResponse(
        ticker=ticker,
        timestamp=datetime.now(timezone.utc).isoformat(),
        cached=False,
        moving_averages=smas,
        technicals=technicals,
        bollinger_bands=bb,
        obv_analysis=obv,
        volume_profile=vol_profile,
        denoised_trend=denoised,
        zscore_analysis=zsc,
        inference_weights=weights,
        analyst_ratings=analyst,
        price_target=pt,
        fear_greed=fg,
        news=news,
        news_sentiment=news_sent,
        gpt_analysis=gpt_result,
        price_history=history,
    )

    _analysis_cache[ticker] = result.model_dump()

    return result


@app.get("/sma", response_model=MovingAverages)
async def sma_only(
    ticker: str = Query(..., min_length=1, max_length=10),
):
    daily = await fetch_daily_ohlcv(ticker.upper().strip())
    return calculate_smas(daily)


@app.get("/fear-greed", response_model=FearGreed)
async def fear_greed_only():
    return await fetch_fear_greed()


@app.get("/news", response_model=list[NewsHeadline])
async def news_only(
    ticker: str = Query(..., min_length=1, max_length=10),
):
    return await fetch_news(ticker.upper().strip())


@app.delete("/cache/{ticker}")
async def clear_cache(ticker: str):
    key = ticker.upper().strip()
    removed = _analysis_cache.pop(key, None) is not None
    return {"ticker": key, "removed": removed, "cache_size": len(_analysis_cache)}


@app.delete("/cache")
async def clear_all_cache():
    count = len(_analysis_cache)
    _analysis_cache.clear()
    return {"cleared": count}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
