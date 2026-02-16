# MSA — Market Sentiment Analyzer

AI-powered stock analysis platform that fuses Savitzky-Golay denoised trend analysis, statistical Z-Score mean reversion, Bollinger Bands, On-Balance Volume, volume profiling, Wall Street analyst consensus, market sentiment, and live news into a single weighted inference signal via GPT-4o — built for quants, built for traders.

## What It Does

### Signal Processing (v3.0)
- **Savitzky-Golay Denoised Trend** — Polynomial regression (window=21, order=3) implemented via numpy Vandermonde pseudo-inverse. Extracts true price velocity (% per day) and acceleration. Detects momentum exhaustion when denoised slope diverges from RSI.
- **Statistical Z-Score** — 20-day rolling Z-Score: (Price - Mean) / StdDev. Reversal probability calculated via error function. |Z| > 2.0 = 95.4% statistical significance for mean reversion.
- **Inference Weighting Engine** — Pre-calculated 4-pillar composite score fed directly to GPT as a mathematical baseline:
  - Technical (35%): SG slope + Z-Score + RSI/MACD
  - Sentiment (25%): CNN Fear & Greed (contrarian-adjusted, 60%) + FinBERT NLP aggregate score (40%)
  - Analyst (20%): Wall Street consensus weighted score
  - Volume (20%): OBV divergence + volume spike confirmation

### Technical Indicators
- **RSI + MACD** — RSI (14-day), MACD (12, 26, 9), 50-day & 200-day SMAs with Golden Cross / Death Cross detection. Combined into a 0–100 gauge score.
- **Bollinger Bands** — 20-period SMA ± 2 standard deviations. Bandwidth %, price position relative to bands, and squeeze detection for breakout signals.
- **On-Balance Volume (OBV)** — Cumulative volume trend analysis with divergence detection. Flags accumulation (stealth buying) vs distribution (smart money exiting).
- **Volume Profile** — 20-day average volume, latest volume ratio, and spike detection for confirmation signals.

### Fundamentals & Sentiment
- **Analyst Ratings** — Real Wall Street analyst breakdown (Strong Buy / Buy / Hold / Sell / Strong Sell) with weighted consensus score and semicircle gauge.
- **1-Year Price Target** — Mean, high, and low analyst targets with current price positioning and upside/downside %. Multiple fallback keys for maximum compatibility across tickers.
- **Live Price & Daily Change** — Current market price with real-time daily $ and % movement (green ▲ / red ▼).
- **Market Sentiment** — CNN Fear & Greed Index scraped live (graceful Neutral fallback).
- **News Intelligence** — Latest Yahoo Finance headlines per ticker, each analyzed by **FinBERT** (ProsusAI/finbert) transformer model via Hugging Face Inference API. Per-headline positive/negative/neutral labels with confidence scores. Aggregate sentiment score (0-1) feeds directly into inference weights. Graceful fallback to keyword matching if FinBERT API is unavailable.

### AI & Platform
- **GPT-4o Hardline Quant** — All signals + pre-calculated inference weights fed to a "Hardline Mathematical Quantitative Analyst" persona. Must reference "Denoised Price Velocity", "Statistical Significance", and specific Z-Score thresholds in every response. Temperature 0.3 for deterministic output.
- **Comparison Mode** — Side-by-side analysis of two tickers with tug-of-war bars, radar chart, and goal-based recommendations.
- **5-Minute TTL Cache** — Repeat requests served instantly, saving API costs.
- **Live Ticker Feed** — WebSocket-powered real-time price belt on the landing page.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, Uvicorn, Pydantic v2 |
| Data Sources | yfinance (prices, news, analysts, targets), CNN (Fear & Greed), httpx, BeautifulSoup4 |
| Signal Processing | NumPy (Savitzky-Golay filter, Z-Score, polyfit) |
| NLP Sentiment | ProsusAI/finbert via Hugging Face Inference API (serverless, zero-dep) |
| AI | OpenAI GPT-4o (strict JSON mode, temperature 0.3, hardline quant persona) |
| Caching | cachetools TTLCache (in-memory, 5 min, 256 entries) |
| Async | asyncio.gather + asyncio.to_thread for non-blocking I/O |
| Frontend | Tailwind CSS, Chart.js, Canvas API, SVG gauges |
| Live Data | WebSocket (wss://fisi-production.up.railway.app) |
| Deployment | Docker on Hugging Face Spaces, Vercel (frontend) |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check + cache size |
| `GET` | `/analyze?ticker=AAPL` | Full pipeline — SG denoising, Z-Score, FinBERT NLP sentiment, inference weights, SMAs, RSI, MACD, BB, OBV, Volume, Analysts, Price Target, Fear & Greed, News, GPT-4o (cached 5 min) |
| `GET` | `/sma?ticker=AAPL` | SMA data only |
| `GET` | `/fear-greed` | CNN Fear & Greed Index |
| `GET` | `/news?ticker=AAPL` | Yahoo Finance news headlines |
| `DELETE` | `/cache/{ticker}` | Evict one ticker from cache |
| `DELETE` | `/cache` | Purge entire cache |

## Project Structure

```
MSA/
├── app.py              # FastAPI backend (data pipeline, indicators, GPT integration)
├── index.html          # Landing page (animated candlestick bg, live ticker belt, stats)
├── dashboard.html      # Analysis terminal (gauges, charts, GPT results, tooltips, loader)
├── compare.html        # Side-by-side ticker comparison (radar chart, tug-of-war, verdicts)
├── base.html           # Shared base template
├── script.js           # Shared frontend utilities
├── style.css           # Shared styles
├── requirements.txt    # Python dependencies
├── Dockerfile          # Hugging Face Spaces deployment
├── vercel.json         # Vercel frontend deployment config
├── .gitignore
├── .env                # Local env vars (not committed)
└── README.md
```

## Data Pipeline

All data is fetched concurrently via `asyncio.gather` for minimum latency:

```
User enters ticker
        │
        ├─── yfinance: 1yr OHLCV → SMA 50/200, RSI-14, MACD, BB, OBV, Volume Profile
        ├─── yfinance: analyst ratings (Strong Buy → Strong Sell)
        ├─── yfinance: 1yr price targets + current price + daily change
        ├─── CNN: Fear & Greed Index
        └─── yfinance: news headlines
                │
                ▼
        FinBERT (ProsusAI/finbert) → per-headline sentiment labels + aggregate score
        Savitzky-Golay denoising → Price Velocity + Acceleration
        20-day Z-Score → Mean Reversion Probability (erf)
        4-Pillar Inference Weights (FinBERT-powered sentiment) → Composite Score (0-100)
                │
                ▼
        Weighted payload → GPT-4o (strict JSON, hardline quant) → Buy/Hold/Sell/Watch + Confidence
```

GPT-4o decision framework:
- Z-Score > +2.0 AND denoised slope decelerating → SELL (95.4% mean reversion, statistically significant)
- Z-Score < -2.0 AND OBV rising → BUY (oversold with accumulation, 95% reversion probability)
- Denoised slope positive + acceleration positive + Z < 1.5 → BUY (healthy trend with room)
- Momentum Exhaustion detected → reversal mathematically likely, WATCH or counter-trade
- Bollinger Squeeze + Z near 0 + dry volume → WATCH (breakout imminent)
- OBV divergence overrides price action: Accumulation vs Distribution
- FinBERT sentiment aggregates factored into composite score (40% weight within sentiment pillar)

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key with GPT-4 access |
| `HF_API_TOKEN` | No | Hugging Face token (higher FinBERT rate limits; works without it on free tier) |

## Deploying to Hugging Face Spaces

1. Create a new Space → select **Docker** as the SDK.
2. Push this repo to the Space.
3. Add `OPENAI_API_KEY` as a Secret in the Space settings.
4. The Space builds and exposes the API on port **7860**.

Live API: `https://vd2mi-msa.hf.space`

## Local Development

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
OPENAI_API_KEY=your_key_here
HF_API_TOKEN=your_hf_token_here  # Optional: improves FinBERT rate limits
```

Run the server:
```bash
python app.py
```

Server starts on `http://localhost:7860`. Interactive docs at `/docs`.

**Note:** FinBERT works on Hugging Face's free tier without `HF_API_TOKEN`, but adding a token provides higher rate limits for production use.

## Frontend

### Landing Page (`index.html`)
- Animated candlestick chart background in the hero section
- WebSocket live ticker belt with seamless looping
- API health stats (status, response time)
- Terminal-style glass-panel feature cards
- Crosshair cursor theme

### Analysis Terminal (`dashboard.html`)
- Animated loading screen with chart line, floating market words, and system logs
- Auto-analysis on page load when `?ticker=SYMBOL` is in URL
- Live price + daily change arrow (green ▲ / red ▼)
- AI recommendation badge (Buy/Sell/Watch/Hold with confidence %)
- SMA crossover signal (Golden Cross / Death Cross)
- Technicals semicircle gauge (RSI + MACD combined score)
- Analyst rating semicircle gauge with bar breakdown
- 1-year price target with range visualization (graceful handling when targets unavailable)
- Fear & Greed animated gauge
- Denoised Trend panel (SG slope velocity, acceleration, momentum exhaustion badge)
- Z-Score panel (value, reversal probability bar, 20d mean/sigma)
- Inference Weights panel (4-pillar horizontal bars + composite score)
- Bollinger Bands panel (upper/lower/bandwidth/squeeze detection)
- On-Balance Volume panel (trend, divergence, accumulation/distribution)
- Volume Profile panel (avg volume, ratio, spike detection)
- FinBERT News Sentiment summary header (+X / -Y / ~Z breakdown)
- News cards with per-headline FinBERT sentiment dots (green/red/gray) and confidence scores
- 30-day price chart with dynamic red/green segments, SMA lines, BB overlay, and SG denoised trendline
- AI analysis card with confidence score and reasoning
- Tooltips on every panel explaining metrics and typical ranges
- "Compare it" CTA linking to comparison mode with pre-filled ticker

### Comparison Mode (`compare.html`)
- Animated loading screen matching the dashboard
- Side-by-side ticker header cards with live price and AI badges
- Tug-of-war bars for every metric (AI confidence, technicals, analyst score, RSI, MACD, SMA signal, BB position, OBV signal, volume ratio, upside %, Fear & Greed, Composite Score, SG Velocity, Z-Score, Momentum Exhaustion, FinBERT Sentiment)
- Radar chart overlaying both tickers across 7 dimensions (AI Confidence, Technicals, Analyst, Composite Score, Sentiment, Upside %, Volume)
- Goal-based verdict system (best to buy, strongest technicals, smart money flow, most upside, stronger quant signal, better news sentiment)
- Winner/loser glow effects on header cards
- Pre-filled ticker inputs when navigating from dashboard

All pages connect to the Hugging Face-hosted API at `https://vd2mi-msa.hf.space`.
