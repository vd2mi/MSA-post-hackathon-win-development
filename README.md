# MSA — Market Sentiment Analyzer

AI-powered stock analysis platform that fuses real technical indicators, Bollinger Bands, On-Balance Volume, volume profiling, Wall Street analyst consensus, market sentiment, and live news into a single actionable signal via GPT-4o — built for speed, built for traders.

## What It Does

- **Technical Indicators** — RSI (14-day), MACD (12, 26, 9), 50-day & 200-day SMAs with Golden Cross / Death Cross detection. Combined into a 0–100 gauge score.
- **Bollinger Bands** — 20-period SMA ± 2 standard deviations. Bandwidth %, price position relative to bands, and squeeze detection for breakout signals.
- **On-Balance Volume (OBV)** — Cumulative volume trend analysis with divergence detection. Flags accumulation (stealth buying) vs distribution (smart money exiting).
- **Volume Profile** — 20-day average volume, latest volume ratio, and spike detection for confirmation signals.
- **Analyst Ratings** — Real Wall Street analyst breakdown (Strong Buy / Buy / Hold / Sell / Strong Sell) with weighted consensus score and semicircle gauge.
- **1-Year Price Target** — Mean, high, and low analyst targets with current price positioning and upside/downside %.
- **Live Price & Daily Change** — Current market price with real-time daily $ and % movement (green ▲ / red ▼).
- **Market Sentiment** — CNN Fear & Greed Index scraped live (graceful Neutral fallback).
- **News Intelligence** — Latest Yahoo Finance headlines per ticker.
- **GPT-4o Synthesis** — All data signals fed to a Senior Quant Analyst persona with rules-based analysis (BB + OBV divergence, squeeze detection, volume confirmation). Returns Buy / Hold / Sell / Watch with confidence score (0–100) and reasoning.
- **Comparison Mode** — Side-by-side analysis of two tickers with tug-of-war bars, radar chart, and goal-based recommendations (best to buy, strongest technicals, smart money flow, upside potential).
- **5-Minute TTL Cache** — Repeat requests served instantly, saving API costs.
- **Live Ticker Feed** — WebSocket-powered real-time price belt on the landing page.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, Uvicorn, Pydantic v2 |
| Data Sources | yfinance (prices, news, analysts, targets), CNN (Fear & Greed), httpx, BeautifulSoup4 |
| AI | OpenAI GPT-4o (strict JSON mode, temperature 0.4) |
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
| `GET` | `/analyze?ticker=AAPL` | Full pipeline — SMAs, RSI, MACD, Bollinger Bands, OBV, Volume Profile, Analysts, Price Target, Fear & Greed, News, GPT-4o (cached 5 min) |
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
        ├─── yfinance: 1yr OHLCV → SMA 50/200, RSI-14, MACD, Bollinger Bands, OBV, Volume Profile
        ├─── yfinance: analyst ratings (Strong Buy → Strong Sell)
        ├─── yfinance: 1yr price targets + current price + daily change
        ├─── CNN: Fear & Greed Index
        └─── yfinance: news headlines
                │
                ▼
        All signals → GPT-4o (strict JSON, rules-based) → Buy/Hold/Sell/Watch + Confidence
```

GPT-4o analysis rules include:
- Price at Lower Band + OBV Rising → Bullish Mean Reversion
- Price at Upper Band + OBV Falling → Bearish Exhaustion
- Bollinger Squeeze + dry volume → Breakout imminent
- OBV rising while price flat/falling → Accumulation (smart money buying)
- OBV falling while price rising → Distribution (fake rally)
- Volume spike + price move = confirmation; Volume spike + no move = absorption

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key with GPT-4 access |

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
```

Run the server:
```bash
python app.py
```

Server starts on `http://localhost:7860`. Interactive docs at `/docs`.

## Frontend

### Landing Page (`index.html`)
- Animated candlestick chart background in the hero section
- WebSocket live ticker belt with seamless looping
- API health stats (status, response time)
- Terminal-style glass-panel feature cards
- Crosshair cursor theme

### Analysis Terminal (`dashboard.html`)
- Animated loading screen with chart line, floating market words, and system logs
- Live price + daily change arrow (green ▲ / red ▼)
- AI recommendation badge (Buy/Sell/Watch/Hold with confidence %)
- SMA crossover signal (Golden Cross / Death Cross)
- Technicals semicircle gauge (RSI + MACD combined score)
- Analyst rating semicircle gauge with bar breakdown
- 1-year price target with range visualization
- Fear & Greed animated gauge
- Bollinger Bands panel (upper/lower/bandwidth/squeeze detection)
- On-Balance Volume panel (trend, divergence, accumulation/distribution)
- Volume Profile panel (avg volume, ratio, spike detection)
- 30-day price chart with dynamic red/green segments, SMA lines, and BB overlay
- AI analysis card with confidence score and reasoning
- Tooltips on every panel explaining metrics and typical ranges
- "Compare it" CTA linking to comparison mode

### Comparison Mode (`compare.html`)
- Animated loading screen matching the dashboard
- Side-by-side ticker header cards with live price and AI badges
- Tug-of-war bars for every metric (AI confidence, technicals, analyst score, RSI, MACD, SMA signal, BB position, OBV signal, volume ratio, upside %, Fear & Greed)
- Radar chart overlaying both tickers across 5 dimensions
- Goal-based verdict system (best to buy, strongest technicals, smart money flow, most upside)
- Winner/loser glow effects on header cards

All pages connect to the Hugging Face-hosted API at `https://vd2mi-msa.hf.space`.
