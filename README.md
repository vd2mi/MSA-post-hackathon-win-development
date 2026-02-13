# MSA — Market Sentiment Analyzer

AI-powered stock analysis platform that fuses real technical indicators, Wall Street analyst consensus, market sentiment, and live news into a single actionable signal via GPT-4o — built for speed, built for traders.

## What It Does

- **Technical Indicators** — RSI (14-day), MACD (12, 26, 9), 50-day & 200-day SMAs with Golden Cross / Death Cross detection. Combined into a 0–100 gauge score.
- **Analyst Ratings** — Real Wall Street analyst breakdown (Strong Buy / Buy / Hold / Sell / Strong Sell) with weighted consensus score and semicircle gauge.
- **1-Year Price Target** — Mean, high, and low analyst targets with current price positioning and upside/downside %.
- **Live Price & Daily Change** — Current market price with real-time daily $ and % movement (green ▲ / red ▼).
- **Market Sentiment** — CNN Fear & Greed Index scraped live (graceful Neutral fallback).
- **News Intelligence** — Latest Yahoo Finance headlines per ticker.
- **GPT-4o Synthesis** — All 6 data signals fed to a Senior Quant Analyst persona. Returns Buy / Hold / Sell / Watch with confidence score (0–100) and reasoning.
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
| Frontend | Tailwind CSS, Chart.js, Three.js, GSAP |
| Live Data | WebSocket (wss://fisi-production.up.railway.app) |
| Deployment | Docker on Hugging Face Spaces |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check + cache size |
| `GET` | `/analyze?ticker=AAPL` | **Full pipeline** — SMAs + RSI + MACD + Analysts + Price Target + Fear & Greed + News + GPT-4o (cached 5 min) |
| `GET` | `/sma?ticker=AAPL` | SMA data only |
| `GET` | `/fear-greed` | CNN Fear & Greed Index |
| `GET` | `/news?ticker=AAPL` | Yahoo Finance news headlines |
| `DELETE` | `/cache/{ticker}` | Evict one ticker from cache |
| `DELETE` | `/cache` | Purge entire cache |

## Project Structure

```
MSA/
├── app.py              # FastAPI backend (data pipeline, indicators, GPT integration)
├── index.html          # Landing page (Three.js animation, live ticker belt, stats)
├── dashboard.html      # Analysis terminal (gauges, charts, GPT results, news)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Hugging Face Spaces deployment
├── .env                # Local env vars (not committed)
└── README.md
```

## Data Pipeline

All data is fetched concurrently via `asyncio.gather` for minimum latency:

```
User enters ticker
        │
        ├─── yfinance: 1yr OHLCV → SMA 50/200, RSI-14, MACD
        ├─── yfinance: analyst ratings (Strong Buy → Strong Sell)
        ├─── yfinance: 1yr price targets + current price + daily change
        ├─── CNN: Fear & Greed Index
        └─── yfinance: news headlines
                │
                ▼
        All signals → GPT-4o (strict JSON) → Buy/Hold/Sell/Watch + Confidence
```

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

- **index.html** — Landing page with Three.js market wave, WebSocket live ticker belt, API health stats, and terminal-style feature cards
- **dashboard.html** — Full analysis terminal with:
  - Live price + daily change arrow
  - AI recommendation badge (Buy/Sell/Watch/Hold)
  - SMA crossover signal
  - Technicals semicircle gauge (RSI + MACD)
  - Analyst rating semicircle gauge with bar breakdown
  - 1-year price target with range visualization
  - Fear & Greed animated gauge
  - Price action chart
  - GPT-4o analysis card with confidence score
  - Latest news headlines

Both pages connect to the Hugging Face-hosted API at `https://vd2mi-msa.hf.space`.
