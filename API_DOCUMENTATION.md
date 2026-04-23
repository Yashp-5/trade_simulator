# Autonomous Trading Simulator API Documentation

This documentation describes the REST API endpoints for the Autonomous Trading Simulator built with FastAPI.

---

## Base URL

    http://localhost:8000/

---

## Endpoints

### 1. Root
- **GET /**
- **Description:** Health check for the API.
- **Response:**
  ```json
  { "status": "ok", "message": "Trading Simulator API" }
  ```

---

### 2. List Available Tickers
- **GET /tickers**
- **Description:** Returns a list of available preset tickers.
- **Response:**
  ```json
  { "tickers": ["SPY", "NIFTY50", ...] }
  ```

---

### 3. Single Strategy Backtest
- **POST /backtest/single**
- **Description:** Runs a backtest for a single strategy on a specified ticker.
- **Request Body:**
  ```json
  {
    "ticker": "SPY",
    "strategy_type": "ema_rsi",
    "ema_fast": 12,
    "ema_slow": 26,
    "rsi_period": 14,
    "stop_loss_atr": 2.0,
    "take_profit_atr": 4.0,
    "position_size": 0.1
  }
  ```
- **Response:**
  ```json
  {
    "strategy_id": "...",
    "fitness": 0.0,
    "train_metrics": { ... },
    "test_metrics": { ... },
    "equity_curve": [ ... ],
    "num_trades": 10,
    "trades": [ { ... }, ... ]
  }
  ```

---

### 4. Start Evolutionary Optimization
- **POST /evolution/start**
- **Description:** Starts a genetic algorithm to evolve strategies for a ticker.
- **Request Body:**
  ```json
  {
    "ticker": "SPY",
    "population_size": 20,
    "generations": 10,
    "mutation_rate": 0.3,
    "elite_fraction": 0.2,
    "train_split": 0.7
  }
  ```
- **Response:**
  ```json
  { "message": "Evolution started", "config": { ... } }
  ```

---

### 5. Evolution Status
- **GET /evolution/status**
- **Description:** Returns the current status and leaderboard of the evolutionary process.
- **Response:**
  ```json
  {
    "running": false,
    "generation": 0,
    "message": "idle",
    "leaderboard": [ ... ]
  }
  ```

---

### 6. List Strategies
- **GET /strategies**
- **Query Parameters:**
  - `ticker` (optional): Filter by ticker symbol
  - `limit` (optional, default=50): Max number of strategies to return
- **Description:** Returns stored strategies and their metrics.
- **Response:**
  ```json
  { "strategies": [ { ... }, ... ] }
  ```

---

## Error Handling
- Errors are returned as JSON with an HTTP status code and a `detail` field.
- Example:
  ```json
  { "detail": "No data returned for SPY. Check ticker or date range." }
  ```

---

## Notes
- All endpoints return JSON.
- For POST endpoints, set `Content-Type: application/json`.
- The API is CORS-enabled for all origins.

---

## Example Usage

### Backtest a Strategy (curl)
```sh
curl -X POST "http://localhost:8000/backtest/single" \
     -H "Content-Type: application/json" \
     -d '{
           "ticker": "SPY",
           "strategy_type": "ema_rsi",
           "ema_fast": 12,
           "ema_slow": 26,
           "rsi_period": 14,
           "stop_loss_atr": 2.0,
           "take_profit_atr": 4.0,
           "position_size": 0.1
         }'
```

### Start Evolution (curl)
```sh
curl -X POST "http://localhost:8000/evolution/start" \
     -H "Content-Type: application/json" \
     -d '{
           "ticker": "SPY",
           "population_size": 20,
           "generations": 10,
           "mutation_rate": 0.3,
           "elite_fraction": 0.2,
           "train_split": 0.7
         }'
```

---

For further details, see the source code or contact the project maintainer.
