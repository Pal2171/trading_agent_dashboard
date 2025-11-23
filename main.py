from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, List, Optional
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


# Carica variabili d'ambiente da .env (se presente)
load_dotenv()


DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL non impostata. Imposta la variabile d'ambiente, "
        "ad esempio: postgresql://user:password@localhost:5432/trading_db",
    )


@contextmanager
def get_connection():
    """Context manager che restituisce una connessione PostgreSQL.

    Usa il DSN in DATABASE_URL.
    """

    conn = psycopg2.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


def to_local_time(dt: Optional[datetime]) -> Optional[datetime]:
    """Converte un datetime (assunto UTC se naive) nel fuso orario locale (Europe/Rome)."""
    if dt is None:
        return None
    
    # Se il datetime è naive (senza timezone), assumiamo sia UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    
    # Converti a Europe/Rome
    return dt.astimezone(ZoneInfo("Europe/Rome"))


# =====================
# Modelli di risposta API
# =====================


class BalancePoint(BaseModel):
    timestamp: datetime
    balance_usd: float


class OpenPosition(BaseModel):
    id: int
    snapshot_id: int
    symbol: str
    side: str
    size: float
    entry_price: Optional[float]
    mark_price: Optional[float]
    pnl_usd: Optional[float]
    leverage: Optional[str]
    snapshot_created_at: datetime


class BotOperation(BaseModel):
    id: int
    created_at: datetime
    operation: str
    symbol: Optional[str]
    direction: Optional[str]
    target_portion_of_balance: Optional[float]
    leverage: Optional[float]
    raw_payload: Any
    system_prompt: Optional[str]


class PerformanceMetrics(BaseModel):
    total_operations: int
    buy_operations: int
    sell_operations: int
    hold_operations: int
    current_balance: Optional[float]
    initial_balance: Optional[float]
    total_return_pct: Optional[float]
    total_return_usd: Optional[float]
    peak_balance: Optional[float]
    max_drawdown_pct: Optional[float]


class CurrentIndicators(BaseModel):
    ticker: Optional[str]
    timestamp: Optional[datetime]
    price: Optional[float]
    ema9: Optional[float]
    ema20: Optional[float]
    ema21: Optional[float]
    supertrend: Optional[str]
    adx: Optional[float]
    macd: Optional[float]
    rsi_7: Optional[float]
    rsi_14: Optional[float]
    candlestick_patterns: Any


class RiskMetrics(BaseModel):
    total_exposure_usd: float
    total_positions: int
    long_positions: int
    short_positions: int
    avg_leverage: Optional[float]
    largest_position_pct: Optional[float]


# =====================
# App FastAPI + Template Jinja2
# =====================


app = FastAPI(
    title="Trading Agent Dashboard API",
    description=(
        "API per leggere i dati del trading agent dal database Postgres: "
        "saldo nel tempo, posizioni aperte, operazioni del bot con full prompt."
    ),
    version="0.3.2",
)

templates = Jinja2Templates(directory="templates")


# =====================
# Endpoint API JSON
# =====================


@app.get("/balance", response_model=List[BalancePoint])
def get_balance() -> List[BalancePoint]:
    """Restituisce TUTTA la storia del saldo (balance_usd) ordinata nel tempo.

    I dati sono presi dalla tabella `account_snapshots`.
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT created_at, balance_usd
                FROM account_snapshots
                ORDER BY created_at ASC;
                """
            )
            rows = cur.fetchall()

    return [
        BalancePoint(timestamp=to_local_time(row[0]), balance_usd=float(row[1]))
        for row in rows
    ]


@app.get("/open-positions", response_model=List[OpenPosition])
def get_open_positions() -> List[OpenPosition]:
    """Restituisce le posizioni aperte dell'ULTIMO snapshot disponibile.

    - Prende l'ultimo record da `account_snapshots`.
    - Recupera le posizioni corrispondenti da `open_positions`.
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Ultimo snapshot
            cur.execute(
                """
                SELECT id, created_at
                FROM account_snapshots
                ORDER BY created_at DESC
                LIMIT 1;
                """
            )
            row = cur.fetchone()
            if not row:
                return []
            snapshot_id = row[0]
            snapshot_created_at = to_local_time(row[1])

            # Posizioni aperte per quello snapshot
            cur.execute(
                """
                SELECT
                    id,
                    snapshot_id,
                    symbol,
                    side,
                    size,
                    entry_price,
                    mark_price,
                    pnl_usd,
                    leverage
                FROM open_positions
                WHERE snapshot_id = %s
                ORDER BY symbol ASC, id ASC;
                """,
                (snapshot_id,),
            )
            rows = cur.fetchall()

    return [
        OpenPosition(
            id=row[0],
            snapshot_id=row[1],
            symbol=row[2],
            side=row[3],
            size=float(row[4]),
            entry_price=float(row[5]) if row[5] is not None else None,
            mark_price=float(row[6]) if row[6] is not None else None,
            pnl_usd=float(row[7]) if row[7] is not None else None,
            leverage=row[8],
            snapshot_created_at=snapshot_created_at,
        )
        for row in rows
    ]


@app.get("/bot-operations", response_model=List[BotOperation])
def get_bot_operations(
    limit: int = Query(
        50,
        ge=1,
        le=500,
        description="Numero massimo di operazioni da restituire (default 50)",
    ),
) -> List[BotOperation]:
    """Restituisce le ULTIME `limit` operazioni del bot con il full system prompt.

    - I dati provengono da `bot_operations` uniti a `ai_contexts`.
    - Ordinati da più recente a meno recente.
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    bo.id,
                    bo.created_at,
                    bo.operation,
                    bo.symbol,
                    bo.direction,
                    bo.target_portion_of_balance,
                    bo.leverage,
                    bo.raw_payload,
                    ac.system_prompt
                FROM bot_operations AS bo
                LEFT JOIN ai_contexts AS ac ON bo.context_id = ac.id
                ORDER BY bo.created_at DESC
                LIMIT %s;
                """,
                (limit,),
            )
            rows = cur.fetchall()

    operations: List[BotOperation] = []
    for row in rows:
        operations.append(
            BotOperation(
                id=row[0],
                created_at=to_local_time(row[1]),
                operation=row[2],
                symbol=row[3],
                direction=row[4],
                target_portion_of_balance=float(row[5]) if row[5] is not None else None,
                leverage=float(row[6]) if row[6] is not None else None,
                raw_payload=row[7],
                system_prompt=row[8],
            )
        )

    return operations


@app.get("/history", response_model=List[BotOperation])
def get_history(
    limit: int = Query(
        100,
        ge=1,
        le=1000,
        description="Numero massimo di operazioni da restituire",
    ),
) -> List[BotOperation]:
    """Restituisce lo storico operazioni filtrando i 'HOLD'."""

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    bo.id,
                    bo.created_at,
                    bo.operation,
                    bo.symbol,
                    bo.direction,
                    bo.target_portion_of_balance,
                    bo.leverage,
                    bo.raw_payload,
                    ac.system_prompt
                FROM bot_operations AS bo
                LEFT JOIN ai_contexts AS ac ON bo.context_id = ac.id
                WHERE bo.operation NOT ILIKE 'hold'
                ORDER BY bo.created_at DESC
                LIMIT %s;
                """,
                (limit,),
            )
            rows = cur.fetchall()

    operations: List[BotOperation] = []
    for row in rows:
        operations.append(
            BotOperation(
                id=row[0],
                created_at=to_local_time(row[1]),
                operation=row[2],
                symbol=row[3],
                direction=row[4],
                target_portion_of_balance=float(row[5]) if row[5] is not None else None,
                leverage=float(row[6]) if row[6] is not None else None,
                raw_payload=row[7],
                system_prompt=row[8],
            )
        )

    return operations


@app.get("/performance-metrics", response_model=PerformanceMetrics)
def get_performance_metrics() -> PerformanceMetrics:
    """Restituisce metriche di performance del trading bot."""

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Conta operazioni per tipo (escluso HOLD dal totale)
            cur.execute(
                """
                SELECT 
                    COUNT(CASE WHEN operation NOT ILIKE 'hold' THEN 1 END) as total,
                    COUNT(CASE WHEN operation ILIKE 'buy' OR operation ILIKE 'open' THEN 1 END) as buys,
                    COUNT(CASE WHEN operation ILIKE 'sell' OR operation ILIKE 'close' THEN 1 END) as sells,
                    COUNT(CASE WHEN operation ILIKE 'hold' THEN 1 END) as holds
                FROM bot_operations;
                """
            )
            ops = cur.fetchone()

            # Balance attuale e iniziale
            cur.execute(
                """
                SELECT balance_usd 
                FROM account_snapshots 
                ORDER BY created_at ASC 
                LIMIT 1;
                """
            )
            initial_row = cur.fetchone()
            initial_balance = float(initial_row[0]) if initial_row else None

            cur.execute(
                """
                SELECT balance_usd 
                FROM account_snapshots 
                ORDER BY created_at DESC 
                LIMIT 1;
                """
            )
            current_row = cur.fetchone()
            current_balance = float(current_row[0]) if current_row else None

            # Peak balance per calcolare drawdown
            cur.execute(
                """
                SELECT MAX(balance_usd) 
                FROM account_snapshots;
                """
            )
            peak_row = cur.fetchone()
            peak_balance = float(peak_row[0]) if peak_row and peak_row[0] else None

    # Calcola metriche derivate
    total_return_pct = None
    total_return_usd = None
    max_drawdown_pct = None

    if initial_balance and current_balance:
        total_return_usd = current_balance - initial_balance
        total_return_pct = (total_return_usd / initial_balance) * 100

    if peak_balance and current_balance:
        max_drawdown_pct = ((peak_balance - current_balance) / peak_balance) * 100

    return PerformanceMetrics(
        total_operations=ops[0] if ops else 0,
        buy_operations=ops[1] if ops else 0,
        sell_operations=ops[2] if ops else 0,
        hold_operations=ops[3] if ops else 0,
        current_balance=current_balance,
        initial_balance=initial_balance,
        total_return_pct=total_return_pct,
        total_return_usd=total_return_usd,
        peak_balance=peak_balance,
        max_drawdown_pct=max_drawdown_pct,
    )


@app.get("/current-indicators", response_model=CurrentIndicators)
def get_current_indicators(
    ticker: Optional[str] = Query(None, description="Filtra per ticker specifico")
) -> CurrentIndicators:
    """Restituisce gli ultimi indicatori tecnici disponibili, opzionalmente filtrati per ticker."""

    with get_connection() as conn:
        with conn.cursor() as cur:
            if ticker:
                cur.execute(
                    """
                    SELECT 
                        ticker,
                        ts,
                        price,
                        ema9,
                        ema20,
                        ema21,
                        supertrend,
                        adx,
                        macd,
                        rsi_7,
                        rsi_14,
                        candlestick_patterns
                    FROM indicators_contexts
                    WHERE ticker = %s
                    ORDER BY ts DESC
                    LIMIT 1;
                    """,
                    (ticker,),
                )
            else:
                cur.execute(
                    """
                    SELECT 
                        ticker,
                        ts,
                        price,
                        ema9,
                        ema20,
                        ema21,
                        supertrend,
                        adx,
                        macd,
                        rsi_7,
                        rsi_14,
                        candlestick_patterns
                    FROM indicators_contexts
                    ORDER BY ts DESC
                    LIMIT 1;
                    """
                )
            row = cur.fetchone()

    if not row:
        return CurrentIndicators(
            ticker=None,
            timestamp=None,
            price=None,
            ema9=None,
            ema20=None,
            ema21=None,
            supertrend=None,
            adx=None,
            macd=None,
            rsi_7=None,
            rsi_14=None,
            candlestick_patterns=None,
        )

    return CurrentIndicators(
        ticker=row[0],
        timestamp=to_local_time(row[1]),
        price=float(row[2]) if row[2] else None,
        ema9=float(row[3]) if row[3] else None,
        ema20=float(row[4]) if row[4] else None,
        ema21=float(row[5]) if row[5] else None,
        supertrend=row[6],
        adx=float(row[7]) if row[7] else None,
        macd=float(row[8]) if row[8] else None,
        rsi_7=float(row[9]) if row[9] else None,
        rsi_14=float(row[10]) if row[10] else None,
        candlestick_patterns=row[11],
    )


@app.get("/risk-metrics", response_model=RiskMetrics)
def get_risk_metrics() -> RiskMetrics:
    """Restituisce metriche di rischio basate sulle posizioni aperte."""

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Ultimo snapshot
            cur.execute(
                """
                SELECT id, balance_usd
                FROM account_snapshots
                ORDER BY created_at DESC
                LIMIT 1;
                """
            )
            snapshot_row = cur.fetchone()
            
            if not snapshot_row:
                return RiskMetrics(
                    total_exposure_usd=0.0,
                    total_positions=0,
                    long_positions=0,
                    short_positions=0,
                    avg_leverage=None,
                    largest_position_pct=None,
                )

            snapshot_id = snapshot_row[0]
            balance = float(snapshot_row[1]) if snapshot_row[1] else 0.0

            # Statistiche posizioni
            try:
                cur.execute(
                    """
                    SELECT 
                        COUNT(*) as total,
                        COUNT(CASE WHEN side ILIKE 'long' THEN 1 END) as longs,
                        COUNT(CASE WHEN side ILIKE 'short' THEN 1 END) as shorts,
                        SUM(ABS(size * COALESCE(mark_price, entry_price, 0))) as total_exposure,
                        MAX(ABS(size * COALESCE(mark_price, entry_price, 0))) as largest_pos
                    FROM open_positions
                    WHERE snapshot_id = %s;
                    """,
                    (snapshot_id,),
                )
                pos_row = cur.fetchone()
            except Exception as e:
                print(f"Error in risk metrics query: {e}")
                pos_row = (0, 0, 0, 0.0, 0.0)

    total_positions = pos_row[0] if pos_row else 0
    long_positions = pos_row[1] if pos_row else 0
    short_positions = pos_row[2] if pos_row else 0
    total_exposure = float(pos_row[3]) if pos_row and pos_row[3] else 0.0
    largest_position = float(pos_row[4]) if pos_row and pos_row[4] else 0.0
    avg_leverage = None  # Rimosso calcolo problematico

    largest_position_pct = None
    if balance > 0 and largest_position > 0:
        largest_position_pct = (largest_position / balance) * 100

    return RiskMetrics(
        total_exposure_usd=total_exposure,
        total_positions=total_positions,
        long_positions=long_positions,
        short_positions=short_positions,
        avg_leverage=avg_leverage,
        largest_position_pct=largest_position_pct,
    )


@app.get("/last-operations-by-symbol", response_model=List[BotOperation])
def get_last_operations_by_symbol() -> List[BotOperation]:
    """Restituisce l'ultima operazione (incluso HOLD) per ogni symbol/valuta."""

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Query per ottenere l'ultima operazione per ogni symbol
            cur.execute(
                """
                WITH ranked_ops AS (
                    SELECT
                        bo.id,
                        bo.created_at,
                        bo.operation,
                        bo.symbol,
                        bo.direction,
                        bo.target_portion_of_balance,
                        bo.leverage,
                        bo.raw_payload,
                        ac.system_prompt,
                        ROW_NUMBER() OVER (PARTITION BY bo.symbol ORDER BY bo.created_at DESC) as rn
                    FROM bot_operations AS bo
                    LEFT JOIN ai_contexts AS ac ON bo.context_id = ac.id
                    WHERE bo.symbol IS NOT NULL
                )
                SELECT
                    id,
                    created_at,
                    operation,
                    symbol,
                    direction,
                    target_portion_of_balance,
                    leverage,
                    raw_payload,
                    system_prompt
                FROM ranked_ops
                WHERE rn = 1
                ORDER BY symbol ASC;
                """
            )
            rows = cur.fetchall()

    operations: List[BotOperation] = []
    for row in rows:
        operations.append(
            BotOperation(
                id=row[0],
                created_at=to_local_time(row[1]),
                operation=row[2],
                symbol=row[3],
                direction=row[4],
                target_portion_of_balance=float(row[5]) if row[5] is not None else None,
                leverage=float(row[6]) if row[6] is not None else None,
                raw_payload=row[7],
                system_prompt=row[8],
            )
        )

    return operations


# =====================
# Endpoint HTML + HTMX
# =====================


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request) -> HTMLResponse:
    """Dashboard principale HTML."""

    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/ui/balance", response_class=HTMLResponse)
async def ui_balance(request: Request) -> HTMLResponse:
    """Partial HTML con il grafico del saldo nel tempo."""

    points = get_balance()
    labels = [p.timestamp.isoformat() for p in points]
    values = [p.balance_usd for p in points]
    return templates.TemplateResponse(
        "partials/balance_table.html",
        {"request": request, "labels": labels, "values": values},
    )


@app.get("/ui/open-positions", response_class=HTMLResponse)
async def ui_open_positions(request: Request) -> HTMLResponse:
    """Partial HTML con le posizioni aperte (ultimo snapshot)."""

    positions = get_open_positions()
    return templates.TemplateResponse(
        "partials/open_positions_table.html",
        {"request": request, "positions": positions},
    )


@app.get("/ui/bot-operations", response_class=HTMLResponse)
async def ui_bot_operations(request: Request) -> HTMLResponse:
    """Partial HTML con le ultime operazioni del bot."""

    operations = get_bot_operations(limit=50)
    return templates.TemplateResponse(
        "partials/bot_operations_table.html",
        {"request": request, "operations": operations},
    )


@app.get("/history-page", response_class=HTMLResponse)
async def history_page(request: Request) -> HTMLResponse:
    """Pagina HTML dedicata allo storico operazioni."""
    return templates.TemplateResponse("history.html", {"request": request})


@app.get("/ui/history", response_class=HTMLResponse)
async def ui_history(request: Request) -> HTMLResponse:
    """Partial HTML per la tabella storico."""
    operations = get_history(limit=100)

    # Recupera equity corrente (ultimo balance registrato)
    current_equity: Optional[float] = None
    equity_timestamp: Optional[datetime] = None
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT balance_usd, created_at
                FROM account_snapshots
                ORDER BY created_at DESC
                LIMIT 1;
                """
            )
            row = cur.fetchone()
            if row:
                current_equity = float(row[0]) if row[0] is not None else None
                equity_timestamp = to_local_time(row[1])

    return templates.TemplateResponse(
        "partials/history_table.html",
        {
            "request": request,
            "operations": operations,
            "equity": current_equity,
            "equity_ts": equity_timestamp,
        },
    )


@app.get("/ui/performance-metrics", response_class=HTMLResponse)
async def ui_performance_metrics(request: Request) -> HTMLResponse:
    """Partial HTML per le metriche di performance."""
    metrics = get_performance_metrics()
    return templates.TemplateResponse(
        "partials/performance_metrics.html",
        {"request": request, "metrics": metrics},
    )


@app.get("/ui/current-indicators", response_class=HTMLResponse)
async def ui_current_indicators(
    request: Request,
    ticker: Optional[str] = Query(None, description="Filtra per ticker")
) -> HTMLResponse:
    """Partial HTML per gli indicatori correnti."""
    indicators = get_current_indicators(ticker=ticker)
    
    # Recupera lista ticker disponibili per il dropdown
    available_tickers: List[str] = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ticker 
                FROM indicators_contexts 
                WHERE ticker IS NOT NULL AND ticker != 'ETH'
                ORDER BY ticker;
                """
            )
            available_tickers = [row[0] for row in cur.fetchall()]
    
    return templates.TemplateResponse(
        "partials/current_indicators.html",
        {
            "request": request,
            "indicators": indicators,
            "available_tickers": available_tickers,
            "selected_ticker": ticker,
        },
    )


@app.get("/ui/risk-metrics", response_class=HTMLResponse)
async def ui_risk_metrics(request: Request) -> HTMLResponse:
    """Partial HTML per le metriche di rischio."""
    metrics = get_risk_metrics()
    return templates.TemplateResponse(
        "partials/risk_metrics.html",
        {"request": request, "metrics": metrics},
    )


@app.get("/ui/last-operations-by-symbol", response_class=HTMLResponse)
async def ui_last_operations_by_symbol(request: Request) -> HTMLResponse:
    """Partial HTML per le ultime operazioni AI per symbol."""
    operations = get_last_operations_by_symbol()
    return templates.TemplateResponse(
        "partials/last_operations_by_symbol.html",
        {"request": request, "operations": operations},
    )


# Comodo per sviluppo locale: `python main.py`
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
