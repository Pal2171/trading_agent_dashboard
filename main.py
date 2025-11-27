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
    supertrend: Optional[str]
    adx: Optional[float]
    macd: Optional[float]
    rsi_7: Optional[float]
    rsi_14: Optional[float]


class RiskMetrics(BaseModel):
    total_exposure_usd: float


class SentimentData(BaseModel):
    value: Optional[int]
    classification: Optional[str]
    timestamp: Optional[datetime]


class NewsItem(BaseModel):
    title: str
    source: Optional[str] = None
    timestamp: Optional[datetime] = None


class ClosedTrade(BaseModel):
    symbol: str
    side: str
    entry_price: float
    exit_price: Optional[float]
    pnl_usd: float
    pnl_pct: Optional[float] = None
    open_time: datetime
    close_time: datetime
    motivation: Optional[str]


class WinLossMetrics(BaseModel):
    win_rate: float
    total_wins: int
    total_losses: int
    total_pnl_usd: float
    trades: List[ClosedTrade]


# Nuovo modello per posizioni reali (v2)
class RealPosition(BaseModel):
    deal_id: str
    symbol: str
    direction: str
    size: float
    entry_price: float
    mark_price: Optional[float]
    pnl_usd: Optional[float]
    pnl_pct: Optional[float]
    stop_level: Optional[float]
    limit_level: Optional[float]
    leverage: Optional[int]
    opened_at: Optional[datetime]
    last_update: Optional[datetime]


# Nuovo modello per storico trade (v2)
class TradeHistoryItem(BaseModel):
    deal_id: str
    symbol: str
    direction: str
    size: float
    entry_price: float
    close_price: Optional[float]
    pnl_usd: Optional[float]
    pnl_pct: Optional[float]
    opened_at: Optional[datetime]
    closed_at: datetime
    close_reason: Optional[str]


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

# Filtro Jinja per formattare decimali con separatore "," e numero di decimali personalizzabile
def format_decimal(value, decimals: int = 4):
    try:
        if value is None:
            return "-"
        return f"{float(value):.{decimals}f}".replace('.', ',')
    except Exception:
        return value

templates.env.filters['format_decimal'] = format_decimal


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


@app.get("/real-positions", response_model=List[RealPosition])
def get_real_positions() -> List[RealPosition]:
    """Restituisce le posizioni REALI attualmente aperte.
    
    Usa la tabella `real_positions` che contiene dati accurati
    sincronizzati dal bot tramite UPSERT (nessun duplicato).
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    deal_id,
                    symbol,
                    direction,
                    size,
                    entry_price,
                    mark_price,
                    pnl_usd,
                    pnl_pct,
                    stop_level,
                    limit_level,
                    leverage,
                    opened_at,
                    last_update
                FROM real_positions
                ORDER BY opened_at DESC
                """
            )
            rows = cur.fetchall()

    return [
        RealPosition(
            deal_id=row[0],
            symbol=row[1],
            direction=row[2],
            size=float(row[3]) if row[3] else 0,
            entry_price=float(row[4]) if row[4] else None,
            mark_price=float(row[5]) if row[5] else None,
            pnl_usd=float(row[6]) if row[6] else None,
            pnl_pct=float(row[7]) if row[7] else None,
            stop_level=float(row[8]) if row[8] else None,
            limit_level=float(row[9]) if row[9] else None,
            leverage=row[10],
            opened_at=row[11],
            last_update=row[12],
        )
        for row in rows
    ]


@app.get("/trades-history", response_model=List[TradeHistoryItem])
def get_trades_history(
    limit: int = Query(
        100,
        ge=1,
        le=1000,
        description="Numero massimo di trade da restituire (default 100)",
    ),
) -> List[TradeHistoryItem]:
    """Restituisce lo storico dei trade chiusi.
    
    Usa la tabella `trades_history` che contiene tutti i trade
    completati con P/L calcolato e motivo di chiusura.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    deal_id,
                    symbol,
                    direction,
                    size,
                    entry_price,
                    close_price,
                    pnl_usd,
                    pnl_pct,
                    opened_at,
                    closed_at,
                    close_reason
                FROM trades_history
                ORDER BY closed_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()

    return [
        TradeHistoryItem(
            deal_id=row[0],
            symbol=row[1],
            direction=row[2],
            size=float(row[3]) if row[3] else 0,
            entry_price=float(row[4]) if row[4] else None,
            close_price=float(row[5]) if row[5] else None,
            pnl_usd=float(row[6]) if row[6] else None,
            pnl_pct=float(row[7]) if row[7] else None,
            opened_at=row[8],
            closed_at=row[9],
            close_reason=row[10],
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
                        supertrend_1h,
                        adx,
                        macd,
                        rsi_7,
                        rsi_14
                    FROM indicators_contexts
                    WHERE ticker = %s
                    ORDER BY ts DESC
                    LIMIT 1;
                    """,
                    (ticker,),
                )
            else:
                # Quando non specificato ticker, prendi il più recente tra tutti
                cur.execute(
                    """
                    WITH ranked AS (
                        SELECT 
                            ticker,
                            ts,
                            price,
                            ema9,
                            ema20,
                            supertrend_1h,
                            adx,
                            macd,
                            rsi_7,
                            rsi_14,
                            ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY ts DESC) as rn
                        FROM indicators_contexts
                        WHERE ticker IS NOT NULL
                    )
                    SELECT 
                        ticker,
                        ts,
                        price,
                        ema9,
                        ema20,
                        supertrend_1h,
                        adx,
                        macd,
                        rsi_7,
                        rsi_14
                    FROM ranked
                    WHERE rn = 1
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
            supertrend=None,
            adx=None,
            macd=None,
            rsi_7=None,
            rsi_14=None,
        )

    return CurrentIndicators(
        ticker=row[0],
        timestamp=to_local_time(row[1]),
        price=float(row[2]) if row[2] else None,
        ema9=float(row[3]) if row[3] else None,
        ema20=float(row[4]) if row[4] else None,
        supertrend=row[5],
        adx=float(row[6]) if row[6] else None,
        macd=float(row[7]) if row[7] else None,
        rsi_7=float(row[8]) if row[8] else None,
        rsi_14=float(row[9]) if row[9] else None,
    )


    return operations


def calculate_closed_trades_logic() -> WinLossMetrics:
    """Calcola le posizioni chiuse confrontando gli snapshot."""
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            # 1. Recupera tutti gli snapshot ordinati
            cur.execute("SELECT id, created_at FROM account_snapshots ORDER BY created_at ASC")
            snapshots = cur.fetchall()
            
            if not snapshots:
                return WinLossMetrics(win_rate=0, total_wins=0, total_losses=0, total_pnl_usd=0, trades=[])

            # 2. Recupera tutte le posizioni aperte
            cur.execute("""
                SELECT snapshot_id, symbol, side, size, entry_price, mark_price, pnl_usd 
                FROM open_positions 
                ORDER BY snapshot_id ASC
            """)
            all_positions = cur.fetchall()
            
            # 3. Recupera operazioni CLOSE per le motivazioni
            cur.execute("""
                SELECT bo.created_at, bo.symbol, bo.direction, ac.system_prompt, bo.raw_payload
                FROM bot_operations bo
                LEFT JOIN ai_contexts ac ON bo.context_id = ac.id
                WHERE bo.operation = 'CLOSE'
                ORDER BY bo.created_at ASC
            """)
            close_ops = cur.fetchall()

    # Organizza posizioni per snapshot
    positions_by_snapshot = {}
    for pos in all_positions:
        snap_id = pos[0]
        if snap_id not in positions_by_snapshot:
            positions_by_snapshot[snap_id] = []
        positions_by_snapshot[snap_id].append({
            'symbol': pos[1],
            'side': pos[2],
            'size': float(pos[3]),
            'entry_price': float(pos[4]) if pos[4] else 0,
            'mark_price': float(pos[5]) if pos[5] else 0,
            'pnl_usd': float(pos[6]) if pos[6] else 0
        })

    closed_trades = []
    
    # Itera sugli snapshot per trovare quando una posizione sparisce
    for i in range(len(snapshots) - 1):
        curr_snap_id = snapshots[i][0]
        next_snap_id = snapshots[i+1][0]
        curr_time = snapshots[i][1]
        next_time = snapshots[i+1][1]
        
        curr_pos_list = positions_by_snapshot.get(curr_snap_id, [])
        next_pos_list = positions_by_snapshot.get(next_snap_id, [])
        
        # Crea mappa per lookup rapido nel prossimo snapshot
        next_pos_map = {(p['symbol'], p['side']): p for p in next_pos_list}
        
        for pos in curr_pos_list:
            key = (pos['symbol'], pos['side'])
            if key not in next_pos_map:
                # La posizione è sparita -> CHIUSA
                # Cerca motivazione nel range temporale [curr_time, next_time + buffer]
                motivation = "Chiusura rilevata da snapshot"
                
                # Cerca l'operazione CLOSE più vicina
                best_match = None
                min_diff = float('inf')
                
                for op in close_ops:
                    op_time = op[0]
                    op_symbol = op[1]
                    # op_direction è solitamente 'LONG' o 'SHORT' che indica cosa si sta chiudendo, o la direzione del trade?
                    # Assumiamo che se chiudo un LONG, direction potrebbe essere null o specificata.
                    # Controlliamo solo symbol e tempo per ora.
                    
                    if op_symbol == pos['symbol'] and curr_time <= op_time <= (next_time + (next_time - curr_time)):
                        # Trovato un close compatibile temporalmente
                        diff = abs((op_time - next_time).total_seconds())
                        if diff < min_diff:
                            min_diff = diff
                            best_match = op
                
                if best_match:
                    # Usa system prompt o raw payload come motivazione
                    # best_match[3] è system_prompt, best_match[4] è raw_payload
                    if best_match[3]:
                        motivation = best_match[3][:100] + "..." # Tronca per brevità
                    else:
                        motivation = "Decisione AI (prompt mancante)"

                # Ricalcolo PnL coerente con direzione (long/short)
                entry_price = pos['entry_price']
                exit_price = pos['mark_price']
                size = pos['size']
                pnl_usd = 0.0
                if entry_price and exit_price and size:
                    price_diff = exit_price - entry_price
                    # Se short, profit quando exit < entry -> inverti segno
                    if pos['side'] and pos['side'].lower() == 'short':
                        price_diff = -price_diff
                    pnl_usd = price_diff * size

                closed_trades.append(ClosedTrade(
                    symbol=pos['symbol'],
                    side=pos['side'],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl_usd=pnl_usd,
                    open_time=to_local_time(curr_time),
                    close_time=to_local_time(next_time),
                    motivation=motivation
                ))

    # Calcola metriche
    wins = [t for t in closed_trades if t.pnl_usd > 0]
    losses = [t for t in closed_trades if t.pnl_usd <= 0]
    
    total_wins = len(wins)
    total_losses = len(losses)
    total_trades = total_wins + total_losses
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0
    total_pnl = sum(t.pnl_usd for t in closed_trades)
    
    # Ordina trades per data chiusura decrescente
    closed_trades.sort(key=lambda x: x.close_time, reverse=True)

    return WinLossMetrics(
        win_rate=win_rate,
        total_wins=total_wins,
        total_losses=total_losses,
        total_pnl_usd=total_pnl,
        trades=closed_trades
    )


@app.get("/last-operations-by-symbol", response_model=List[BotOperation])
def get_last_operations_by_symbol() -> List[BotOperation]:
    """Restituisce l'ultima operazione (incluso HOLD) solo per symbol con posizioni aperte."""

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Prima ottieni i symbol con posizioni aperte dall'ultimo snapshot
            cur.execute(
                """
                SELECT DISTINCT op.symbol
                FROM open_positions op
                INNER JOIN (
                    SELECT id FROM account_snapshots
                    ORDER BY created_at DESC LIMIT 1
                ) snap ON op.snapshot_id = snap.id;
                """
            )
            active_symbols = {row[0] for row in cur.fetchall()}
            
            if not active_symbols:
                return []
            
            # Query per ottenere l'ultima operazione per ogni symbol attivo
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
                    WHERE bo.symbol = ANY(%s)
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
                """,
                (list(active_symbols),)
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


@app.get("/win-loss-metrics", response_model=WinLossMetrics)
def get_win_loss_metrics() -> WinLossMetrics:
    """Restituisce metriche sulle operazioni chiuse (stimate)."""
    return calculate_closed_trades_logic()


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
    """Partial HTML con le posizioni aperte (dalla tabella real_positions)."""

    positions = get_real_positions()
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
    """Partial HTML per la tabella storico (Closed Trades)."""
    metrics = calculate_closed_trades_logic()
    trades = metrics.trades

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
            "operations": trades, # Passa i ClosedTrade invece di BotOperation
            "equity": current_equity,
            "equity_ts": equity_timestamp,
        },
    )


@app.get("/ui/performance-metrics", response_class=HTMLResponse)
async def ui_performance_metrics(request: Request) -> HTMLResponse:
    """Partial HTML per le metriche di performance."""
    metrics = get_performance_metrics()
    win_loss = get_win_loss_metrics()
    return templates.TemplateResponse(
        "partials/performance_metrics.html",
        {"request": request, "metrics": metrics, "win_loss": win_loss},
    )


@app.get("/ui/current-indicators", response_class=HTMLResponse)
async def ui_current_indicators(
    request: Request,
    ticker: Optional[str] = Query(None, description="Filtra per ticker")
) -> HTMLResponse:
    """Partial HTML per gli indicatori correnti."""
    try:
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
    except Exception as e:
        # Gestione errori per evitare loader infinito
        return templates.TemplateResponse(
            "partials/current_indicators.html",
            {
                "request": request,
                "indicators": CurrentIndicators(
                    ticker=None,
                    timestamp=None,
                    price=None,
                    ema9=None,
                    ema20=None,
                    supertrend=None,
                    adx=None,
                    macd=None,
                    rsi_7=None,
                    rsi_14=None,
                ),
                "available_tickers": [],
                "selected_ticker": ticker,
                "error": str(e),
            },
        )


@app.get("/ui/win-loss-metrics", response_class=HTMLResponse)
async def ui_win_loss_metrics(request: Request) -> HTMLResponse:
    """Partial HTML per le metriche Win/Loss."""
    metrics = get_win_loss_metrics()
    return templates.TemplateResponse(
        "partials/win_loss_metrics.html",
        {"request": request, "metrics": metrics},
    )


@app.get("/ui/sentiment-news", response_class=HTMLResponse)
async def ui_sentiment_news(request: Request) -> HTMLResponse:
    """Partial HTML per Sentiment e ultime News."""
    
    sentiment_data = None
    news_items = []
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Recupera l'ultimo sentiment
            cur.execute(
                """
                SELECT value, classification, sentiment_timestamp
                FROM sentiment_contexts
                ORDER BY id DESC
                LIMIT 1;
                """
            )
            row = cur.fetchone()
            if row:
                # sentiment_timestamp è un BIGINT (unix timestamp)
                ts = None
                if row[2]:
                    try:
                        ts = to_local_time(datetime.utcfromtimestamp(row[2]))
                    except:
                        pass
                sentiment_data = SentimentData(
                    value=row[0],
                    classification=row[1],
                    timestamp=ts
                )
            
            # Recupera le ultime 5 news (dal campo news_text che contiene testo formattato)
            cur.execute(
                """
                SELECT nc.news_text, ac.created_at
                FROM news_contexts nc
                JOIN ai_contexts ac ON nc.context_id = ac.id
                ORDER BY nc.id DESC
                LIMIT 1;
                """
            )
            news_row = cur.fetchone()
            if news_row and news_row[0]:
                # Il news_text contiene testo formattato, lo parsiamo per estrarre titoli
                news_text = news_row[0]
                timestamp = to_local_time(news_row[1]) if news_row[1] else None
                
                # Estrai linee che sembrano titoli di news (ignora linee vuote e separatori)
                lines = news_text.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    # Salta linee vuote, separatori, e header
                    if not line or line.startswith('---') or line.startswith('==='):
                        continue
                    if line.lower().startswith('here are') or line.lower().startswith('latest'):
                        continue
                    if len(line) > 15:  # Solo linee abbastanza lunghe
                        # Cerca di estrarre la fonte se presente (es. "[CoinDesk]" o "Source:")
                        source = "Crypto"
                        if line.startswith('[') and ']' in line:
                            source = line[1:line.index(']')]
                            line = line[line.index(']')+1:].strip()
                        elif ':' in line[:30]:
                            parts = line.split(':', 1)
                            if len(parts[0]) < 20:
                                source = parts[0]
                                line = parts[1].strip()
                        
                        news_items.append(NewsItem(
                            title=line[:200],  # Tronca a 200 char
                            source=source,
                            timestamp=timestamp
                        ))
                        
                        if len(news_items) >= 5:
                            break
    
    return templates.TemplateResponse(
        "partials/sentiment_news.html",
        {
            "request": request,
            "sentiment": sentiment_data,
            "news_items": news_items,
        },
    )


@app.get("/ui/last-operations-by-symbol", response_class=HTMLResponse)
async def ui_last_operations_by_symbol(request: Request) -> HTMLResponse:
    """Partial HTML per le ultime operazioni AI per symbol."""
    operations = get_last_operations_by_symbol()
    # Mappa posizioni aperte per symbol per mostrare LONG/SHORT anche se ultimo segnale è HOLD
    open_positions = get_open_positions()
    open_pos_map = {p.symbol: {"side": p.side} for p in open_positions}
    return templates.TemplateResponse(
        "partials/last_operations_by_symbol.html",
        {
            "request": request,
            "operations": operations,
            "open_pos_map": open_pos_map,
        },
    )


# Comodo per sviluppo locale: `python main.py`
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
