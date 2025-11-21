-- =============================================================================
-- Trading Agent Database Schema
-- Generated based on project requirements and ER diagram
-- =============================================================================

-- 1. AI Contexts
-- Tabella centrale che collega i dati di contesto usati dall'AI per prendere decisioni.
CREATE TABLE IF NOT EXISTS ai_contexts (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    system_prompt TEXT
);

-- 2. Account Snapshots
-- Registra lo stato del saldo nel tempo.
CREATE TABLE IF NOT EXISTS account_snapshots (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    balance_usd DOUBLE PRECISION,
    raw_payload JSONB
);

-- 3. Open Positions
-- Posizioni aperte collegate a uno specifico snapshot del conto.
CREATE TABLE IF NOT EXISTS open_positions (
    id SERIAL PRIMARY KEY,
    snapshot_id INTEGER REFERENCES account_snapshots(id) ON DELETE CASCADE,
    symbol TEXT,
    side TEXT, -- 'long' oppure 'short'
    size DOUBLE PRECISION,
    entry_price DOUBLE PRECISION,
    mark_price DOUBLE PRECISION,
    pnl_usd DOUBLE PRECISION,
    leverage TEXT, -- In main.py è definito come stringa opzionale
    raw_payload JSONB
);

-- 4. Bot Operations
-- Le decisioni operative prese dal bot, collegate al contesto AI che le ha generate.
CREATE TABLE IF NOT EXISTS bot_operations (
    id SERIAL PRIMARY KEY,
    context_id INTEGER REFERENCES ai_contexts(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    operation TEXT, -- es. 'buy', 'sell', 'hold'
    symbol TEXT,
    direction TEXT, -- 'long', 'short'
    target_portion_of_balance DOUBLE PRECISION,
    leverage DOUBLE PRECISION,
    raw_payload JSONB
);

-- 5. Indicators Contexts
-- Dati tecnici e indicatori di mercato analizzati.
CREATE TABLE IF NOT EXISTS indicators_contexts (
    id SERIAL PRIMARY KEY,
    context_id INTEGER REFERENCES ai_contexts(id) ON DELETE CASCADE,
    ticker TEXT,
    ts TIMESTAMP WITH TIME ZONE,
    price DOUBLE PRECISION,
    
    -- Indicatori Tecnici - EMA
    ema9 DOUBLE PRECISION,
    ema20 DOUBLE PRECISION,
    ema21 DOUBLE PRECISION,
    
    -- Indicatori Tecnici - Trend
    supertrend TEXT, -- 'BULLISH' o 'BEARISH'
    adx DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    
    -- Indicatori Tecnici - Momentum
    rsi_7 DOUBLE PRECISION,
    rsi_14 DOUBLE PRECISION,
    
    -- Volume e Derivati
    volume_bid DOUBLE PRECISION,
    volume_ask DOUBLE PRECISION,
    open_interest_latest DOUBLE PRECISION,
    open_interest_average DOUBLE PRECISION,
    funding_rate DOUBLE PRECISION,
    
    -- Indicatori su Timeframe 15m
    ema20_15m DOUBLE PRECISION,
    ema50_15m DOUBLE PRECISION,
    atr3_15m DOUBLE PRECISION,
    atr14_15m DOUBLE PRECISION,
    volume_15m_current DOUBLE PRECISION,
    volume_15m_average DOUBLE PRECISION,
    
    -- Pattern e Serie Temporali
    candlestick_patterns JSONB,
    intraday_mid_prices JSONB,
    intraday_ema20_series JSONB,
    intraday_macd_series JSONB,
    intraday_rsi7_series JSONB,
    intraday_rsi14_series JSONB,
    lt15m_macd_series JSONB,
    lt15m_rsi14_series JSONB,
    
    -- Pivot Points (DEPRECATED - mantenuti per compatibilità)
    pp DOUBLE PRECISION,
    s1 DOUBLE PRECISION,
    s2 DOUBLE PRECISION,
    r1 DOUBLE PRECISION,
    r2 DOUBLE PRECISION
);

-- 6. News Contexts
-- Notizie analizzate dall'AI.
CREATE TABLE IF NOT EXISTS news_contexts (
    id SERIAL PRIMARY KEY,
    context_id INTEGER REFERENCES ai_contexts(id) ON DELETE CASCADE,
    news_text TEXT
);

-- 7. Sentiment Contexts
-- Analisi del sentiment di mercato.
CREATE TABLE IF NOT EXISTS sentiment_contexts (
    id SERIAL PRIMARY KEY,
    context_id INTEGER REFERENCES ai_contexts(id) ON DELETE CASCADE,
    value DOUBLE PRECISION,
    classification TEXT,
    sentiment_timestamp TIMESTAMP WITH TIME ZONE,
    raw JSONB
);

-- 8. Forecasts Contexts
-- Previsioni generate dai modelli.
CREATE TABLE IF NOT EXISTS forecasts_contexts (
    id SERIAL PRIMARY KEY,
    context_id INTEGER REFERENCES ai_contexts(id) ON DELETE CASCADE,
    ticker TEXT,
    timeframe TEXT,
    last_price DOUBLE PRECISION,
    prediction TEXT,
    lower_bound DOUBLE PRECISION,
    upper_bound DOUBLE PRECISION,
    change_pct DOUBLE PRECISION,
    forecast_timestamp TIMESTAMP WITH TIME ZONE,
    raw JSONB
);

-- Indici per migliorare le performance delle query più frequenti
CREATE INDEX IF NOT EXISTS idx_account_snapshots_created_at ON account_snapshots(created_at);
CREATE INDEX IF NOT EXISTS idx_bot_operations_created_at ON bot_operations(created_at);
CREATE INDEX IF NOT EXISTS idx_open_positions_snapshot_id ON open_positions(snapshot_id);
