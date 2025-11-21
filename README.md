# 📊 Trading Agent Dashboard

Dashboard web interattiva per monitorare in tempo reale le performance e l'attività del Trading Bot AI.

## ✨ Caratteristiche Principali

### 📈 Visualizzazioni Avanzate

- **Equity Curve Migliorata** - Grafico dell'andamento del portafoglio con statistiche integrate
- **Performance Metrics** - Metriche chiave: rendimento totale, drawdown, conteggio operazioni
- **Indicatori di Mercato** - Visualizzazione real-time di tutti gli indicatori tecnici:
  - EMA (9, 20, 21)
  - Supertrend (BULLISH/BEARISH)
  - ADX (forza del trend)
  - RSI (7 e 14 periodi)
  - MACD
  - Candlestick Patterns rilevati
- **Risk Management** - Metriche di esposizione, bilanciamento Long/Short, leverage medio
- **Posizioni Aperte** - Tabella dettagliata con P&L, variazioni percentuali e statistiche aggregate
- **Storico Operazioni** - Cronologia completa delle decisioni del bot con reasoning AI

### 🔄 Aggiornamento Automatico

Tutti i componenti si aggiornano automaticamente ogni 30-60 secondi via HTMX, senza ricaricare la pagina.

## 🚀 Quick Start

### Installazione

```bash
# Clona il repository
git clone https://github.com/Pal2171/trading_agent_dashboard.git
cd trading_agent_dashboard

# Installa le dipendenze
pip install -r requirements.txt

# Configura il database
cp .env.example .env
# Modifica .env e inserisci il DATABASE_URL di Railway
```

### Configurazione

Crea un file `.env` nella root del progetto:

```env
DATABASE_URL=postgresql://user:password@host:port/database
```

### Avvio

```bash
# Sviluppo locale
python main.py

# Oppure con uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

La dashboard sarà disponibile su `http://localhost:8000`

## 📁 Struttura del Progetto

```
trading_agent_dashboard/
├── main.py                          # Backend FastAPI
├── database_schema.sql              # Schema database PostgreSQL
├── init_db.py                       # Script inizializzazione DB
├── requirements.txt                 # Dipendenze Python
├── templates/
│   ├── base.html                    # Template base con navbar e stili
│   ├── dashboard.html               # Dashboard principale
│   ├── history.html                 # Pagina storico operazioni
│   └── partials/
│       ├── balance_table.html       # Equity curve con Chart.js
│       ├── performance_metrics.html # Metriche performance
│       ├── current_indicators.html  # Indicatori di mercato
│       ├── risk_metrics.html        # Metriche di rischio
│       ├── open_positions_table.html # Tabella posizioni aperte
│       ├── bot_operations_table.html # Operazioni bot
│       └── history_table.html       # Storico completo
└── README.md
```

## 🎨 Componenti della Dashboard

### 1. Performance Overview
Mostra le metriche chiave:
- Saldo corrente
- Rendimento totale (% e USD)
- Max drawdown
- Conteggio operazioni (Buy/Sell/Hold)

### 2. Equity Curve
Grafico interattivo con:
- Andamento del saldo nel tempo
- Linea valore iniziale
- Linea massimo storico
- Statistiche quick access sopra il grafico

### 3. Indicatori di Mercato
Card con gli ultimi indicatori tecnici disponibili:
- **Supertrend** con badge colorati (🟢 BULLISH / 🔴 BEARISH)
- **ADX** con indicatore di forza trend
- **EMA** (9, 20, 21) per analisi trend
- **RSI** (7, 14) con evidenziazione zone ipercomprato/ipervenduto
- **MACD** per momentum
- **Candlestick Patterns** con interpretazione e descrizioni

### 4. Gestione del Rischio
Visualizza:
- Numero totale posizioni e split Long/Short
- Esposizione totale in USD
- Leverage medio utilizzato
- Posizione più grande (% del capitale)
- Barra di bilanciamento Long/Short

### 5. Posizioni Aperte
Tabella con:
- Simbolo, direzione, size
- Entry price vs Mark price
- Variazione percentuale
- Leverage applicato
- P&L in tempo reale
- Statistiche aggregate (totale P&L, count Long/Short)

## 🛠️ Stack Tecnologico

- **Backend**: FastAPI (Python)
- **Database**: PostgreSQL (Railway)
- **Frontend**: HTML + HTMX + Pico CSS
- **Grafici**: Chart.js
- **Auto-refresh**: HTMX polling

## 📊 API Endpoints

### JSON Endpoints

- `GET /balance` - Tutti gli snapshot del saldo
- `GET /open-positions` - Posizioni aperte (ultimo snapshot)
- `GET /bot-operations?limit=50` - Ultime operazioni del bot
- `GET /history?limit=100` - Storico operazioni (esclusi HOLD)
- `GET /performance-metrics` - Metriche di performance aggregate
- `GET /current-indicators` - Ultimi indicatori tecnici disponibili
- `GET /risk-metrics` - Metriche di rischio e esposizione

### HTML Partials (HTMX)

- `GET /ui/balance` - Partial equity curve
- `GET /ui/performance-metrics` - Partial metriche performance
- `GET /ui/current-indicators` - Partial indicatori
- `GET /ui/risk-metrics` - Partial risk management
- `GET /ui/open-positions` - Partial posizioni aperte
- `GET /ui/bot-operations` - Partial operazioni bot
- `GET /ui/history` - Partial storico completo

## 🔗 Integrazione con Trading System

Questa dashboard è **READ-ONLY** e si connette allo stesso database PostgreSQL utilizzato dal Trading System.

**Flusso dati**:
```
Trading System → PostgreSQL (Railway) → Dashboard
   (WRITE)            (DATABASE)          (READ)
```

Il Trading System scrive i dati, la dashboard li visualizza.

## 📝 Note

- **Auto-refresh**: I componenti si aggiornano automaticamente ogni 30-60 secondi
- **Responsive**: La dashboard è ottimizzata per desktop e mobile
- **Performance**: Le query sono ottimizzate con indici sul database
- **Sicurezza**: La dashboard è read-only, non può modificare dati

## 📄 Licenza

MIT License

## 👤 Autore

**Pal2171**
- GitHub: [@Pal2171](https://github.com/Pal2171) 
