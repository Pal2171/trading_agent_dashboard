"""
Script di test per verificare la connessione al database PostgreSQL.
Testa sia la connessione diretta che la compatibilita con il trading agent.
"""

import os
import sys
from dotenv import load_dotenv
import psycopg2
from urllib.parse import urlparse

# Carica variabili d'ambiente
load_dotenv()

def print_section(title):
    """Stampa una sezione formattata."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def test_database_url():
    """Verifica che DATABASE_URL sia configurata correttamente."""
    print_section("1. VERIFICA DATABASE_URL")
    
    database_url = os.getenv("DATABASE_URL")
    
    if not database_url:
        print("[X] ERRORE: DATABASE_URL non trovata!")
        print("   Crea un file .env con:")
        print("   DATABASE_URL=postgresql://user:password@host:port/database")
        return None
    
    print(f"[OK] DATABASE_URL trovata")
    
    # Parse URL per mostrare dettagli (senza password)
    try:
        parsed = urlparse(database_url)
        print(f"\n[INFO] Dettagli connessione:")
        print(f"   Schema:   {parsed.scheme}")
        print(f"   Host:     {parsed.hostname}")
        print(f"   Port:     {parsed.port}")
        print(f"   Database: {parsed.path.lstrip('/')}")
        print(f"   User:     {parsed.username}")
        print(f"   Password: {'*' * len(parsed.password) if parsed.password else 'N/A'}")
        
        # Verifica se e postgres:// invece di postgresql://
        if parsed.scheme == "postgres":
            print("\n[!] ATTENZIONE: URL usa 'postgres://' invece di 'postgresql://'")
            print("   Alcune librerie potrebbero avere problemi.")
            print("   Provo a convertire automaticamente...")
            database_url = database_url.replace("postgres://", "postgresql://", 1)
            print(f"   [OK] Convertito in: postgresql://...")
            
    except Exception as e:
        print(f"[!] Impossibile parsare URL: {e}")
    
    return database_url

def test_connection(database_url):
    """Testa la connessione al database."""
    print_section("2. TEST CONNESSIONE DATABASE")
    
    if not database_url:
        print("[X] Impossibile testare: DATABASE_URL non valida")
        return False
    
    try:
        print("[...] Tentativo di connessione...")
        conn = psycopg2.connect(database_url)
        print("[OK] Connessione riuscita!")
        
        # Testa una query semplice
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            version = cur.fetchone()[0]
            print(f"\n[INFO] Versione PostgreSQL:")
            print(f"   {version[:80]}...")
        
        conn.close()
        return True
        
    except psycopg2.OperationalError as e:
        print(f"[X] ERRORE di connessione:")
        print(f"   {str(e)}")
        
        # Suggerimenti basati sull'errore
        error_str = str(e).lower()
        print("\n[TIP] Possibili soluzioni:")
        
        if "could not connect" in error_str or "connection refused" in error_str:
            print("   - Verifica che il database sia online")
            print("   - Controlla host e porta")
            print("   - Verifica le regole del firewall")
            
        if "authentication failed" in error_str or "password" in error_str:
            print("   - Verifica username e password")
            print("   - Controlla i permessi dell'utente")
            
        if "ssl" in error_str:
            print("   - Il database potrebbe richiedere SSL")
            print("   - Prova ad aggiungere ?sslmode=require all'URL")
            
        if "does not exist" in error_str:
            print("   - Verifica che il database esista")
            print("   - Controlla il nome del database nell'URL")
            
        return False
        
    except Exception as e:
        print(f"[X] ERRORE inaspettato: {type(e).__name__}")
        print(f"   {str(e)}")
        return False

def test_tables():
    """Verifica l'esistenza delle tabelle necessarie."""
    print_section("3. VERIFICA TABELLE DATABASE")
    
    database_url = os.getenv("DATABASE_URL")
    if database_url and database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    
    if not database_url:
        print("[X] Impossibile testare: DATABASE_URL non valida")
        return
    
    try:
        conn = psycopg2.connect(database_url)
        
        # Tabelle richieste dal trading agent
        required_tables = [
            'account_snapshots',
            'open_positions',
            'bot_operations',
            'ai_contexts',
            'indicators_contexts'
        ]
        
        with conn.cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            existing_tables = [row[0] for row in cur.fetchall()]
        
        print(f"[INFO] Tabelle trovate: {len(existing_tables)}")
        
        for table in required_tables:
            if table in existing_tables:
                # Conta i record
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {table};")
                    count = cur.fetchone()[0]
                print(f"   [OK] {table:<25} ({count} record)")
            else:
                print(f"   [X] {table:<25} (MANCANTE)")
        
        # Mostra tabelle extra
        extra_tables = set(existing_tables) - set(required_tables)
        if extra_tables:
            print(f"\n[INFO] Altre tabelle presenti:")
            for table in sorted(extra_tables):
                print(f"   - {table}")
        
        conn.close()
        
    except Exception as e:
        print(f"[X] Errore durante verifica tabelle: {e}")

def test_recent_data():
    """Mostra i dati piu recenti per debug."""
    print_section("4. VERIFICA DATI RECENTI")
    
    database_url = os.getenv("DATABASE_URL")
    if database_url and database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    
    if not database_url:
        print("[X] Impossibile testare: DATABASE_URL non valida")
        return
    
    try:
        conn = psycopg2.connect(database_url)
        
        # Ultimo snapshot
        with conn.cursor() as cur:
            cur.execute("""
                SELECT created_at, balance_usd 
                FROM account_snapshots 
                ORDER BY created_at DESC 
                LIMIT 1;
            """)
            row = cur.fetchone()
            if row:
                print(f"[BALANCE] Ultimo snapshot account:")
                print(f"   Data:    {row[0]}")
                print(f"   Balance: ${row[1]:,.2f}")
            else:
                print("[!] Nessun snapshot trovato")
        
        # Ultima operazione
        with conn.cursor() as cur:
            cur.execute("""
                SELECT created_at, operation, symbol 
                FROM bot_operations 
                ORDER BY created_at DESC 
                LIMIT 1;
            """)
            row = cur.fetchone()
            if row:
                print(f"\n[BOT] Ultima operazione bot:")
                print(f"   Data:      {row[0]}")
                print(f"   Operation: {row[1]}")
                print(f"   Symbol:    {row[2] or 'N/A'}")
            else:
                print("\n[!] Nessuna operazione trovata")
        
        # Posizioni aperte
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) 
                FROM open_positions op
                JOIN account_snapshots s ON op.snapshot_id = s.id
                WHERE s.created_at = (
                    SELECT MAX(created_at) FROM account_snapshots
                );
            """)
            count = cur.fetchone()[0]
            print(f"\n[POSITIONS] Posizioni aperte correnti: {count}")
        
        conn.close()
        
    except Exception as e:
        print(f"[X] Errore durante verifica dati: {e}")

def main():
    """Esegue tutti i test."""
    print("\n" + "=" * 60)
    print("  TEST CONNESSIONE DATABASE - TRADING AGENT DASHBOARD")
    print("=" * 60)
    
    # Test 1: Verifica DATABASE_URL
    database_url = test_database_url()
    
    if not database_url:
        print("\n[X] Test interrotto: DATABASE_URL non configurata")
        sys.exit(1)
    
    # Test 2: Connessione
    connection_ok = test_connection(database_url)
    
    if not connection_ok:
        print("\n[X] Test interrotto: connessione fallita")
        sys.exit(1)
    
    # Test 3: Tabelle
    test_tables()
    
    # Test 4: Dati recenti
    test_recent_data()
    
    # Riepilogo finale
    print_section("RIEPILOGO")
    print("[OK] Tutti i test completati con successo!")
    print("\n[TIP] La dashboard puo connettersi al database.")
    print("   Se il Trading Agent non riesce a connettersi, verifica:")
    print("   1. Che usi la stessa DATABASE_URL")
    print("   2. Che abbia le stesse dipendenze (psycopg2)")
    print("   3. Che non ci siano problemi di SSL/firewall")
    print("   4. Che le variabili d'ambiente siano caricate correttamente")
    
    print("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    main()

