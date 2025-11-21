import os
import psycopg2
from dotenv import load_dotenv

# Carica le variabili d'ambiente (in particolare DATABASE_URL)
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def init_db():
    if not DATABASE_URL:
        print("Errore: DATABASE_URL non trovata.")
        print("Assicurati di aver creato un file .env con: DATABASE_URL=...")
        print("Oppure imposta la variabile d'ambiente nel tuo terminale.")
        return

    # Leggi il file SQL
    schema_file = "database_schema.sql"
    if not os.path.exists(schema_file):
        print(f"Errore: File {schema_file} non trovato.")
        return

    with open(schema_file, "r") as f:
        schema_sql = f.read()

    print(f"Connessione al database in corso...")
    try:
        # Connessione al database
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()

        print("Esecuzione dello script SQL...")
        cur.execute(schema_sql)
        
        # Conferma le modifiche
        conn.commit()
        
        cur.close()
        conn.close()
        print("Successo! Tutte le tabelle sono state create correttamente.")
        
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

if __name__ == "__main__":
    init_db()
