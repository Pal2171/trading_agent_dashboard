import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("DATABASE_URL not found")
    exit(1)

try:
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # List all tables
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """)
    tables = cur.fetchall()
    print("Tables found:")
    for table in tables:
        print(f"- {table[0]}")

    # Inspect columns for specific tables
    target_tables = ['bot_operations', 'indicators_contexts', 'account_snapshots', 'open_positions']
    # Add any new tables found to this list if needed, but for now let's check these plus any obvious new ones from the list above.
    
    print("\nColumns in 'bot_operations':")
    cur.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'bot_operations'
    """)
    for col in cur.fetchall():
        print(f"  - {col[0]} ({col[1]})")

    print("\nColumns in 'indicators_contexts':")
    cur.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'indicators_contexts'
    """)
    for col in cur.fetchall():
        print(f"  - {col[0]} ({col[1]})")

    # Check for other potential indicator tables
    print("\nChecking for other indicator tables...")
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_name LIKE '%indicator%'
    """)
    other_ind_tables = cur.fetchall()
    for t in other_ind_tables:
        if t[0] != 'indicators_contexts':
            print(f"Found table: {t[0]}")
            print(f"Columns in '{t[0]}':")
            cur.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{t[0]}'
            """)
            for col in cur.fetchall():
                print(f"  - {col[0]} ({col[1]})")

    conn.close()

except Exception as e:
    print(f"Error: {e}")
