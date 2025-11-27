import os
import psycopg2
import json
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

try:
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    print("Columns in 'open_positions':")
    cur.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'open_positions'
    """)
    for col in cur.fetchall():
        print(f"  - {col[0]} ({col[1]})")

    print("\nSample 'CLOSE' operations from 'bot_operations' (checking raw_payload):")
    cur.execute("""
        SELECT raw_payload 
        FROM bot_operations 
        WHERE operation = 'CLOSE' 
        LIMIT 3
    """)
    rows = cur.fetchall()
    for row in rows:
        print(f"  - {row[0]}")

    conn.close()

except Exception as e:
    print(f"Error: {e}")
