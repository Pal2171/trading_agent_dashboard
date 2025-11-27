import os
import psycopg2
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def to_local_time(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(ZoneInfo("Europe/Rome"))

def verify():
    print("--- Verifying Time Zone Logic (No DB) ---")
    
    # Test case 1: Naive datetime (assumed UTC)
    naive_dt = datetime(2025, 11, 23, 12, 0, 0)
    local_dt = to_local_time(naive_dt)
    print(f"\nTest 1: Naive UTC Input")
    print(f"  Input:      {naive_dt}")
    print(f"  Converted:  {local_dt}")
    
    # Test case 2: Aware UTC datetime
    aware_dt = datetime(2025, 11, 23, 12, 0, 0, tzinfo=ZoneInfo("UTC"))
    local_dt_2 = to_local_time(aware_dt)
    print(f"\nTest 2: Aware UTC Input")
    print(f"  Input:      {aware_dt}")
    print(f"  Converted:  {local_dt_2}")

    # Check offset
    expected_offset = "+01:00" # CET
    if str(local_dt)[-6:] == expected_offset:
        print(f"\nSUCCESS: Offset is {expected_offset} as expected for November.")
    else:
        print(f"\nFAILURE: Expected offset {expected_offset}, got {str(local_dt)[-6:]}")

if __name__ == "__main__":
    verify()
