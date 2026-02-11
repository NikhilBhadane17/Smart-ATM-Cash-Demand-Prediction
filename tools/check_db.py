import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.db import get_db

if __name__ == '__main__':
    db = None
    cur = None
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute("SHOW TABLES;")
        tables = [r[0] for r in cur.fetchall()]
        print('Tables in database:', tables)

        # Check if atm_data exists
        if 'atm_data' in tables:
            cur.execute('SELECT atm_id, location, current_cash, prediction, status FROM atm_data LIMIT 10')
            rows = cur.fetchall()
            print('Sample rows (first 10):')
            for r in rows:
                print(r)
        else:
            print("Table 'atm_data' not found.\nYou may need to create it or adjust the query used by /api/atms.")
    except Exception as e:
        print('DB error:', e)
    finally:
        if cur:
            cur.close()
        if db:
            db.close()
