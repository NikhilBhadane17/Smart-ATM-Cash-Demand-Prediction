import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.db import get_db

db = get_db()
cur = db.cursor()
cur.execute("DELETE FROM atm_data WHERE atm_id='0'")
db.commit()
cur.close()
db.close()
print('Deleted atm_id 0 if present')
