import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.db import get_db

db = get_db()
cur = db.cursor()
cur.execute("ALTER TABLE atm_data MODIFY atm_id VARCHAR(64) NOT NULL;")
cur.execute("ALTER TABLE atm_data MODIFY location VARCHAR(128);")
cur.execute("ALTER TABLE atm_data MODIFY current_cash BIGINT;")
cur.execute("ALTER TABLE atm_data MODIFY prediction BIGINT;")
cur.execute("ALTER TABLE atm_data MODIFY status VARCHAR(32);")
db.commit()
cur.close()
db.close()
print('Altered table types')
