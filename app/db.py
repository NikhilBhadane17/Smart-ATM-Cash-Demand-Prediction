import mysql.connector


def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",        # XAMPP default
        database="atm"      # 🔴 MUST be 'atm' (from screenshot)
    )


def execute_query(db, query, params=(), fetchone=False, fetchall=False):
    """Execute a query using the given connection.

    - For SELECT queries: set `fetchone=True` or `fetchall=True` to return rows (dicts).
    - For non-SELECT queries: the function will commit and return the last inserted id.
    """
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute(query, params)
        if fetchone:
            return cursor.fetchone()
        if fetchall:
            return cursor.fetchall()
        # non-select: commit and return lastrowid
        db.commit()
        return cursor.lastrowid
    finally:
        cursor.close()
