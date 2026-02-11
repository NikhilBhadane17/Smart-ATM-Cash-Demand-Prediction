import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.db import get_db
import pandas as pd
import joblib
import numpy as np

PROCESSED = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'processed_atm_transactions_5yr.csv'))
MODEL = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'model', 'atm_cash_model.pkl'))
METADATA = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'model', 'model_metadata.pkl'))


def main():
    if not os.path.exists(PROCESSED):
        print('Processed dataset not found:', PROCESSED)
        return

    df = pd.read_csv(PROCESSED, parse_dates=['date'])
    if 'atm_id' not in df.columns:
        print('processed CSV must contain atm_id')
        return

    # latest per atm
    latest = df.sort_values(['atm_id', 'date']).groupby('atm_id').tail(1)

    # try load model
    model = None
    features = None
    if os.path.exists(MODEL) and os.path.exists(METADATA):
        try:
            model = joblib.load(MODEL)
            metadata = joblib.load(METADATA)
            features = metadata.get('features', [])
        except Exception as e:
            print('Could not load model:', e)
            model = None

    db = get_db()
    cur = db.cursor()
    try:
        cur.execute('''
            CREATE TABLE IF NOT EXISTS atm_data (
                atm_id VARCHAR(64) PRIMARY KEY,
                location VARCHAR(128),
                current_cash BIGINT,
                prediction BIGINT,
                status VARCHAR(32)
            ) ENGINE=InnoDB;
        ''')
        db.commit()

        for _, row in latest.iterrows():
            atm_id = str(row['atm_id'])
            location = atm_id
            current_cash = int(row['cash_withdrawn']) if not pd.isna(row['cash_withdrawn']) else 0
            prediction = None
            status = 'OK'

            if model and features:
                # build feature vector using some fields if present
                fv = []
                for f in features:
                    if f == 'lag_1': fv.append(row.get('lag_1', current_cash))
                    elif f == 'lag_7_avg': fv.append(row.get('lag_7_avg', current_cash))
                    elif f == 'lag_30_avg': fv.append(row.get('lag_30_avg', current_cash))
                    elif f == 'is_weekend': fv.append(int(row['date'].dayofweek >= 5))
                    elif f == 'is_holiday': fv.append(int(row.get('is_holiday', 0)))
                    elif f == 'is_admission_season': fv.append(int(row['date'].month in [7,8,9]))
                    elif f == 'month': fv.append(int(row['date'].month))
                    elif f == 'is_salary': fv.append(int(row['date'].day >= 25))
                    elif f == 'day_sin': fv.append(float(np.sin(2 * np.pi * row['date'].dayofweek / 7)))
                    elif f == 'day_cos': fv.append(float(np.cos(2 * np.pi * row['date'].dayofweek / 7)))
                    elif f == 'atm_enc': fv.append(0)
                    else: fv.append(0)
                try:
                    pred_val = model.predict([fv])[0]
                    prediction = int(round(pred_val))
                except Exception as e:
                    prediction = int(current_cash * 1.2)
            else:
                prediction = int(current_cash * 1.2)

            if prediction is None:
                prediction = int(current_cash * 1.2)
            if current_cash < prediction:
                status = 'LOW'

            # insert or replace
            cur.execute(
                "REPLACE INTO atm_data (atm_id, location, current_cash, prediction, status) VALUES (%s,%s,%s,%s,%s)",
                (atm_id, location, current_cash, prediction, status)
            )
        db.commit()
        print('Populated atm_data table with', len(latest), 'rows')
    finally:
        cur.close()
        db.close()


if __name__ == '__main__':
    main()
