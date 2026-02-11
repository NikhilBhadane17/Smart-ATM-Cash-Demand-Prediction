from flask import Flask, render_template, request, redirect, session, url_for, Response, jsonify, flash
from flask_cors import CORS
from numpy import record
from werkzeug.security import generate_password_hash, check_password_hash
import csv
import random
import sys
from datetime import datetime, timedelta
try:
    from .db import get_db, execute_query, get_db_connection
except ImportError:
    from db import get_db, execute_query, get_db_connection
import mysql.connector
import os

MAX_CASH = 400000   # ATM full capacity (must match training)
LOW_CASH_THRESHOLD = 50000


base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
templates_path = os.path.join(base_dir, 'templates')
static_path = os.path.join(base_dir, 'static')

# Ensure project root is on sys.path for `model` package imports when running as script
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

# ML helpers
from model.train_model import train_and_evaluate
from model.predict import (
    predict_demand,
    predict_bulk,
    predict_latest_from_dataset,
)
from model.model_analysis import analyze_model


app = Flask(__name__, template_folder=templates_path, static_folder=static_path)
app.secret_key = "secret123"
CORS(app)

# lazy-loaded model predictor
_predict_fn = predict_demand


def get_predict_fn():
    """Return the cached prediction function if available."""
    return _predict_fn


def attach_prediction(record):
    if not record.get('location'):
        record['location'] = (
            record.get('atm_name')
            or record.get('place')
            or record.get('bank_name')
            or "Unknown Location"
        )


    prev_cash_raw = record.get('current_cash') or record.get('cash') or record.get('currentcash')

    try:
        prev_cash_raw = float(str(prev_cash_raw).replace(',', ''))
    except Exception:
        prev_cash_raw = 0.0

    # de-normalize current cash if needed
    current_cash = prev_cash_raw * MAX_CASH if prev_cash_raw <= 1 else prev_cash_raw

    predict_fn = get_predict_fn()
    try:
        pred = predict_fn(
            current_cash,
            datetime.utcnow().isoweekday(),
            is_holiday=0,
            atm_id=record.get('atm_id') or record.get('id')
        )
        predicted_cash = float(pred) * MAX_CASH
    except Exception:
        predicted_cash = current_cash * 0.9

    refill_amount = max(0, MAX_CASH - current_cash)
    status = "ALERT" if current_cash < LOW_CASH_THRESHOLD else "OK"

    record['current_cash'] = round(current_cash)
    record['prediction'] = round(predicted_cash)
    record['refill_amount'] = round(refill_amount)
    record['status'] = status


    # fallback heuristic if model not available
    if current_cash is not None:
        predicted_cash = current_cash * 0.9
        record['prediction'] = round(predicted_cash)
    else:
        record['prediction'] = 0


def load_aggregate_timeseries(window_history=7, window_future=7):
    """Return past and future aggregate cash series for charts.

    - Reads dataset/processed_atm_transactions_5yr.csv if present.
    - Aggregates cash_by_date across all ATMs.
    - Returns 8 historical points (past 7 + today) and 7 simple forecasts.
    """
    csv_path = os.path.join(base_dir, 'dataset', 'processed_atm_transactions_5yr.csv')
    if not os.path.exists(csv_path):
        return None

    daily_totals = {}
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                date_raw = (row.get('date') or '').strip()
                if not date_raw:
                    continue
                # parse date in ISO first, then fallback to common dd-mm-yyyy
                date_obj = None
                for fmt in ('%Y-%m-%d', '%d-%m-%Y', '%d/%m/%Y'):
                    if date_obj:
                        break
                    try:
                        date_obj = datetime.strptime(date_raw, fmt).date()
                    except ValueError:
                        continue
                if not date_obj:
                    try:
                        date_obj = datetime.fromisoformat(date_raw).date()
                    except ValueError:
                        continue

                cash_raw = row.get('cash_withdrawn') or row.get('cash') or row.get('withdrawal')
                try:
                    cash_val = float(str(cash_raw).replace(',', '')) if cash_raw not in (None, '') else 0.0
                except (ValueError, TypeError):
                    cash_val = 0.0

                daily_totals[date_obj] = daily_totals.get(date_obj, 0.0) + cash_val
    except (OSError, IOError, csv.Error):
        return None

    if not daily_totals:
        return None

    all_dates = sorted(daily_totals.keys())
    anchor = all_dates[-1]

    hist_dates = [anchor - timedelta(days=i) for i in range(window_history, -1, -1)]
    hist_vals = [daily_totals.get(d, 0.0) for d in hist_dates]

    recent_vals = [v for v in hist_vals if v > 0]
    baseline = sum(recent_vals) / len(recent_vals) if recent_vals else (sum(daily_totals.values()) / max(len(daily_totals), 1))

    future_dates = [anchor + timedelta(days=i) for i in range(1, window_future + 1)]
    future_vals = []
    for i in range(window_future):
        jitter = random.uniform(-0.08, 0.12)  # gentle noise
        drift = (i / max(1, window_future)) * 0.02  # light upward drift
        val = max(0.0, baseline * (1 + jitter + drift))
        future_vals.append(round(val, 2))

    dates_out = [d.isoformat() for d in hist_dates + future_dates]
    series = [round(v, 2) for v in hist_vals + future_vals]
    return {"dates": dates_out, "series": series}


def load_atm_locations_with_predictions():
    """Return ATM rows enriched with model predictions when possible."""
    csv_path = os.path.join(base_dir, 'dataset', 'final_atm_location_model_ready.csv')
    if not os.path.exists(csv_path):
        return []

    rows = []
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except (OSError, IOError, csv.Error):
        return []

    enriched = []
    for row in rows:
        atm_id = row.get('atm_id') or row.get('id') or 'UNKNOWN'
        try:
            current_cash = float(row.get('current_cash') or row.get('cash') or 0)
        except (ValueError, TypeError):
            current_cash = 0.0

        # Predict using minimal inputs; fall back to attach_prediction if it fails
        try:
           lag_1_norm = current_cash / MAX_CASH
           
           pred = predict_demand(
                lag_1=lag_1_norm,
                day_of_week=datetime.utcnow().weekday(),
                is_holiday=0,
                atm_id=atm_id,
)

        except Exception:
            pred = None

        out_row = dict(row)

        # ================= FIX MAP COORDINATES =================
        try:
            lat_norm = float(row.get('latitude') or 0)
            lon_norm = float(row.get('longitude') or 0)
        except (ValueError, TypeError):
            lat_norm = 0
            lon_norm = 0

# Shirpur area bounding box (approx)
        LAT_MIN, LAT_MAX = 21.30, 21.40
        LON_MIN, LON_MAX = 74.85, 74.95

# de-normalize (0–1 → real coordinates)
        out_row['latitude'] = LAT_MIN + lat_norm * (LAT_MAX - LAT_MIN)
        out_row['longitude'] = LON_MIN + lon_norm * (LON_MAX - LON_MIN)
# =======================================================


        # skip ATMs without valid coordinates
        if out_row['latitude'] == 0 or out_row['longitude'] == 0:
            continue

        # ensure latitude & longitude for map
        try:
            out_row['latitude'] = float(
                row.get('latitude') or row.get('lat') or 0
    )
            out_row['longitude'] = float(
                row.get('longitude') or row.get('lon') or row.get('lng') or 0
    )
        except (ValueError, TypeError):
            out_row['latitude'] = 0
            out_row['longitude'] = 0

        out_row['location'] = (
            row.get('location')
            or row.get('atm_name')
            or row.get('place')
            or row.get('bank_name')
            or "Unknown Location"
        )

        if pred is not None:
            predicted_cash = float(pred) * MAX_CASH
            out_row['prediction'] = round(predicted_cash)
            out_row['refill_amount'] = max(0, MAX_CASH - out_row['prediction'])
            out_row['status'] = "ALERT" if out_row['prediction'] < LOW_CASH_THRESHOLD else "OK"
            
            attach_prediction(out_row)
        enriched.append(out_row)

    return enriched

# Serve a tiny inline favicon to avoid 404 in logs
@app.route('/favicon.ico')
def favicon():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><rect width="100%" height="100%" fill="#2e7d32"/></svg>'
    return Response(svg, mimetype='image/svg+xml')

# ---------------- HOME ----------------
@app.route("/")
def home():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        password = request.form.get("password")

        if not name or not email or not password:
            return "All fields are required"

        db = get_db()
        try:
            existing = execute_query(
                db,
                "SELECT id FROM users WHERE email=%s",
                (email,),
                fetchone=True
            )

            if existing:
                return "Email already registered"

            hashed_password = generate_password_hash(password)

            execute_query(
                db,
                "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
                (name, email, hashed_password)
            )

            return redirect(url_for("login"))
        except mysql.connector.Error as e:
            return f"Register Error: {e}"
        finally:
            db.close()

    return render_template("register.html")

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        db = get_db()
        try:
            # First check if admin login
            admin = execute_query(
                db,
                "SELECT * FROM admin WHERE email=%s",
                (email,),
                fetchone=True
            )
            
            if admin and admin["password"] == password:
                session["admin_logged_in"] = True
                session["admin_email"] = admin["email"]
                return redirect(url_for("admin_dashboard"))
            
            # Then check regular user login
            user = execute_query(
                db,
                "SELECT * FROM users WHERE email=%s",
                (email,),
                fetchone=True
            )

            if user and check_password_hash(user["password"], password):
                session["user"] = user["name"]
                session["user_id"] = user["id"]
                return redirect(url_for("dashboard"))
            else:
                return "Invalid email or password"
        finally:
            db.close()

    return render_template("login.html")

# ---------------- ADMIN LOGIN ----------------
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        query = "SELECT * FROM admin WHERE email=%s AND password=%s"
        cursor.execute(query, (email, password))
        admin = cursor.fetchone()

        cursor.close()
        conn.close()

        if admin:
            session['admin_logged_in'] = True
            session['admin_email'] = admin['email']
            return redirect(url_for('admin_dashboard'))
        else:
            flash("Invalid Admin Email or Password", "danger")

    return render_template('admin_login.html')

# ---------------- ADMIN DASHBOARD ----------------
@app.route('/admin/dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('admin_login'))

    return render_template('admin_dashboard.html',admin_email=session.get('admin_email')
    )

# ---------------- ADMIN LOGOUT ----------------
@app.route('/admin/logout')
def admin_logout():
    session.clear()
    return redirect(url_for('admin_login'))

# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("user_dashboard.html", user=session["user"])


@app.route("/api/atms", methods=["GET"])
def api_atms():
    # look for CSV in workspace dataset folder
    csv_path = os.path.join(base_dir, 'dataset', 'atm_locations_shirpur.csv')
    if not os.path.exists(csv_path):
        # try alternate name
        csv_path = os.path.join(base_dir, 'dataset', 'atm_locations.csv')
    if not os.path.exists(csv_path):
        # no CSV available -> return empty list (client falls back to sample)
        return jsonify([])

    out = []
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # normalize keys (lowercase) to support various CSV header names
                row_l = { (k or '').strip().lower(): (v or '').strip() for k,v in row.items() }
                item = {}
                # id / atm id
                item['id'] = row_l.get('atm_id') or row_l.get('atm') or 'Unknown'
                # location/name
                item['location'] = row_l.get('location') or row_l.get('atm_name') or row_l.get('place') or row_l.get('bank_name')
                # numeric fields: distance, cash
                try:
                    if row_l.get('distance'):
                        item['distance'] = float(row_l.get('distance'))
                except (ValueError, TypeError):
                    pass
                for key in ('current_cash', 'currentcash', 'cash', 'current_cash'):
                    try:
                        if row_l.get(key):
                            item['cash'] = float(row_l.get(key))
                            break
                    except (ValueError, TypeError):
                        pass
                # latitude / longitude if present
                try:
                    lat = row_l.get('latitude') or row_l.get('lat')
                    lon = row_l.get('longitude') or row_l.get('lon') or row_l.get('lng')
                    if lat and lon:
                        item['latitude'] = float(lat)
                        item['longitude'] = float(lon)
                except (ValueError, TypeError):
                    pass
                attach_prediction(item)
                out.append(item)
    except (OSError, IOError, csv.Error):
        return jsonify([])
    return jsonify(out)


@app.route("/api/atms_public")
def api_atms_public():
    db = get_db()
    try:
        rows = execute_query(
            db,
            "SELECT atm_id, location, current_cash, prediction, status FROM atm_data",
            fetchall=True
        )
        # add small random perturbation to prediction for variability
        if isinstance(rows, list):
            for row in rows:
                try:
                    base = int(row.get('prediction', 0) or 0)
                    row['prediction'] = max(0, base + random.randint(-3000, 3000))
                except (ValueError, TypeError):
                    # leave prediction as-is on error
                    pass

        if isinstance(rows, list):
            for row in rows:
                attach_prediction(row)

        return jsonify(rows)
    finally:
        db.close()


@app.route('/api/avg_prediction')
def api_avg_prediction():
    """Return the rounded average of the `prediction` column from `atm_data`."""
    db = get_db()
    try:
        row = execute_query(
            db,
            "SELECT ROUND(AVG(prediction),0) AS avg_prediction FROM atm_data",
            fetchone=True
        )
        if not row:
            return jsonify({"avg_prediction": 0})
        if isinstance(row, dict):
            val = row.get('avg_prediction')
        else:
            val = row[0] if len(row) > 0 else 0
        try:
            val = float(val)
        except (ValueError, TypeError):
            pass
        return jsonify({"avg_prediction": val})
    finally:
        db.close()


@app.route('/api/atms_timeseries')
def api_atms_timeseries():
    """Return past 7 days + today + next 7 days aggregate cash demand."""
    data = load_aggregate_timeseries()
    if not data:
        # Fallback synthetic data to keep the chart usable even without dataset
        today = datetime.utcnow().date()
        dates = [ (today - timedelta(days=i)).isoformat() for i in range(7, -1, -1) ]
        baseline = 250000.0
        series = [baseline + random.randint(-20000, 30000) for _ in dates]
        future_dates = [ (today + timedelta(days=i)).isoformat() for i in range(1, 8) ]
        future_series = [baseline + random.randint(-25000, 35000) for _ in future_dates]
        data = {"dates": dates + future_dates, "series": series + future_series}

    return jsonify({
        "dates": data.get("dates", []),
        "aggregate": {"series": data.get("series", [])}
    })


# ---------------- MODEL: SINGLE PREDICT ----------------
@app.route('/api/model/predict', methods=['POST'])
def api_model_predict():
    payload = request.get_json(silent=True) or {}

    try:
        # 1️⃣ Read inputs
        lag_1 = float(payload.get('lag_1', 0))
        day_of_week = int(payload.get('day_of_week', datetime.utcnow().weekday()))

        # 2️⃣ Convert current cash to ₹ if normalized
        current_cash = lag_1 * MAX_CASH if lag_1 <= 1 else lag_1

        # 3️⃣ Normalize input for model (model expects 0–1)
        lag_1_norm = lag_1 if lag_1 <= 1 else lag_1 / MAX_CASH

        # 4️⃣ Predict (model output is normalized)
        prediction = predict_demand(
            lag_1=lag_1_norm,
            day_of_week=day_of_week,
            is_holiday=int(payload.get('is_holiday', 0) or 0),
            atm_id=payload.get('atm_id') or 'UNKNOWN',
            lag_7_avg=payload.get('lag_7_avg'),
            lag_30_avg=payload.get('lag_30_avg'),
            is_admission_season=int(payload.get('is_admission_season', 0) or 0),
            month=payload.get('month'),
            is_salary=payload.get('is_salary'),
        )

        # 5️⃣ Convert prediction back to ₹
        predicted_cash = float(prediction) * MAX_CASH

        # 6️⃣ Calculate refill & status
        refill_amount = max(0, MAX_CASH - current_cash)
        status = "ALERT" if current_cash < LOW_CASH_THRESHOLD else "OK"

        # 7️⃣ Response
        return jsonify({
            "current_cash": round(current_cash),
            "prediction": round(predicted_cash),
            "refill_amount": round(refill_amount),
            "status": status
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


# ---------------- MODEL: BULK PREDICT FROM DATASET ----------------
@app.route('/api/model/predict_latest')
def api_model_predict_latest():
    try:
        rows = predict_latest_from_dataset()
        return jsonify(rows)
    except Exception:
        return jsonify([])


# ---------------- MODEL: ANALYSIS METRICS ----------------
@app.route('/api/model/analysis')
def api_model_analysis():
    try:
        metrics = analyze_model()
        return jsonify(metrics)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------- MODEL: TRAIN ----------------
@app.route('/api/model/train', methods=['POST'])
def api_model_train():
    try:
        result = train_and_evaluate()
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------- MODEL: LOCATIONS + PREDICTIONS ----------------
@app.route('/api/model/atms_with_predictions')
def api_model_atms_with_predictions():
    try:
        rows = load_atm_locations_with_predictions()
        return jsonify(rows)
    except Exception:
        return jsonify([])

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
