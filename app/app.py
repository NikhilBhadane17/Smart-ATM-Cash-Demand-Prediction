from flask import Flask, render_template, request, redirect, session, url_for
from werkzeug.security import generate_password_hash, check_password_hash
try:
    from .db import get_db, execute_query
except ImportError:
    from db import get_db, execute_query
import mysql.connector
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
templates_path = os.path.join(base_dir, 'templates')
static_path = os.path.join(base_dir, 'static')

app = Flask(__name__, template_folder=templates_path, static_folder=static_path)
app.secret_key = "secret123"

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

# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", user=session["user"])

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
