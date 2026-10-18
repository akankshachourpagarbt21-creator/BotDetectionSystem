"""
site: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import pandas as pd
import io
import json

from ml_model import prepare_dataset, train_model, simulate_followers, predict_bots

app = Flask(__name__)

# In-memory store — model lives here while server is running
_store = {
    "model":  None,
    "scaler": None,
    "df":     None,
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    Receives the CSV file, prepares the dataset, trains the model.
    Returns: dataset stats (row count, bot%, human%)
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename.endswith(".csv"):
        return jsonify({"error": "Please upload a .csv file"}), 400

    try:
        raw_bytes = f.read()
        df_raw = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        return jsonify({"error": f"Could not read CSV: {str(e)}"}), 400

    try:
        df = prepare_dataset(df_raw)
    except Exception as e:
        return jsonify({"error": f"Dataset preparation failed: {str(e)}"}), 400

    try:
        model, scaler, metrics = train_model(df)
    except Exception as e:
        return jsonify({"error": f"Model training failed: {str(e)}"}), 400

    _store["model"]  = model
    _store["scaler"] = scaler
    _store["df"]     = df

    total   = len(df)
    n_bots  = int(df["label"].sum())
    n_humans = total - n_bots

    return jsonify({
        "success":  True,
        "filename": f.filename,
        "total":    total,
        "n_bots":   n_bots,
        "n_humans": n_humans,
        "metrics":  metrics,          # accuracy, roc_auc, feature importances
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Receives a username, simulates followers, runs bot detection.
    Returns: per-follower predictions + summary stats.
    """
    if _store["model"] is None:
        return jsonify({"error": "Please upload and process the dataset first."}), 400

    body = request.get_json(force=True)
    username = (body.get("username") or "").strip().lstrip("@")
    if not username:
        return jsonify({"error": "Username is required"}), 400

    try:
        followers_df, account_info = simulate_followers(
            _store["df"], username, sample_size=100
        )
        results_df = predict_bots(followers_df, _store["model"], _store["scaler"])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    total    = len(results_df)
    n_bots   = int((results_df["prediction"] == 1).sum())
    n_humans = total - n_bots
    bot_pct  = round(n_bots / total * 100, 1) if total else 0

    rows = []
    for _, r in results_df.sort_values("bot_probability", ascending=False).iterrows():
        rows.append({
            "username":        r["username"],
            "bot_probability": round(float(r["bot_probability"]) * 100, 1),
            "result":          r["result"],
            "account_age_days":int(r["account_age_days"]),
            "followers_count": int(r["followers_count"]),
            "tweet_freq_per_day": round(float(r["tweet_freq_per_day"]), 1),
        })

    return jsonify({
        "username":    username,
        "description": account_info.get("description", "Twitter/X User"),
        "total":       total,
        "n_bots":      n_bots,
        "n_humans":    n_humans,
        "bot_pct":     bot_pct,
        "rows":        rows,
    })


if __name__ == "__main__":
    print("\n  BotDetector is running!")
    print("  Open your browser → http://localhost:5000\n")
    app.run(debug=True, port=5000)