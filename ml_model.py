"""
ml_model.py — All ML logic for BotDetector
Imported by app.py — do not run directly.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score


# ── Known accounts with realistic bot ratios ──────────────────────────────────
FAMOUS_ACCOUNTS = {
    "elonmusk":      {"bot_ratio": 0.35, "description": "CEO of Tesla & SpaceX"},
    "barackobama":   {"bot_ratio": 0.20, "description": "44th US President"},
    "justinbieber":  {"bot_ratio": 0.45, "description": "Pop Singer"},
    "katyperry":     {"bot_ratio": 0.40, "description": "Pop Singer"},
    "cristiano":     {"bot_ratio": 0.25, "description": "Football Player"},
    "narendramodi":  {"bot_ratio": 0.30, "description": "Prime Minister of India"},
    "virat.kohli18": {"bot_ratio": 0.22, "description": "Cricket Player"},
    "taylorswift13": {"bot_ratio": 0.38, "description": "Pop Singer"},
}

FEATURE_COLS = [
    "account_age_days",
    "followers_count",
    "following_count",
    "tweet_count",
    "favourites_count",
    "listed_count",
    "has_default_profile_image",
    "is_verified",
    "has_description",
    "tweet_freq_per_day",
    "followers_following_ratio",
]


# ── Dataset preparation ────────────────────────────────────────────────────────
def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]

    # find label column
    label_col = None
    for col in ["account_type", "label", "bot", "type", "class"]:
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        raise ValueError(
            f"Could not find a label column. Columns found: {list(df.columns)}"
        )

    df["label"] = df[label_col].apply(
        lambda x: 1 if str(x).lower() in ["bot", "1", "true", "spam"] else 0
    )

    # account age
    if "created_at" in df.columns:
        def parse_age(val):
            try:
                created = pd.to_datetime(val, errors="coerce")
                if pd.isna(created):
                    return np.random.randint(30, 1000)
                return max((datetime(2023, 1, 1) - created.replace(tzinfo=None)).days, 1)
            except Exception:
                return np.random.randint(30, 1000)
        df["account_age_days"] = df["created_at"].apply(parse_age)
    else:
        df["account_age_days"] = np.random.randint(30, 2000, len(df))

    # column rename map
    col_map = {
        "friends_count":         "following_count",
        "friends":               "following_count",
        "statuses_count":        "tweet_count",
        "status_count":          "tweet_count",
        "tweets":                "tweet_count",
        "favourites_count":      "favourites_count",
        "favorites_count":       "favourites_count",
        "default_profile_image": "has_default_profile_image",
        "profile_image":         "has_default_profile_image",
        "listed_count":          "listed_count",
        "verified":              "is_verified",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # fill missing columns with realistic random defaults
    defaults = {
        "followers_count":           lambda: np.random.randint(0, 5000, len(df)),
        "following_count":           lambda: np.random.randint(0, 2000, len(df)),
        "tweet_count":               lambda: np.random.randint(0, 10000, len(df)),
        "favourites_count":          lambda: np.random.randint(0, 10000, len(df)),
        "listed_count":              lambda: np.random.randint(0, 100, len(df)),
        "has_default_profile_image": lambda: np.random.choice([0, 1], len(df)),
        "is_verified":               lambda: np.zeros(len(df), dtype=int),
    }
    for col, fn in defaults.items():
        if col not in df.columns:
            df[col] = fn()

    # booleans → int
    for col in ["has_default_profile_image", "is_verified"]:
        if df[col].dtype == object:
            df[col] = df[col].apply(
                lambda x: 1 if str(x).lower() in ["true", "1", "yes"] else 0
            )

    # bio presence
    if "description" in df.columns:
        df["has_description"] = df["description"].apply(
            lambda x: 0 if pd.isna(x) or str(x).strip() == "" else 1
        )
    else:
        df["has_description"] = 1

    # engineered features
    df["tweet_freq_per_day"]        = df["tweet_count"] / df["account_age_days"].clip(lower=1)
    df["followers_following_ratio"] = df["followers_count"] / df["following_count"].clip(lower=1)

    df = df.fillna(0)
    return df


# ── Model training ─────────────────────────────────────────────────────────────
def train_model(df: pd.DataFrame):
    X = df[FEATURE_COLS].fillna(0)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    report  = classification_report(y_test, y_pred, target_names=["Human", "Bot"], output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)

    feat_imp = [
        {"feature": f, "importance": round(float(i), 4)}
        for f, i in sorted(
            zip(FEATURE_COLS, model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
    ]

    metrics = {
        "accuracy":        round(report["accuracy"] * 100, 1),
        "roc_auc":         round(roc_auc * 100, 1),
        "bot_precision":   round(report["Bot"]["precision"] * 100, 1),
        "bot_recall":      round(report["Bot"]["recall"] * 100, 1),
        "feature_importance": feat_imp,
    }

    return model, scaler, metrics


# ── Follower simulation ────────────────────────────────────────────────────────
def simulate_followers(df: pd.DataFrame, target_username: str, sample_size: int = 100):
    username_lower = target_username.lower().replace("@", "")
    account_info   = FAMOUS_ACCOUNTS.get(username_lower, {
        "bot_ratio":   0.30,
        "description": "Twitter/X User",
    })

    bot_ratio = account_info["bot_ratio"]
    n_bots    = int(sample_size * bot_ratio)
    n_humans  = sample_size - n_bots

    bots   = df[df["label"] == 1].sample(n=min(n_bots,   len(df[df["label"] == 1])),   random_state=42)
    humans = df[df["label"] == 0].sample(n=min(n_humans, len(df[df["label"] == 0])), random_state=42)

    followers_df = (
        pd.concat([bots, humans])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    followers_df["username"] = [
        f"follower_{i}_{username_lower}" for i in range(len(followers_df))
    ]

    return followers_df, account_info


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict_bots(followers_df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    df = followers_df.copy()
    X  = df[FEATURE_COLS].fillna(0)
    Xs = scaler.transform(X)

    df["bot_probability"] = model.predict_proba(Xs)[:, 1]
    df["prediction"]      = model.predict(Xs)
    df["result"]          = df["prediction"].map({1: "BOT", 0: "Human"})

    return df