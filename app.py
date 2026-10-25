from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from train_model import CATEGORICAL_FEATURES, CLASSIFIER_NUMERIC_FEATURES, main as train_main

ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "music_model.joblib"
TRACKS_PATH = ARTIFACTS_DIR / "tracks_clean.csv"
ARTISTS_PATH = ARTIFACTS_DIR / "artists_clean.csv"
EDA_PATH = ARTIFACTS_DIR / "eda_metrics.json"

app = Flask(__name__)

TRACK_COLUMNS = [
    "track_id",
    "track_name",
    "artists",
    "album_name",
    "track_genre",
    "popularity",
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "mood",
]

MOOD_FILTERS = {
    "happy": lambda df: df[(df.valence >= 0.60) & (df.energy >= 0.55)],
    "sad": lambda df: df[(df.valence <= 0.42) & (df.energy <= 0.55)],
    "calm": lambda df: df[(df.valence >= 0.50) & (df.energy <= 0.50)],
    "energetic": lambda df: df[df.energy >= 0.72],
    "dance": lambda df: df[(df.danceability >= 0.72) & (df.energy >= 0.60)],
    "chill": lambda df: df[(df.acousticness >= 0.55) & (df.energy <= 0.55)],
    "intense": lambda df: df[(df.energy >= 0.65) & (df.valence <= 0.45)],
    "neutral": lambda df: df[df.mood == "neutral"],
}

PLAYLIST_PROFILES = {
    "workout": {
        "danceability": 0.70,
        "energy": 0.92,
        "valence": 0.62,
        "tempo": 132,
        "acousticness": 0.12,
        "instrumentalness": 0.05,
        "speechiness": 0.08,
        "popularity": 72,
    },
    "study": {
        "danceability": 0.42,
        "energy": 0.28,
        "valence": 0.48,
        "tempo": 92,
        "acousticness": 0.62,
        "instrumentalness": 0.70,
        "speechiness": 0.04,
        "popularity": 48,
    },
    "party": {
        "danceability": 0.90,
        "energy": 0.86,
        "valence": 0.72,
        "tempo": 124,
        "acousticness": 0.10,
        "instrumentalness": 0.03,
        "speechiness": 0.09,
        "popularity": 76,
    },
    "sad_evening": {
        "danceability": 0.45,
        "energy": 0.34,
        "valence": 0.18,
        "tempo": 86,
        "acousticness": 0.58,
        "instrumentalness": 0.20,
        "speechiness": 0.06,
        "popularity": 58,
    },
    "chill": {
        "danceability": 0.52,
        "energy": 0.36,
        "valence": 0.55,
        "tempo": 96,
        "acousticness": 0.76,
        "instrumentalness": 0.28,
        "speechiness": 0.05,
        "popularity": 60,
    },
    "road_trip": {
        "danceability": 0.66,
        "energy": 0.72,
        "valence": 0.68,
        "tempo": 118,
        "acousticness": 0.22,
        "instrumentalness": 0.04,
        "speechiness": 0.08,
        "popularity": 82,
    },
}

PLAYLIST_SOFT_FILTERS = {
    "workout": lambda df: df[(df.energy >= 0.58) & (df.tempo >= 105)],
    "study": lambda df: df[(df.speechiness <= 0.12) & (df.energy <= 0.62)],
    "party": lambda df: df[(df.danceability >= 0.58) & (df.energy >= 0.50)],
    "sad_evening": lambda df: df[(df.valence <= 0.52)],
    "chill": lambda df: df[(df.acousticness >= 0.38) & (df.energy <= 0.68)],
    "road_trip": lambda df: df[(df.energy >= 0.45) & (df.popularity >= 40)],
}


def ensure_artifacts() -> None:
    if not (MODEL_PATH.exists() and TRACKS_PATH.exists() and ARTISTS_PATH.exists() and EDA_PATH.exists()):
        train_main()


def load_state() -> dict[str, Any]:
    ensure_artifacts()
    artifacts = joblib.load(MODEL_PATH)
    tracks = pd.read_csv(TRACKS_PATH)
    artists = pd.read_csv(ARTISTS_PATH)
    eda = json.loads(EDA_PATH.read_text(encoding="utf-8"))
    return {"artifacts": artifacts, "tracks": tracks, "artists": artists, "eda": eda}


STATE = load_state()


def clean_number(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return round(float(value), 4)
    if pd.isna(value):
        return None
    return value


def records(df: pd.DataFrame, limit: int = 12) -> list[dict[str, Any]]:
    existing = [column for column in TRACK_COLUMNS if column in df.columns]
    out = df.head(limit)[existing].copy()
    return [{key: clean_number(value) for key, value in row.items()} for row in out.to_dict("records")]


def artist_records(df: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    columns = [
        "artist_primary",
        "tracks_count",
        "mean_popularity",
        "mean_energy",
        "mean_danceability",
        "mean_valence",
        "mean_acousticness",
        "top_genre",
    ]
    out = df.head(limit)[columns].copy()
    return [{key: clean_number(value) for key, value in row.items()} for row in out.to_dict("records")]


def apply_common_filters(df: pd.DataFrame, payload: dict[str, Any]) -> pd.DataFrame:
    result = df.copy()
    genre = payload.get("genre", "any")
    energy = payload.get("energy", "any")
    danceability = payload.get("danceability", "any")

    if genre and genre != "any":
        result = result[result["track_genre"] == genre]
    if energy == "high":
        result = result[result["energy"] >= 0.65]
    elif energy == "medium":
        result = result[(result["energy"] >= 0.40) & (result["energy"] < 0.65)]
    elif energy == "low":
        result = result[result["energy"] < 0.40]
    if danceability == "high":
        result = result[result["danceability"] >= 0.68]
    elif danceability == "medium":
        result = result[(result["danceability"] >= 0.45) & (result["danceability"] < 0.68)]
    elif danceability == "low":
        result = result[result["danceability"] < 0.45]
    return result


def sorted_tracks(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["popularity", "energy", "danceability"], ascending=[False, False, False])


def find_tracks(tracks: pd.DataFrame, query: str) -> pd.DataFrame:
    query = str(query or "").strip().lower()
    if not query:
        return tracks.iloc[0:0]

    normalized = query.replace(" — ", " - ").replace(" – ", " - ").replace(" by ", " - ")
    separators = [" - ", ",", "|"]
    parts = [normalized]
    for separator in separators:
        split_parts = []
        for part in parts:
            split_parts.extend(piece.strip() for piece in part.split(separator) if piece.strip())
        parts = split_parts or parts

    candidates = list(dict.fromkeys([query, normalized, *parts]))

    mask = pd.Series(False, index=tracks.index)
    for candidate in candidates:
        mask = mask | tracks["track_name"].str.lower().str.contains(candidate, na=False, regex=False)
        mask = mask | tracks["artists"].str.lower().str.contains(candidate, na=False, regex=False)

    if len(parts) >= 2:
        for first in parts:
            for second in parts:
                if first == second:
                    continue
                mask = mask | (
                    tracks["track_name"].str.lower().str.contains(first, na=False, regex=False)
                    & tracks["artists"].str.lower().str.contains(second, na=False, regex=False)
                )

    return tracks[mask]


def playlist_recommendations(df: pd.DataFrame, mode: str, limit: int) -> pd.DataFrame:
    artifacts = STATE["artifacts"]
    features = artifacts["track_features"]
    profile = PLAYLIST_PROFILES.get(mode, PLAYLIST_PROFILES["party"])
    soft_filter = PLAYLIST_SOFT_FILTERS.get(mode, PLAYLIST_SOFT_FILTERS["party"])

    candidates = soft_filter(df.copy())
    if len(candidates) < limit:
        candidates = df.copy()

    profile_df = pd.DataFrame([{feature: profile[feature] for feature in features}])
    track_matrix = artifacts["track_scaler"].transform(candidates[features])
    profile_vector = artifacts["track_scaler"].transform(profile_df)[0]

    profile_norm = np.linalg.norm(profile_vector)
    track_norms = np.linalg.norm(track_matrix, axis=1)
    similarity = np.divide(
        track_matrix @ profile_vector,
        track_norms * profile_norm,
        out=np.zeros(len(candidates)),
        where=(track_norms * profile_norm) != 0,
    )

    ranked = candidates.copy()
    ranked["match_score"] = np.clip((similarity + 1) / 2, 0, 1)
    ranked["playlist_score"] = ranked["match_score"] * 0.82 + (ranked["popularity"] / 100) * 0.18
    ranked = ranked.sort_values(
        ["playlist_score", "match_score", "popularity"],
        ascending=[False, False, False],
    )
    return ranked.head(limit)


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/initial")
def initial():
    tracks = STATE["tracks"]
    genres = sorted(tracks["track_genre"].dropna().unique().tolist())
    sample = tracks.sort_values("popularity", ascending=False).head(80)
    artists = STATE["artists"].head(80)
    return jsonify(
        {
            "eda": STATE["eda"],
            "genres": genres,
            "tracks": records(sample, 80),
            "artists": artist_records(artists, 80),
            "moods": list(MOOD_FILTERS.keys()),
            "playlist_modes": list(PLAYLIST_PROFILES.keys()),
        }
    )


@app.post("/api/recommend/mood")
def recommend_mood():
    payload = request.get_json(force=True)
    mood = payload.get("mood", "happy")
    limit = int(payload.get("limit", 12))
    tracks = apply_common_filters(STATE["tracks"], payload)
    filtered = MOOD_FILTERS.get(mood, MOOD_FILTERS["happy"])(tracks)
    if filtered.empty:
        filtered = MOOD_FILTERS.get(mood, MOOD_FILTERS["happy"])(STATE["tracks"])
    return jsonify({"items": records(sorted_tracks(filtered), limit)})


@app.post("/api/playlist")
def playlist():
    payload = request.get_json(force=True)
    mode = payload.get("mode", "party")
    limit = int(payload.get("limit", 20))
    tracks = apply_common_filters(STATE["tracks"], payload)
    if tracks.empty:
        tracks = STATE["tracks"]
    ranked = playlist_recommendations(tracks, mode, limit)
    return jsonify({"items": records(ranked, limit), "mode": mode, "method": "content_based_similarity"})


@app.post("/api/similar-track")
def similar_track():
    payload = request.get_json(force=True)
    query = str(payload.get("query", "")).strip().lower()
    tracks = STATE["tracks"]
    if not query:
        return jsonify({"items": []})

    matches = find_tracks(tracks, query)
    if matches.empty:
        return jsonify({"items": []})

    source_index = int(matches.sort_values("popularity", ascending=False).index[0])
    artifacts = STATE["artifacts"]
    vector = artifacts["track_scaler"].transform(tracks.loc[[source_index], artifacts["track_features"]])
    distances, indices = artifacts["track_neighbors"].kneighbors(vector, n_neighbors=16)
    similar = tracks.iloc[indices[0]].copy()
    similar = similar[similar.index != source_index]
    return jsonify({"source": records(tracks.loc[[source_index]], 1)[0], "items": records(similar, 12)})


@app.post("/api/similar-artists")
def similar_artists():
    payload = request.get_json(force=True)
    query = str(payload.get("query", "")).strip().lower()
    artists = STATE["artists"]
    if not query:
        return jsonify({"items": []})

    mask = artists["artist_primary"].str.lower().str.contains(query, na=False)
    matches = artists[mask]
    if matches.empty:
        return jsonify({"items": []})

    source_index = int(matches.sort_values("mean_popularity", ascending=False).index[0])
    artifacts = STATE["artifacts"]
    vector = artifacts["artist_scaler"].transform(artists.loc[[source_index], artifacts["artist_features"]])
    distances, indices = artifacts["artist_neighbors"].kneighbors(vector, n_neighbors=11)
    similar = artists.iloc[indices[0]].copy()
    similar = similar[similar.index != source_index]
    return jsonify({"source": artist_records(artists.loc[[source_index]], 1)[0], "items": artist_records(similar, 10)})


@app.post("/api/predict")
def predict():
    payload = request.get_json(force=True)
    query = str(payload.get("query", "")).strip().lower()
    tracks = STATE["tracks"]
    matches = find_tracks(tracks, query)
    if matches.empty:
        return jsonify({"error": "Track not found"}), 404

    row = matches.sort_values("popularity", ascending=False).iloc[[0]]
    artifacts = STATE["artifacts"]
    if "classifier_numeric_features" not in artifacts:
        return jsonify({"error": "Model artifacts are outdated. Run python train_model.py again."}), 500
    classifier_features = artifacts.get("classifier_numeric_features", CLASSIFIER_NUMERIC_FEATURES)
    categorical_features = artifacts.get("categorical_features", CATEGORICAL_FEATURES)
    x = row[classifier_features + categorical_features]
    mood = artifacts["model"].predict(x)[0]
    result = records(row, 1)[0]
    result["predicted_mood"] = mood
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
