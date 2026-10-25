from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset.csv"
ARTIFACTS_DIR = ROOT / "artifacts"

TARGET_DERIVED_FEATURES = ["danceability", "energy", "valence", "acousticness"]
NUMERIC_FEATURES = [
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]
CLASSIFIER_NUMERIC_FEATURES = [
    feature for feature in NUMERIC_FEATURES if feature not in TARGET_DERIVED_FEATURES
]
CATEGORICAL_FEATURES = ["explicit", "track_genre"]
TRACK_RECOMMENDER_FEATURES = [
    "danceability",
    "energy",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "popularity",
]
ARTIST_RECOMMENDER_FEATURES = [
    "mean_danceability",
    "mean_energy",
    "mean_valence",
    "mean_tempo",
    "mean_acousticness",
    "mean_popularity",
]


def assign_mood(row: pd.Series) -> str:
    energy = float(row["energy"])
    valence = float(row["valence"])
    danceability = float(row["danceability"])
    acousticness = float(row["acousticness"])

    if danceability >= 0.72 and energy >= 0.62:
        return "dance"
    if acousticness >= 0.65 and energy <= 0.50:
        return "chill"
    if energy >= 0.60 and valence >= 0.60:
        return "happy"
    if energy >= 0.60 and valence <= 0.40:
        return "intense"
    if energy <= 0.45 and valence >= 0.55:
        return "calm"
    if energy <= 0.45 and valence <= 0.45:
        return "sad"
    return "neutral"


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    unnamed = [column for column in df.columns if column.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    required = [
        "track_id",
        "artists",
        "album_name",
        "track_name",
        "track_genre",
        "explicit",
        *NUMERIC_FEATURES,
    ]
    df = df.dropna(subset=required)
    df = df.drop_duplicates(subset=["track_id"]).reset_index(drop=True)
    df["explicit"] = df["explicit"].astype(str)
    df["artist_primary"] = df["artists"].astype(str).str.split(";").str[0]
    df["mood"] = df.apply(assign_mood, axis=1)
    return df


def make_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_model() -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_encoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, CLASSIFIER_NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )
    classifier = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def scale_matrix(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    matrix = scaler.fit_transform(df[features])
    return matrix, scaler


def build_artist_table(df: pd.DataFrame) -> pd.DataFrame:
    artist_df = (
        df.groupby("artist_primary", as_index=False)
        .agg(
            tracks_count=("track_id", "count"),
            mean_danceability=("danceability", "mean"),
            mean_energy=("energy", "mean"),
            mean_valence=("valence", "mean"),
            mean_tempo=("tempo", "mean"),
            mean_acousticness=("acousticness", "mean"),
            mean_popularity=("popularity", "mean"),
            top_genre=("track_genre", lambda s: s.mode().iat[0] if not s.mode().empty else "unknown"),
        )
        .sort_values("mean_popularity", ascending=False)
        .reset_index(drop=True)
    )
    return artist_df[artist_df["tracks_count"] >= 2].reset_index(drop=True)


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA_PATH)
    df = clean_dataset(raw)

    train_df = df
    if len(train_df) > 30000:
        train_df = train_df.sample(n=30000, random_state=42).reset_index(drop=True)

    x = train_df[CLASSIFIER_NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = train_df["mood"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    grid = GridSearchCV(
        make_model(),
        param_grid={
            "classifier__n_estimators": [80, 120],
            "classifier__max_depth": [8, None],
            "classifier__min_samples_leaf": [1, 3],
        },
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(x_train, y_train)

    best_model = grid.best_estimator_
    predictions = best_model.predict(x_test)

    track_matrix, track_scaler = scale_matrix(df, TRACK_RECOMMENDER_FEATURES)
    track_neighbors = NearestNeighbors(n_neighbors=16, metric="cosine")
    track_neighbors.fit(track_matrix)

    artist_df = build_artist_table(df)
    artist_matrix, artist_scaler = scale_matrix(artist_df, ARTIST_RECOMMENDER_FEATURES)
    artist_neighbors = NearestNeighbors(n_neighbors=11, metric="cosine")
    artist_neighbors.fit(artist_matrix)

    eda = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "unique_tracks": int(df["track_id"].nunique()),
        "unique_artists": int(df["artist_primary"].nunique()),
        "genres": int(df["track_genre"].nunique()),
        "missing_values_before_cleaning": int(raw.isna().sum().sum()),
        "mood_distribution": {k: int(v) for k, v in df["mood"].value_counts().items()},
        "top_genres": {k: int(v) for k, v in df["track_genre"].value_counts().head(12).items()},
        "feature_means": {k: float(v) for k, v in df[TRACK_RECOMMENDER_FEATURES].mean().round(3).items()},
        "model_accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "cv_accuracy": round(float(grid.best_score_), 4),
        "best_params": grid.best_params_,
        "excluded_from_classifier": TARGET_DERIVED_FEATURES,
        "classifier_numeric_features": CLASSIFIER_NUMERIC_FEATURES,
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }

    joblib.dump(
        {
            "model": best_model,
            "classifier_numeric_features": CLASSIFIER_NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "excluded_from_classifier": TARGET_DERIVED_FEATURES,
            "track_neighbors": track_neighbors,
            "track_scaler": track_scaler,
            "track_features": TRACK_RECOMMENDER_FEATURES,
            "artist_neighbors": artist_neighbors,
            "artist_scaler": artist_scaler,
            "artist_features": ARTIST_RECOMMENDER_FEATURES,
        },
        ARTIFACTS_DIR / "music_model.joblib",
    )
    df.to_csv(ARTIFACTS_DIR / "tracks_clean.csv", index=False)
    artist_df.to_csv(ARTIFACTS_DIR / "artists_clean.csv", index=False)
    (ARTIFACTS_DIR / "eda_metrics.json").write_text(
        json.dumps(eda, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("Model trained")
    print(f"Rows: {eda['rows']}")
    print(f"Accuracy: {eda['model_accuracy']}")
    print(f"CV accuracy: {eda['cv_accuracy']}")
    print(f"Best params: {eda['best_params']}")


if __name__ == "__main__":
    main()
