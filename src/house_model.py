"""
House Price Prediction - core modelling module.

Pipeline:
    load -> clean / impute -> feature engineering -> One-Hot encoding (pd.get_dummies)
    -> train/test split -> StandardScaler -> LinearRegression -> evaluation (R², RMSE, MAE)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "housing_data.csv"

TARGET = "price"
REFERENCE_YEAR = 2015          # latest construction year in the data -> age is measured relative to it
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Categorical variables that are One-Hot encoded with pd.get_dummies
CATEGORICAL = ["waterfront", "view"]
# Numeric variables used as-is (sqft_above is dropped: sqft_living == sqft_above + sqft_basement)
NUMERIC = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
           "condition", "sqft_basement", "house_age"]


# --------------------------------------------------------------------------- #
# Data loading & preprocessing
# --------------------------------------------------------------------------- #
def find_data_file(folder: Path | str | None = None) -> Path:
    """Locate 'Project housing data' (.csv or .xlsx) in a folder, falling back to data/housing_data.csv."""
    if folder is not None:
        folder = Path(folder)
        for pattern in ("Project housing data*.csv", "Project housing data*.xlsx"):
            matches = sorted(folder.glob(pattern))
            if matches:
                return matches[0]
    return DATA_PATH


def load_data(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path) if path else DATA_PATH
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates / invalid targets and handle missing values."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])               # a row without a price cannot be used for training
    df = df[df[TARGET] > 0]

    # Missing categorical values -> mode, missing numeric values -> median
    for col in CATEGORICAL + ["condition", "floors"]:
        if col in df and df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["house_age"] = (REFERENCE_YEAR - df["yr_built"]).clip(lower=0)
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-Hot encoding of categorical/dummy variables (drop_first avoids the dummy-variable trap)."""
    X = df[NUMERIC + CATEGORICAL].copy()
    for col in CATEGORICAL:
        X[col] = X[col].astype(int).astype("category")
    X = pd.get_dummies(X, columns=CATEGORICAL, drop_first=True, dtype=int)
    return X


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class ModelBundle:
    pipeline: Pipeline
    feature_names: list[str]
    metrics: dict
    coefficients: pd.DataFrame          # real-unit and standardized coefficients
    train_means: pd.Series
    data: pd.DataFrame                  # cleaned data (for comparisons in the app)
    y_test: np.ndarray = field(repr=False)
    y_pred_test: np.ndarray = field(repr=False)

    @property
    def intercept_at_mean(self) -> float:
        """Prediction for an 'average' house (all features at their training mean)."""
        return float(self.pipeline.named_steps["model"].intercept_)


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),   # safety net for unseen missing values
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])


def evaluate(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def train_model(path: Path | str | None = None) -> ModelBundle:
    raw = load_data(path)
    df = engineer_features(clean_data(raw))
    X = encode(df)
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    pipe = build_pipeline().fit(X_train, y_train)
    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)

    test_metrics = evaluate(y_test, y_pred_test)
    metrics = {
        "test": test_metrics,
        "train": evaluate(y_train, y_pred_train),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_rows": int(len(df)),
        "n_rows_raw": int(len(raw)),
        "missing_values_raw": int(raw.isna().sum().sum()),
        "mean_price_test": float(np.mean(y_test)),
        "mape": float(np.mean(np.abs(y_test - y_pred_test) / y_test)),
    }

    scaler: StandardScaler = pipe.named_steps["scaler"]
    lr: LinearRegression = pipe.named_steps["model"]
    coefs = pd.DataFrame({
        "feature": X.columns,
        "coef_std": lr.coef_,                          # $ change per 1 std-dev  (comparable importance)
        "coef_real": lr.coef_ / scaler.scale_,         # $ change per 1 real unit (sqft, room, year...)
        "std": scaler.scale_,
    }).assign(abs_std=lambda d: d.coef_std.abs()).sort_values("abs_std", ascending=False)

    return ModelBundle(
        pipeline=pipe,
        feature_names=list(X.columns),
        metrics=metrics,
        coefficients=coefs.reset_index(drop=True),
        train_means=pd.Series(scaler.mean_, index=X.columns),
        data=df,
        y_test=y_test,
        y_pred_test=y_pred_test,
    )


# --------------------------------------------------------------------------- #
# Prediction & explanation
# --------------------------------------------------------------------------- #
def make_input_frame(house: dict, feature_names: list[str]) -> pd.DataFrame:
    """Turn a raw house description (same columns as the CSV) into an encoded one-row frame."""
    row = pd.DataFrame([house])
    if "house_age" not in row:
        row["house_age"] = max(REFERENCE_YEAR - int(row.at[0, "yr_built"]), 0)
    X = pd.DataFrame(0, index=[0], columns=feature_names, dtype=float)
    for col in NUMERIC:
        X[col] = float(row.at[0, col])
    for col in CATEGORICAL:
        dummy = f"{col}_{int(row.at[0, col])}"
        if dummy in X.columns:
            X[dummy] = 1.0
    return X


def predict_with_explanation(bundle: ModelBundle, house: dict) -> dict:
    """
    Linear model => prediction = baseline (average house) + Σ coef_real * (x - mean).
    Each term is the exact dollar contribution of a feature relative to the average house.
    """
    X = make_input_frame(house, bundle.feature_names)
    prediction = float(bundle.pipeline.predict(X)[0])

    coef = bundle.coefficients.set_index("feature")["coef_real"]
    diff = X.iloc[0] - bundle.train_means
    contributions = (coef * diff[coef.index]).rename("contribution")

    # Group dummy columns back into their original variable
    grouped = {}
    for feat, val in contributions.items():
        key = next((c for c in CATEGORICAL if feat.startswith(f"{c}_")), feat)
        grouped[key] = grouped.get(key, 0.0) + float(val)
    contrib = pd.Series(grouped).sort_values(key=np.abs, ascending=False)

    # Market comparison: houses with similar living area (±15%)
    df = bundle.data
    sqft = float(house["sqft_living"])
    similar = df[df["sqft_living"].between(sqft * 0.85, sqft * 1.15)]
    rmse = bundle.metrics["test"]["rmse"]

    return {
        "prediction": prediction,
        "baseline": bundle.intercept_at_mean,
        "contributions": contrib,
        "price_per_sqft": prediction / sqft if sqft else np.nan,
        "market_price_per_sqft": float((df[TARGET] / df["sqft_living"]).median()),
        "similar_count": int(len(similar)),
        "similar_median": float(similar[TARGET].median()) if len(similar) else np.nan,
        "similar_p25": float(similar[TARGET].quantile(0.25)) if len(similar) else np.nan,
        "similar_p75": float(similar[TARGET].quantile(0.75)) if len(similar) else np.nan,
        "low": max(prediction - rmse, 0.0),
        "high": prediction + rmse,
    }


SAMPLE_HOUSE = {
    # A realistic, typical family home in the dataset
    "bedrooms": 3, "bathrooms": 2, "sqft_living": 1900, "sqft_lot": 7500, "floors": 1,
    "waterfront": 0, "view": 0, "condition": 3, "sqft_basement": 400, "yr_built": 1975,
}
