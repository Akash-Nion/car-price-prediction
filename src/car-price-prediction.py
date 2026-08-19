"""
Car Price Prediction
---------------------
Cleans the UCI Automobile dataset, engineers ordinal/nominal features
correctly, trains and compares four regression models (Linear Regression,
Ridge, Random Forest, Gradient Boosting), and reports 5-fold cross-validated
R^2 plus held-out MAE/RMSE.

Run from the `src/` directory:
    python car-price-prediction.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

RANDOM_STATE = 42

WORD_TO_NUM = {
    "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "eight": 8, "twelve": 12,
}


def load_and_clean(path="../data/cars.csv"):
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0"])

    # '?' is this dataset's missing-value marker
    df = df.replace("?", np.nan)

    # normalized-losses is ~20% missing and is an insurance risk-rating
    # figure, not a vehicle spec -- drop it rather than impute a fifth
    # of a feature that only weakly relates to price.
    df = df.drop(columns=["normalized-losses"])

    # Columns that were read as strings only because of stray '?' values
    for col in ["bore", "stroke", "horsepower", "peak-rpm", "price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Can't impute the target -- drop rows missing price
    df = df.dropna(subset=["price"])

    # num-of-doors / num-of-cylinders are ordinal counts spelled out as
    # words ("four", "six"). Converting to real integers preserves that
    # ordering; the original notebook fed these into LabelEncoder as if
    # they were unordered categories, which shuffled the ordering (and
    # aliased right into the same LabelEncoder as nominal columns like
    # `make`, which doesn't have a natural order at all).
    df["num-of-doors"] = df["num-of-doors"].map(WORD_TO_NUM)
    df["num-of-cylinders"] = df["num-of-cylinders"].map(WORD_TO_NUM)

    # A handful of remaining numeric gaps (bore/stroke/horsepower/peak-rpm/
    # num-of-doors) -- fill with the column median rather than dropping
    # more rows from an already-small (204-row) dataset.
    numeric_gap_cols = ["bore", "stroke", "horsepower", "peak-rpm", "num-of-doors"]
    for col in numeric_gap_cols:
        df[col] = df[col].fillna(df[col].median())

    df = df.drop_duplicates()
    return df


def build_pipeline(categorical_cols, model):
    preprocess = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
        remainder="passthrough",
    )
    return Pipeline([("prep", preprocess), ("model", model)])


def main():
    df = load_and_clean()
    print("Clean shape:", df.shape)

    y_raw = df["price"].values
    # .copy() so the feature frame can never alias back onto anything
    # that gets the target column added to it later (the bug in the
    # original notebook: `DX = x; DX['price'] = y` silently added
    # `price` back into `x` too, because `DX = x` is a reference, not a
    # copy -- the model was then trained with the target as a feature,
    # producing a meaningless R2 of 1.0).
    X = df.drop(columns=["price"]).copy()

    # Only genuinely unordered categories get one-hot encoded.
    # num-of-doors / num-of-cylinders are already numeric at this point.
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    y = np.log1p(y_raw)  # price is right-skewed

    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
        X, y, y_raw, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for name, model in models.items():
        pipe = build_pipeline(categorical_cols, model)
        pipe.fit(X_train, y_train)

        pred_price = np.expm1(pipe.predict(X_test))
        r2 = r2_score(y_test_raw, pred_price)
        mae = mean_absolute_error(y_test_raw, pred_price)
        rmse = np.sqrt(mean_squared_error(y_test_raw, pred_price))
        cv_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")

        results.append(
            {
                "model": name,
                "test_r2": round(r2, 4),
                "test_mae": round(mae, 0),
                "test_rmse": round(rmse, 0),
                "cv_r2_mean": round(cv_scores.mean(), 4),
                "cv_r2_std": round(cv_scores.std(), 4),
            }
        )
        print(
            f"{name:20s} test_R2={r2:.4f}  MAE={mae:,.0f}  RMSE={rmse:,.0f}  "
            f"CV_R2={cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
        )

    results_df = pd.DataFrame(results).sort_values("cv_r2_mean", ascending=False)
    print("\n=== Summary (sorted by CV R2) ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
