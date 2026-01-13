import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

from config import DATA_PROCESSED, RESULTS


IN_FILE = DATA_PROCESSED / "ina_speech_clean.parquet"
OUT_DIR = RESULTS / "regression_female_share"
OUT_DIR.mkdir(exist_ok=True, parents=True)



def plot_residuals_vs_fitted(y_true, y_pred, out_path, sample=50000):
    resid = y_true - y_pred
    if len(resid) > sample:
        idx = np.random.RandomState(42).choice(len(resid), sample, replace=False)
        y_pred = y_pred[idx]
        resid = resid[idx]

    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, resid, s=6, alpha=0.4)
    plt.axhline(0)
    plt.xlabel("Fitted values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Fitted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_qq(resid, out_path, sample=50000):
    if len(resid) > sample:
        resid = np.random.RandomState(42).choice(resid, sample, replace=False)

    from scipy import stats
    plt.figure(figsize=(7, 5))
    stats.probplot(resid, dist="norm", plot=plt)
    plt.title("Normal Q-Q")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_obs_vs_pred(y_true, y_pred, out_path, sample=50000):
    if len(y_true) > sample:
        idx = np.random.RandomState(42).choice(len(y_true), sample, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=6, alpha=0.4)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title("Observed vs Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_resid_hist(resid, out_path):
    plt.figure(figsize=(7, 5))
    plt.hist(resid, bins=60)
    plt.xlabel("Residuals")
    plt.ylabel("Count")
    plt.title("Histogram of residuals")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def main():
    df = pd.read_parquet(IN_FILE)


    df = df.dropna(subset=["female_share", "total_speech"])
    df = df[df["total_speech"] >= 60].copy()

    y = df["female_share"].astype(float)


    features = [
        "channel_name", "media_type", "hour", "week_day",
        "year", "month", "civil_holyday", "school_holiday_zones",
        "is_public_channel"
    ]
    features = [c for c in features if c in df.columns]
    X = df[features].copy()

    cat_cols = [c for c in ["channel_name", "media_type", "week_day", "school_holiday_zones"] if c in X.columns]
    bool_cols = [c for c in ["is_public_channel", "civil_holyday"] if c in X.columns]
    num_cols = [c for c in X.columns if c not in set(cat_cols + bool_cols)]

    for c in cat_cols:
        X[c] = X[c].astype("category")
    for c in bool_cols:
        X[c] = X[c].astype("float64")


    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), num_cols + bool_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ])

    pipe = Pipeline([
        ("pre", pre),
        ("model", Ridge(alpha=3.0))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)


    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    metrics = pd.DataFrame([{
        "model_name": "female_share_ridge",
        "target": "female_share",
        "filter": "total_speech>=60",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "r2_test": r2,
        "rmse_test": rmse,
        "alpha": 3.0
    }])

    metrics_path = OUT_DIR / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)


    resid = y_test.values - pred

    plot_residuals_vs_fitted(
        y_test.values, pred,
        OUT_DIR / "residuals_vs_fitted.png"
    )
    plot_qq(
        resid,
        OUT_DIR / "qqplot.png"
    )
    plot_obs_vs_pred(
        y_test.values, pred,
        OUT_DIR / "obs_vs_pred.png"
    )
    plot_resid_hist(
        resid,
        OUT_DIR / "residuals_hist.png"
    )


    summary_path = OUT_DIR / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Model: Ridge regression\n")
        f.write("Target: female_share\n")
        f.write("Filter: total_speech >= 60 seconds\n\n")
        f.write(f"R2 (test): {r2:.6f}\n")
        f.write(f"RMSE (test): {rmse:.6f}\n")
        f.write("Note: Ridge regression does not provide p-values.\n")

    print(f"[OK] female_share_ridge | R2={r2:.4f} RMSE={rmse:.4f}")
    print(f"[OK] Results written to {OUT_DIR}")


if __name__ == "__main__":
    main()
