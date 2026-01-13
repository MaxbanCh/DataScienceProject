import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error

from scipy import stats

from config import DATA_PROCESSED, RESULTS

IN_FILE = DATA_PROCESSED / "ina_speech_clean.parquet"

OUT_DIR = RESULTS / "regression_total_speech_ridge"
OUT_DIR.mkdir(parents=True, exist_ok=True)



def plot_residuals_vs_fitted(y_true, y_pred, out_path, sample=50000):
    resid = y_true - y_pred
    if len(resid) > sample:
        idx = np.random.RandomState(42).choice(len(resid), size=sample, replace=False)
        y_pred = y_pred[idx]
        resid = resid[idx]

    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, resid, s=6, alpha=0.4)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Valeurs ajustées (prédictions)")
    plt.ylabel("Résidus")
    plt.title("Résidus vs valeurs ajustées")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_qq(resid, out_path, sample=50000):
    if len(resid) > sample:
        resid = np.random.RandomState(42).choice(resid, size=sample, replace=False)

    plt.figure(figsize=(7, 5))
    stats.probplot(resid, dist="norm", plot=plt)
    plt.title("Normal Q-Q plot")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_obs_vs_pred(y_true, y_pred, out_path, sample=50000):
    if len(y_true) > sample:
        idx = np.random.RandomState(42).choice(len(y_true), size=sample, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=6, alpha=0.4)

    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx], linewidth=1)

    plt.xlabel("Observé (y)")
    plt.ylabel("Prédit (y_pred)")
    plt.title("Observé vs Prédit")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_resid_hist(resid, out_path):
    plt.figure(figsize=(7, 5))
    plt.hist(resid, bins=60)
    plt.xlabel("Résidus")
    plt.ylabel("Effectif")
    plt.title("Histogramme des résidus")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def main():
    df = pd.read_parquet(IN_FILE)
    df = df.dropna(subset=["total_speech"]).copy()

    y = df["total_speech"].astype(float)

    features = [
        "channel_name", "media_type", "hour", "week_day",
        "year", "month", "civil_holyday", "school_holiday_zones",
        "school_holiday_zone_a", "school_holiday_zone_b", "school_holiday_zone_c"
    ]
    features = [c for c in features if c in df.columns]
    X = df[features].copy()

    cat_cols = [c for c in features if str(X[c].dtype) in ("category", "object")]
    num_cols = [c for c in features if c not in cat_cols]

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ])

    alpha = 5.0
    pipe = Pipeline([("pre", pre), ("model", Ridge(alpha=alpha))])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    r2 = r2_score(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))

    metrics = pd.DataFrame([{
        "model_name": "total_speech_ridge",
        "target": "total_speech",
        "alpha": alpha,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "r2_test": float(r2),
        "rmse_test": float(rmse)
    }])

    metrics_path = OUT_DIR / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    resid = y_test.values - pred
    plot_residuals_vs_fitted(y_test.values, pred, OUT_DIR / "residuals_vs_fitted.png")
    plot_qq(resid, OUT_DIR / "qqplot.png")
    plot_obs_vs_pred(y_test.values, pred, OUT_DIR / "obs_vs_pred.png")
    plot_resid_hist(resid, OUT_DIR / "residuals_hist.png")

    summary_path = OUT_DIR / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Model: Ridge regression\n")
        f.write("Target: total_speech\n")
        f.write(f"alpha: {alpha}\n\n")
        f.write(f"R2_test: {r2:.6f}\n")
        f.write(f"RMSE_test: {rmse:.6f}\n")

    print(f"[OK] total_speech_ridge | R2={r2:.4f} RMSE={rmse:.2f}")
    print(f"[OK] Résultats enregistrés dans: {OUT_DIR}")


if __name__ == "__main__":
    main()
