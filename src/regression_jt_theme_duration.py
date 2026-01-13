import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge

import statsmodels.api as sm
from scipy import stats

from config import DATA_PROCESSED, RESULTS

IN_FILE = DATA_PROCESSED / "jt_themes_clean.parquet"

# dossier de sortie
OUT_DIR = RESULTS / "regression_jt_theme_duration"
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



def fit_ols_like_course(df, y_col, feature_cols, name, test_size=0.2):
    X = df[feature_cols].copy()
    y = df[y_col].astype(float).copy()

    cat_cols = [c for c in feature_cols if str(X[c].dtype) in ("category", "object")]
    num_cols = [c for c in feature_cols if c not in cat_cols]

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)

    X_train_sm = sm.add_constant(X_train_t.toarray(), has_constant="add")
    X_test_sm = sm.add_constant(X_test_t.toarray(), has_constant="add")

    model = sm.OLS(y_train.values, X_train_sm).fit()
    y_pred = model.predict(X_test_sm)

    r2 = r2_score(y_test.values, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test.values, y_pred))

    metrics_path = OUT_DIR / f"{name}_metrics.csv"
    pd.DataFrame([{
        "model_name": name,
        "target": y_col,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "r2_test": float(r2),
        "rmse_test": float(rmse)
    }]).to_csv(metrics_path, index=False)

    summary_path = OUT_DIR / f"{name}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(model.summary().as_text())

    resid = y_test.values - y_pred
    plot_residuals_vs_fitted(y_test.values, y_pred, OUT_DIR / f"{name}_residuals_vs_fitted.png")
    plot_qq(resid, OUT_DIR / f"{name}_qqplot.png")
    plot_obs_vs_pred(y_test.values, y_pred, OUT_DIR / f"{name}_obs_vs_pred.png")
    plot_resid_hist(resid, OUT_DIR / f"{name}_resid_hist.png")

    print(f"[OK] {name} (OLS) | R2={r2:.4f} RMSE={rmse:.2f}")
    return model


# -----------------------------
# Ridge 
# -----------------------------
def fit_ridge(df, y_col, feature_cols, name, alpha=2.0, test_size=0.2):
    X = df[feature_cols].copy()
    y = df[y_col].astype(float).copy()

    cat_cols = [c for c in feature_cols if str(X[c].dtype) in ("category", "object")]
    num_cols = [c for c in feature_cols if c not in cat_cols]

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

    pipe = Pipeline([("pre", pre), ("model", Ridge(alpha=alpha))])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    r2 = r2_score(y_test.values, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test.values, y_pred))

    metrics_path = OUT_DIR / f"{name}_metrics.csv"
    pd.DataFrame([{
        "model_name": name,
        "target": y_col,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "r2_test": float(r2),
        "rmse_test": float(rmse),
        "alpha": alpha
    }]).to_csv(metrics_path, index=False)

    resid = y_test.values - y_pred
    plot_residuals_vs_fitted(y_test.values, y_pred, OUT_DIR / f"{name}_residuals_vs_fitted.png")
    plot_qq(resid, OUT_DIR / f"{name}_qqplot.png")
    plot_obs_vs_pred(y_test.values, y_pred, OUT_DIR / f"{name}_obs_vs_pred.png")
    plot_resid_hist(resid, OUT_DIR / f"{name}_resid_hist.png")

    summary_path = OUT_DIR / f"{name}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Model: Ridge regression\n")
        f.write(f"Target: {y_col}\n")
        f.write(f"alpha: {alpha}\n\n")
        f.write(f"R2_test: {r2:.6f}\n")
        f.write(f"RMSE_test: {rmse:.6f}\n")

    print(f"[OK] {name} (Ridge) | R2={r2:.4f} RMSE={rmse:.2f}")
    return pipe


def main():
    df = pd.read_parquet(IN_FILE).dropna(subset=["duree_sec", "nb_sujets"]).copy()

    y_col = "duree_sec"
    features = ["nb_sujets", "chaine", "theme", "year", "month", "dow"]
    features = [c for c in features if c in df.columns]

    # 1) OLS 
    fit_ols_like_course(df, y_col, features, name="JT_OLS")

    # 2) Ridge 
    fit_ridge(df, y_col, features, name="JT_RIDGE", alpha=2.0)

    print(f"[OK] Tout est enregistré dans: {OUT_DIR}")


if __name__ == "__main__":
    main()
