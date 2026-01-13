import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    classification_report, confusion_matrix,
    roc_curve
)

from config import DATA_PROCESSED, RESULTS

IN_FILE = DATA_PROCESSED / "arcom2024_clean.parquet"

OUT_DIR = RESULTS / "regression_logit_high_confidence"
OUT_DIR.mkdir(parents=True, exist_ok=True)



def plot_confusion_matrix(cm, out_path):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, aspect="auto")
    plt.title("Matrice de confusion (seuil 0.5)")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_roc_curve(y_true, proba, out_path):
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Hasard")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Courbe ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_proba_hist(y_true, proba, out_path):
    plt.figure(figsize=(7, 5))
    plt.hist(proba[y_true == 0], bins=40, alpha=0.7, label="Classe réelle 0")
    plt.hist(proba[y_true == 1], bins=40, alpha=0.7, label="Classe réelle 1")
    plt.xlabel("Probabilité prédite (classe 1)")
    plt.ylabel("Effectif")
    plt.title("Distribution des probabilités prédites")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_top_coefficients(pipe, feature_names, out_path, top_n=20):
    """
    Affiche les coefficients les plus forts (positifs et négatifs).
    Attention: coefficients sur variables standardisées / one-hot.
    """
    model = pipe.named_steps["model"]
    coefs = model.coef_.ravel()

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs
    })

    top_pos = coef_df.sort_values("coef", ascending=False).head(top_n)
    top_neg = coef_df.sort_values("coef", ascending=True).head(top_n)

    top = pd.concat([top_neg, top_pos], axis=0)

    plt.figure(figsize=(8, 10))
    plt.barh(top["feature"].astype(str), top["coef"])
    plt.xlabel("Coefficient logistique")
    plt.title(f"Top coefficients (±{top_n})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def get_feature_names_from_preprocessor(pre):
    names = []

    for name, transformer, cols in pre.transformers_:
        if name == "remainder" or transformer == "drop":
            continue

        if name == "num":
            names.extend(list(cols))

        if name == "cat" and cols is not None and len(cols) > 0:
            ohe = transformer.named_steps["oh"]
            try:
                cat_names = ohe.get_feature_names_out(cols)
                names.extend(cat_names.tolist())
            except Exception:
                names.extend([str(c) for c in cols])

    return names




def main():
    print("[INFO] Logistic ARCOM start")
    print("[INFO] Reading:", IN_FILE)

    df = pd.read_parquet(IN_FILE)
    print("[INFO] Raw shape:", df.shape)

    if "CONF1_R" not in df.columns:
        raise ValueError("CONF1_R introuvable.")

    df["CONF1_R"] = pd.to_numeric(df["CONF1_R"], errors="coerce")
    df = df.dropna(subset=["CONF1_R"]).copy()
    print("[INFO] After dropna(CONF1_R):", df.shape)

    df["high_conf"] = (df["CONF1_R"] >= 4).astype(int)
    print("[INFO] high_conf distribution:\n", df["high_conf"].value_counts())

    exclude = {"CONF1_R", "high_conf"}
    candidates = [c for c in df.columns if c not in exclude]

    non_null_rate = df[candidates].notna().mean()
    candidates = [c for c in candidates if non_null_rate[c] >= 0.70]

    id_like = [c for c in candidates if "id" in c.lower() or "ident" in c.lower()]
    candidates = [c for c in candidates if c not in id_like]

    obj_cols = [c for c in candidates if df[c].dtype == "object"]
    keep_obj = []
    for c in obj_cols:
        nun = df[c].nunique(dropna=True)
        if 2 <= nun <= 50:
            keep_obj.append(c)

    num_cols = [c for c in candidates if c not in obj_cols]
    if len(num_cols) > 200:
        num_cols = num_cols[:200]

    feature_cols = num_cols + keep_obj
    print(f"[INFO] Selected features: {len(feature_cols)} (num={len(num_cols)}, obj={len(keep_obj)})")

    X = df[feature_cols].copy()
    y = df["high_conf"].astype(int)

    cat_cols = keep_obj
    num_cols_final = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler())
        ]), num_cols_final),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ])

    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs"
    )

    pipe = Pipeline([("pre", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print("[INFO] Train/Test:", X_train.shape, X_test.shape)

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred)

    metrics_path = OUT_DIR / "metrics.csv"
    pd.DataFrame([{
        "model_name": "logit_high_confidence",
        "target": "high_conf",
        "n_train": len(X_train),
        "n_test": len(X_test),
        "auc_test": float(auc),
        "accuracy_test": float(acc),
        "pos_rate_test": float(y_test.mean())
    }]).to_csv(metrics_path, index=False)

    report_path = OUT_DIR / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Matrice de confusion ===\n")
        f.write(str(cm) + "\n\n")
        f.write("=== Classification report ===\n")
        f.write(report + "\n\n")
        f.write(f"AUC = {auc:.6f}\n")
        f.write(f"Accuracy = {acc:.6f}\n")

    plot_confusion_matrix(cm, OUT_DIR / "confusion_matrix.png")
    plot_roc_curve(y_test.values, proba, OUT_DIR / "roc_curve.png")
    plot_proba_hist(y_test.values, proba, OUT_DIR / "proba_distribution.png")

    feature_names = get_feature_names_from_preprocessor(pipe.named_steps["pre"])
    plot_top_coefficients(pipe, feature_names, OUT_DIR / "top_coefficients.png", top_n=15)

    print(f"[OK] AUC={auc:.4f} ACC={acc:.4f}")
    print("[OK] Wrote:", metrics_path)
    print("[OK] Wrote:", report_path)
    print("[OK] Saved plots in:", OUT_DIR)


if __name__ == "__main__":
    main()
