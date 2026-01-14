import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.cluster import AgglomerativeClustering

from config import DATA_PROCESSED, RESULTS

IN_FILE = DATA_PROCESSED / "jt_themes_clean.parquet"
OUT_DIR = RESULTS / "jt_channel_proximity_rf"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# contruction de la matrice chaîne x thème
def build_chain_theme_matrix(df: pd.DataFrame) -> pd.DataFrame:

    # Durée totale par couple (chaine, theme)
    agg = (
        df.groupby(["chaine", "theme"], as_index=False)["duree_sec"]
          .sum()
    )

    mat = agg.pivot_table(
        index="chaine",
        columns="theme",
        values="duree_sec",
        aggfunc="sum",
        fill_value=0.0
    )

    mat_prop = mat.div(mat.sum(axis=1), axis=0).fillna(0.0)
    return mat_prop


# heatmap de la similarité cosinus entre chaînes
def plot_heatmap_similarity(sim: pd.DataFrame, out_png: Path):
    plt.figure(figsize=(10, 8))
    plt.imshow(sim.values, aspect="auto")
    plt.xticks(range(sim.shape[1]), sim.columns, rotation=45, ha="right")
    plt.yticks(range(sim.shape[0]), sim.index)
    plt.colorbar(label="Similarité cosinus (1 = très proche)")
    plt.title("Proximité entre chaînes (similarité cosinus) selon thèmes JT")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# plot de la matrice de similarité ordonnée par clustering hiérarchique
def plot_cluster_order(sim: pd.DataFrame, labels_ordered, out_png: Path):
    sim2 = sim.loc[labels_ordered, labels_ordered]
    plt.figure(figsize=(10, 8))
    plt.imshow(sim2.values, aspect="auto")
    plt.xticks(range(sim2.shape[1]), sim2.columns, rotation=45, ha="right")
    plt.yticks(range(sim2.shape[0]), sim2.index)
    plt.colorbar(label="Similarité cosinus")
    plt.title("Similarité (ordonnée par clustering hiérarchique)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    df = pd.read_parquet(IN_FILE)

    needed = {"chaine", "theme", "duree_sec"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing} (reçu: {df.columns.tolist()})")

    # Nettoyage : on enlève les lignes incomplètes et les durées non positives
    df = df.dropna(subset=["chaine", "theme", "duree_sec"]).copy()
    df = df[df["duree_sec"] > 0].copy()

    # Matrice chaîne x thème 
    mat_prop = build_chain_theme_matrix(df)

    mat_prop.to_csv(OUT_DIR / "chain_theme_profile_proportions.csv")
    print("[OK] wrote:", OUT_DIR / "chain_theme_profile_proportions.csv")

    # Similarité cosinus entre chaînes
    sim = cosine_similarity(mat_prop.values)
    sim_df = pd.DataFrame(sim, index=mat_prop.index, columns=mat_prop.index)
    sim_df.to_csv(OUT_DIR / "chain_similarity_cosine.csv")
    print("[OK] wrote:", OUT_DIR / "chain_similarity_cosine.csv")

    plot_heatmap_similarity(sim_df, OUT_DIR / "fig_chain_similarity_heatmap.png")
    print("[OK] wrote:", OUT_DIR / "fig_chain_similarity_heatmap.png")

    # Top 20 paires les plus similaires
    pairs = []
    labels = sim_df.index.tolist()
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pairs.append((labels[i], labels[j], sim_df.iloc[i, j]))
    pairs = sorted(pairs, key=lambda x: x[2], reverse=True)

    top_pairs = pd.DataFrame(pairs[:20], columns=["chaine_A", "chaine_B", "cosine_similarity"])
    top_pairs.to_csv(OUT_DIR / "top20_most_similar_pairs.csv", index=False)
    print("[OK] wrote:", OUT_DIR / "top20_most_similar_pairs.csv")

    # Clustering (regroupement de chaînes)
    dist = 1 - sim_df.values
    n_clusters = min(5, len(labels))  
    cluster = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average"
    )
    cluster_labels = cluster.fit_predict(dist)

    clust_df = pd.DataFrame({
        "chaine": labels,
        "cluster": cluster_labels
    }).sort_values(["cluster", "chaine"])
    clust_df.to_csv(OUT_DIR / "chain_clusters.csv", index=False)
    print("[OK] wrote:", OUT_DIR / "chain_clusters.csv")

    ordered = clust_df["chaine"].tolist()
    plot_cluster_order(sim_df, ordered, OUT_DIR / "fig_chain_similarity_clustered.png")
    print("[OK] wrote:", OUT_DIR / "fig_chain_similarity_clustered.png")

    # Random Forest : prédire la chaîne à partir du profil de thèmes
    if "year" in df.columns:
        df_year = df.dropna(subset=["year"]).copy()
        # profil par chaîne et année
        agg2 = df_year.groupby(["chaine", "year", "theme"], as_index=False)["duree_sec"].sum()
        mat2 = agg2.pivot_table(
            index=["chaine", "year"],
            columns="theme",
            values="duree_sec",
            aggfunc="sum",
            fill_value=0.0
        )
        X = mat2.div(mat2.sum(axis=1), axis=0).fillna(0.0)
        y = X.index.get_level_values("chaine")
        X_values = X.values
        feature_names = X.columns.tolist()
        unit_name = "chaine-year"
    else:
        print("[WARN] Pas de colonne 'year' -> RandomForest classification non pertinente (1 point par chaîne).")
        print("[INFO] Similarités + clustering OK. Ajoute 'year' dans jt_themes_clean pour la RF.")
        return


    X_train, X_test, y_train, y_test = train_test_split(
        X_values, y, test_size=0.25, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    rf.fit(X_train, y_train)

    pred = rf.predict(X_test)
    acc = accuracy_score(y_test, pred)

    # rapports et matrice de confusion
    rep = classification_report(y_test, pred, zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=sorted(pd.unique(y)))

    with open(OUT_DIR / "rf_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Unit: {unit_name}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(rep)

    pd.DataFrame(cm, index=sorted(pd.unique(y)), columns=sorted(pd.unique(y))) \
        .to_csv(OUT_DIR / "rf_confusion_matrix.csv")

    print(f"[OK] RandomForest ({unit_name}) ACC={acc:.4f}")
    print("[OK] wrote:", OUT_DIR / "rf_classification_report.txt")
    print("[OK] wrote:", OUT_DIR / "rf_confusion_matrix.csv")

    # Importance des thèmes 
    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    importances.head(50).to_csv(OUT_DIR / "rf_feature_importances_top50.csv", header=["importance"])

    top20 = importances.head(20)[::-1]  
    plt.figure(figsize=(10, 7))
    plt.barh(top20.index, top20.values)
    plt.title("Random Forest - thèmes les plus discriminants (Top 20)")
    plt.xlabel("Importance (Gini importance)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_rf_top20_feature_importances.png", dpi=200)
    plt.close()
    print("[OK] wrote:", OUT_DIR / "fig_rf_top20_feature_importances.png")


if __name__ == "__main__":
    main()
