import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
RESULTS = BASE_DIR / "results"

IN_FILE = DATA_PROCESSED / "jt_themes_clean.parquet"


def main():
    df = pd.read_parquet(IN_FILE)

    # On élimine les lignes inexploitables : si thème, chaîne ou durée manquent
    df = df.dropna(subset=["theme", "chaine", "duree_sec"])

    # Sélection des thèmes les plus représentés (Top 8)
    top_themes = (
        df.groupby("theme")["duree_sec"]
          .sum()
          .sort_values(ascending=False)
          .head(8)
          .index
    )

    df_ht = df[df["theme"].isin(top_themes)].copy()

    # Construction du tableau chaîne × thème
    pivot = df_ht.pivot_table(
        index="chaine",
        columns="theme",
        values="duree_sec",
        aggfunc="mean"
    )

    # Heatmap
    plt.figure(figsize=(10, 6))
    im = plt.imshow(pivot.values, aspect="auto")

    # Axe X: thèmes
    plt.xticks(
        ticks=range(len(pivot.columns)),
        labels=pivot.columns.astype(str),
        rotation=45,
        ha="right"
    )

    # Axe Y: chaines 
    plt.yticks(
        ticks=range(len(pivot.index)),
        labels=pivot.index.astype(str)
    )

    plt.colorbar(im, label="Durée moyenne par thème (secondes)")
    plt.xlabel("Thème")
    plt.ylabel("Chaîne")
    plt.title("Durée moyenne des thèmes des JT par chaîne\n(top 8 thèmes)")

    plt.tight_layout()
    plt.savefig(RESULTS / "fig_jt_theme_channel_heatmap.png", dpi=200)
    plt.close()

    print("[OK] Figure enregistrée :", RESULTS / "fig_jt_theme_channel_heatmap.png")


if __name__ == "__main__":
    main()
