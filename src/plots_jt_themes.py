import pandas as pd
import matplotlib.pyplot as plt
from config import DATA_PROCESSED, RESULTS

IN_FILE = DATA_PROCESSED / "jt_themes_clean.parquet"

def savefig(name: str):
    out = RESULTS / name
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print("[OK] Saved", out)

def main():
    df = pd.read_parquet(IN_FILE).dropna(subset=["theme", "chaine", "year", "duree_sec", "nb_sujets"])

    # ---------- 1) Top 10 thèmes (durée totale) ----------
    top = (df.groupby("theme")["duree_sec"].sum()
             .sort_values(ascending=False)
             .head(10))
    plt.figure()
    plt.barh(top.index.astype(str)[::-1], top.values[::-1])
    plt.xlabel("Durée totale (secondes)")
    plt.title("Top 10 thèmes des JT (durée cumulée, 2000–2020)")
    savefig("fig_jt_top10_themes_total_duration.png")

    # ---------- 2) Évolution annuelle des 5 thèmes principaux ----------
    top5 = top.index[:5]
    d5 = df[df["theme"].isin(top5)].copy()
    gy = (d5.groupby(["year", "theme"])["duree_sec"].sum().reset_index())

    plt.figure()
    for th in top5.astype(str):
        sub = gy[gy["theme"].astype(str) == th].sort_values("year")
        plt.plot(sub["year"], sub["duree_sec"], marker="o", label=th)
    plt.xlabel("Année")
    plt.ylabel("Durée totale annuelle (secondes)")
    plt.title("Évolution annuelle – durée totale des thèmes (Top 5)")
    plt.legend()
    savefig("fig_jt_top5_themes_over_time.png")

    # ---------- 3) Comparaison chaînes sur un thème (si trouvé) ----------
    # On essaie "Faits divers" sinon on prend le thème #1
    theme_candidates = ["Faits divers", "FAITS DIVERS", "faits divers"]
    chosen = None
    for t in theme_candidates:
        if (df["theme"].astype(str) == t).any():
            chosen = t
            break
    if chosen is None:
        chosen = str(top.index[0])

    dth = df[df["theme"].astype(str) == chosen].copy()
    gch = (dth.groupby(["year", "chaine"])["duree_sec"].sum().reset_index())

    # Pour lisibilité, on garde les 5 chaînes les plus présentes sur ce thème
    top_ch = (dth["chaine"].value_counts().head(5).index)
    gch = gch[gch["chaine"].isin(top_ch)]

    plt.figure()
    for ch in top_ch.astype(str):
        sub = gch[gch["chaine"].astype(str) == ch].sort_values("year")
        plt.plot(sub["year"], sub["duree_sec"], marker="o", label=ch)
    plt.xlabel("Année")
    plt.ylabel("Durée totale annuelle (secondes)")
    plt.title(f"Comparaison chaînes – thème '{chosen}' (Top 5 chaînes)")
    plt.legend()
    savefig("fig_jt_theme_compare_channels.png")

if __name__ == "__main__":
    main()
