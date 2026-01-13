import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import DATA_PROCESSED, RESULTS

IN_FILE = DATA_PROCESSED / "ina_speech_clean.parquet"

def savefig(name: str):
    out = RESULTS / name
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print("[OK] Saved", out)

def main():
    df = pd.read_parquet(IN_FILE)

    # On garde les lignes où la part est définie et où il y a un minimum de parole
    df = df.dropna(subset=["female_share", "total_speech", "channel_name", "hour", "year"]).copy()
    df = df[df["total_speech"] >= 60].copy()

    # ---------- 1) Barplot: moyenne female_share par chaîne ----------
    g = (df.groupby("channel_name")["female_share"].mean()
           .sort_values(ascending=True))
    plt.figure(figsize=(8, 12))  # figure plus haute
    plt.barh(g.index.astype(str), g.values)
    plt.xlabel("Moyenne de female_share (part de parole femmes)")
    plt.title("Part de parole des femmes par chaîne (moyenne)")
    plt.yticks(fontsize=8)  # police plus petite
    savefig("fig_ina_female_share_by_channel.png")

    # ---------- 2) Courbe: évolution annuelle female_share ----------
    gy = df.groupby("year")["female_share"].mean().sort_index()
    plt.figure()
    plt.plot(gy.index, gy.values, marker="o")
    plt.xlabel("Année")
    plt.ylabel("Moyenne female_share")
    plt.title("Évolution annuelle de la part de parole des femmes (tous médias)")
    savefig("fig_ina_female_share_over_time.png")

    # ---------- 3) Heatmap: chaîne x heure ----------
    # on prend un top de chaînes pour que ce soit lisible
    top_channels = df["channel_name"].value_counts().head(15).index
    df_h = df[df["channel_name"].isin(top_channels)].copy()

    pivot = df_h.pivot_table(
        index="channel_name", columns="hour", values="female_share", aggfunc="mean"
    ).sort_index()

    plt.figure()
    plt.imshow(pivot.values, aspect="auto")
    plt.xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns.astype(int), rotation=90)
    plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index.astype(str))
    plt.colorbar(label="Moyenne female_share")
    plt.xlabel("Heure")
    plt.ylabel("Chaîne")
    plt.title("Heatmap – part de parole des femmes (chaîne × heure)\n(top 15 chaînes, total_speech≥60s)")
    savefig("fig_ina_heatmap_channel_hour_female_share.png")

    # ---------- 4) Courbe: TV vs Radio ----------
    if "media_type" in df.columns:
        gmr = (df.groupby(["year", "media_type"])["female_share"]
                 .mean()
                 .reset_index())
        plt.figure()
        for mt in sorted(gmr["media_type"].astype(str).unique()):
            sub = gmr[gmr["media_type"].astype(str) == mt]
            plt.plot(sub["year"], sub["female_share"], marker="o", label=mt)
        plt.xlabel("Année")
        plt.ylabel("Moyenne female_share")
        plt.title("Évolution annuelle de female_share : TV vs Radio")
        plt.legend()
        savefig("fig_ina_female_share_tv_vs_radio.png")

if __name__ == "__main__":
    main()
