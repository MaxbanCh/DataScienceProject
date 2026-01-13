import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

offre = pd.read_csv("../../data/processed/StructureOffreTVParGenreProgrammes.csv")

offre = offre.rename(columns={offre.columns[0]: "genre"})

offre["genre"] = offre["genre"].str.strip()

info_genre = "journaux télévisés"
divertissement_genres = ["fictions TV", "jeux", "variétés", "films"]

info = offre[offre["genre"] == info_genre].iloc[0, 1:].astype(float)

divertissement = (
    offre[offre["genre"].isin(divertissement_genres)]
    .iloc[:, 1:]
    .astype(float)
    .sum()
)

annees = offre.columns[1:].astype(int)

indice = divertissement / info

plt.figure(figsize=(9,5))
plt.plot(annees, indice, linewidth=2)

plt.title("Indice divertissement / information dans l’offre télévisuelle")
plt.xlabel("Année")
plt.ylabel("Indice (Divertissement / Information)")
plt.grid(True)
plt.tight_layout()

# Sauvegarde
output_path = Path(__file__).resolve().parents[2] / "results" / "courbe_comparaison_informationJT_divertissement.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")

plt.show()
