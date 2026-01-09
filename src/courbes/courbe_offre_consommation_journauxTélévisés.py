import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

offre = pd.read_csv("../../data/processed/StructureOffreTVParGenreProgrammes.csv")
conso = pd.read_csv("../../data/processed/StructureConsommationTVParGenreProgrammes.csv")

offre = offre.rename(columns={offre.columns[0]: "genre"})
conso = conso.rename(columns={conso.columns[0]: "genre"})

offre["genre"] = offre["genre"].str.strip()
conso["genre"] = conso["genre"].str.strip()

offre_jt = offre[offre["genre"] == "journaux télévisés"]
conso_jt = conso[conso["genre"] == "journaux télévisés"]

annees = offre_jt.columns[1:].astype(int)
offre_vals = offre_jt.iloc[0, 1:].values
conso_vals = conso_jt.iloc[0, 1:].values

plt.figure(figsize=(9,5))
plt.plot(annees, offre_vals, label="Offre (part du temps d’antenne)")
plt.plot(annees, conso_vals, label="Consommation (part du temps d’écoute)")

plt.title("Évolution de l’offre et de la consommation des journaux télévisés")
plt.xlabel("Année")
plt.ylabel("Part (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
output_path = Path(__file__).resolve().parents[2] / "results" / "courbe_offre_consommation_journauxTélévisés.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.show()
