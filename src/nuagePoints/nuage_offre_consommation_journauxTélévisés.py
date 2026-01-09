import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

offre = pd.read_csv("../../data/processed/StructureOffreTVParGenreProgrammes.csv")
conso = pd.read_csv("../../data/processed/StructureConsommationTVParGenreProgrammes.csv")

offre = offre.rename(columns={offre.columns[0]: "genre"})
conso = conso.rename(columns={conso.columns[0]: "genre"})

offre["genre"] = offre["genre"].str.strip()
conso["genre"] = conso["genre"].str.strip()

genre = "journaux télévisés"

offre_jt = offre[offre["genre"] == genre].iloc[0, 1:].astype(float)
conso_jt = conso[conso["genre"] == genre].iloc[0, 1:].astype(float)

X = offre_jt.values
Y = conso_jt.values

a, b = np.polyfit(X, Y, 1)

x_reg = np.linspace(X.min(), X.max(), 100)
y_reg = a * x_reg + b

plt.figure(figsize=(7,6))
plt.scatter(X, Y, label="Observations")
plt.plot(x_reg, y_reg, linestyle="--", linewidth=2,
         label=f"Régression : y = {a:.2f}x + {b:.2f}")

plt.title("Offre vs Consommation des journaux télévisés")
plt.xlabel("Offre (part du temps d’antenne, %)")
plt.ylabel("Consommation (part du temps d’écoute, %)")
plt.grid(True)
plt.legend()
plt.tight_layout()

output_path = Path(__file__).resolve().parents[2] / "results" / "nuage_offre_consommation_journauxTélévisés.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")

plt.show()
