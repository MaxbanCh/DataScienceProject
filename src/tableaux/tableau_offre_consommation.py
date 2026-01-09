import pandas as pd
from pathlib import Path

offre = pd.read_csv("../../data/processed/StructureOffreTVParGenreProgrammes.csv")
conso = pd.read_csv("../../data/processed/StructureConsommationTVParGenreProgrammes.csv")

offre = offre.rename(columns={offre.columns[0]: "genre"})
conso = conso.rename(columns={conso.columns[0]: "genre"})

offre["genre"] = offre["genre"].str.strip()
conso["genre"] = conso["genre"].str.strip()

genres = [
    "journaux télévisés",
    "fictions TV",
    "films",
    "jeux",
    "variétés"
]

annees = offre.columns[1:].astype(int)

rows = []

for g in genres:
    offre_g = offre[offre["genre"] == g].iloc[0, 1:].astype(float)
    conso_g = conso[conso["genre"] == g].iloc[0, 1:].astype(float)

    rows.append({
        "Genre": g,
        "Offre moyenne (%)": offre_g.mean(),
        "Consommation moyenne (%)": conso_g.mean(),
        "Écart moyen (Conso - Offre)": conso_g.mean() - offre_g.mean()
    })

tableau = pd.DataFrame(rows)

tableau = tableau.round(2)

print(tableau)

output_path = Path(__file__).resolve().parents[2] / "results" / "tableau_offre_vs_consommation.csv"
tableau.to_csv(output_path, index=False)
