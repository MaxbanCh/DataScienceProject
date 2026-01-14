import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import statsmodels.formula.api as smf

# 1) Chemins
current_dir = Path(__file__).resolve().parent
file_path = current_dir / "ressources" / "20190308-radio-years(2).csv"

if not file_path.exists():
    print(f"ERREUR : Fichier CSV introuvable : {file_path}")
    raise SystemExit(1)

graph_dir = current_dir / "graphes" / "year_par_radio"
graph_dir.mkdir(parents=True, exist_ok=True)
analysis_dir = current_dir / "analyses"
analysis_dir.mkdir(parents=True, exist_ok=True)

# 2) Chargement
df = pd.read_csv(file_path)
df.columns = [c.strip() for c in df.columns]

# Colonnes attendues
base_cols = ["year", "Médiane", "Médiane privé", "Médiane public"]
for col in base_cols:
    if col not in df.columns:
        print(f"ERREUR : Colonne manquante : {col}")
        raise SystemExit(1)

# 3) Graphes
sns.set_theme(style="whitegrid")

# Evolution annuelle (global/public/privé)
plt.figure(figsize=(10, 6))
plt.plot(df["year"], df["Médiane"], label="Médiane (global)", linewidth=2.5)
plt.plot(df["year"], df["Médiane public"], label="Médiane public", linewidth=2.5)
plt.plot(df["year"], df["Médiane privé"], label="Médiane privé", linewidth=2.5)
plt.title("Évolution annuelle du taux de parole des femmes (radio)")
plt.xlabel("Année")
plt.ylabel("Taux de parole (médiane)")
plt.legend()
plt.tight_layout()
plt.savefig(graph_dir / "radio_trend_global_public_prive.png", dpi=300)
plt.show()

# Écart public vs privé
df["ecart_public_prive"] = df["Médiane public"] - df["Médiane privé"]
plt.figure(figsize=(10, 5))
plt.plot(df["year"], df["ecart_public_prive"], color="darkred", linewidth=2.5)
plt.axhline(0, color="black", linestyle="--", alpha=0.6)
plt.title("Écart du taux de parole (public - privé) - radio")
plt.xlabel("Année")
plt.ylabel("Écart (points)")
plt.tight_layout()
plt.savefig(graph_dir / "radio_gap_public_prive.png", dpi=300)
plt.show()

# Top 10 stations sur la période
channel_cols = [c for c in df.columns if c not in base_cols and c != "ecart_public_prive"]
channel_means = df[channel_cols].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=channel_means.values, y=channel_means.index, palette="mako")
plt.title("Top 10 radios (moyenne sur la période)")
plt.xlabel("Taux de parole (moyenne)")
plt.ylabel("Radio")
plt.tight_layout()
plt.savefig(graph_dir / "radio_top10_channels.png", dpi=300)
plt.show()

# Heatmap année x radio (top 10)
top10_cols = list(channel_means.index)
df_heat = df[["year"] + top10_cols].set_index("year")
plt.figure(figsize=(12, 6))
sns.heatmap(df_heat, cmap="RdYlBu_r", linewidths=0.3, linecolor="white")
plt.title("Heatmap (Top 10 radios)")
plt.xlabel("Radio")
plt.ylabel("Année")
plt.tight_layout()
plt.savefig(graph_dir / "radio_heatmap_top10.png", dpi=300)
plt.show()

# 4) Résumé texte
summary_lines = [
    "Résumé 20190308-radio-years(2).csv",
    f"Années couvertes : {df['year'].min()} - {df['year'].max()}",
    f"Médiane globale moyenne : {df['Médiane'].mean():.2f}",
    f"Médiane public moyenne : {df['Médiane public'].mean():.2f}",
    f"Médiane privé moyenne : {df['Médiane privé'].mean():.2f}",
    f"Écart moyen (public - privé) : {df['ecart_public_prive'].mean():.2f}",
    "",
    "Top 10 radios (moyenne):",
]
summary_lines += [f"- {name}: {value:.2f}" for name, value in channel_means.items()]

with open(analysis_dir / "radio_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("✅ Graphes enregistrés dans /graphes/year_par_radio")
print("✅ Résumé enregistré dans /analyses/radio_summary.txt")

# 5) Régression (taux par radio et année)
df_long = df.melt(
    id_vars=["year"],
    value_vars=channel_cols,
    var_name="channel",
    value_name="female_rate",
).dropna()

model = smf.ols("female_rate ~ year + C(channel)", data=df_long).fit()
with open(analysis_dir / "radio_regression_female_rate.txt", "w", encoding="utf-8") as f:
    f.write("=== OLS COMPLET -> female_rate ~ year + channel ===\n")
    f.write(model.summary().as_text())
    f.write("\n")

print("✅ Régression enregistrée dans /analyses/radio_regression_female_rate.txt")
