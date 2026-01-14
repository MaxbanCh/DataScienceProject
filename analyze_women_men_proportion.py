import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from pathlib import Path


# 1. GESTION DES CHEMINS ET DOSSIERS

current_dir = Path(__file__).resolve().parent
data_dir = current_dir / "ressources" / "WomenMenProportion"
output_dir = current_dir / "graphes" / "women_men_proportion"
output_dir.mkdir(exist_ok=True)
analysis_dir = current_dir / "analyses"
analysis_dir.mkdir(exist_ok=True)

files = sorted(data_dir.glob("women_men_proportion_*.csv"))
if not files:
    print(f"ERREUR : Aucun fichier trouvé dans {data_dir}")
    raise SystemExit(1)


# 2. CHARGEMENT ET PREPARATION

df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["percent"] = (
    df["percent"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .str.replace(",", ".", regex=False)
)
df["percent"] = pd.to_numeric(df["percent"], errors="coerce") / 100.0

df = df[df["type"].isin(["Femme", "Homme"])].copy()
df = df.dropna(subset=["date", "percent"])

df_pivot = (
    df.pivot_table(
        index=["date", "channel"],
        columns="type",
        values="percent",
        aggfunc="mean",
    )
    .reset_index()
)
df_pivot = df_pivot.dropna(subset=["channel"])

df_pivot["month"] = df_pivot["date"].dt.to_period("M").dt.to_timestamp()
df_pivot["year"] = df_pivot["date"].dt.year
df_pivot["month_num"] = df_pivot["date"].dt.month
df_month = df_pivot.groupby("month")[["Femme", "Homme"]].mean().reset_index()

df_year = (
    df_pivot.groupby(df_pivot["date"].dt.year)[["Femme", "Homme"]]
    .mean()
    .reset_index()
    .rename(columns={"date": "year"})
)

top_channels = df_pivot["channel"].value_counts().head(6).index
df_top = df_pivot[df_pivot["channel"].isin(top_channels)]
df_top_month = (
    df_top.groupby(["month", "channel"])["Femme"]
    .mean()
    .reset_index()
)

sns.set_theme(style="whitegrid")


# 3. EVOLUTION MENSUELLE GLOBALe (FEMME/HOMME)

plt.figure(figsize=(12, 6))
plt.plot(df_month["month"], df_month["Femme"], label="Femme", linewidth=2.5)
plt.plot(df_month["month"], df_month["Homme"], label="Homme", linewidth=2.5)
plt.axhline(0.5, color="red", linestyle="--", alpha=0.6, label="Parite (0.5)")
plt.title("Evolution mensuelle du temps de parole (tous canaux)")
plt.xlabel("Mois")
plt.ylabel("Proportion")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "women_men_1_trend_monthly.png", dpi=300)
print("✅ Graphe 1 enregistre : women_men_1_trend_monthly.png")
plt.show()


# 4. FEMMES PAR CHAINE (TOP 6) - MENSUEL

plt.figure(figsize=(12, 6))
sns.lineplot(
    data=df_top_month,
    x="month",
    y="Femme",
    hue="channel",
    linewidth=2,
)
plt.axhline(0.5, color="red", linestyle="--", alpha=0.4, label="Parite (0.5)")
plt.title("Evolution mensuelle du temps de parole feminin (Top 6 chaines)")
plt.xlabel("Mois")
plt.ylabel("Proportion Femme")
plt.ylim(0, 1)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(output_dir / "women_men_2_top_channels.png", dpi=300, bbox_inches="tight")
print("✅ Graphe 2 enregistre : women_men_2_top_channels.png")
plt.show()

# ==========================================
# 5. MOYENNE ANNUELLE (FEMMES)
# ==========================================
plt.figure(figsize=(10, 5))
ax_year = sns.barplot(data=df_year, x="year", y="Femme", color="steelblue")
plt.axhline(0.5, color="red", linestyle="--", alpha=0.6, label="Parite (0.5)")
plt.title("Moyenne annuelle du temps de parole feminin")
plt.xlabel("Annee")
plt.ylabel("Proportion Femme")
plt.ylim(0, 1)
plt.legend()
for p in ax_year.patches:
    height = p.get_height()
    if height > 0:
        ax_year.annotate(
            f"{height:.2f}",
            (p.get_x() + p.get_width() / 2, height),
            ha="center",
            va="center",
            xytext=(0, -10),
            textcoords="offset points",
            color="white",
            fontsize=9,
            fontweight="bold",
        )
plt.tight_layout()
plt.savefig(output_dir / "women_men_3_yearly_female.png", dpi=300)
print("✅ Graphe 3 enregistre : women_men_3_yearly_female.png")
plt.show()

# ==========================================
# 6. HEATMAP ANNEE x MOIS (FEMMES)
# ==========================================
df_heat = (
    df_pivot.groupby(["year", "month_num"])["Femme"]
    .mean()
    .reset_index()
    .pivot(index="year", columns="month_num", values="Femme")
)

plt.figure(figsize=(12, 6))
norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
sns.heatmap(
    df_heat,
    cmap="RdYlBu_r",
    norm=norm,
    linewidths=0.3,
    linecolor="white",
    cbar_kws={"label": "Proportion Femme"},
)
plt.title("Evolution mensuelle (toutes annees) - proportion Femme")
plt.xlabel("Mois")
plt.ylabel("Annee")
plt.tight_layout()
plt.savefig(output_dir / "women_men_4_heatmap_year_month.png", dpi=300)
print("✅ Graphe 4 enregistre : women_men_4_heatmap_year_month.png")
plt.show()

# ==========================================
# 6. RESUME CONSOLE
# ==========================================
last_month = df_month.iloc[-1]
min_f = df_month.loc[df_month["Femme"].idxmin()]
max_f = df_month.loc[df_month["Femme"].idxmax()]

print("\n" + "=" * 65)
print("RESUME GLOBAL (tous canaux)")
print(f"Dernier mois : {last_month['month'].date()} | Femme: {last_month['Femme']:.2%}")
print(f"Min Femme    : {min_f['month'].date()} | {min_f['Femme']:.2%}")
print(f"Max Femme    : {max_f['month'].date()} | {max_f['Femme']:.2%}")

# ==========================================
# 7. CORRELATIONS ET REGRESSION
# ==========================================
print("\n" + "=" * 65)
print("CORRELATIONS (tous canaux, mensuel)")
df_corr = df_month.copy()
df_corr["year"] = df_corr["month"].dt.year
df_corr["month_num"] = df_corr["month"].dt.month

corr_year = df_corr["Femme"].corr(df_corr["year"])
corr_month = df_corr["Femme"].corr(df_corr["month_num"])
print(f"Corr(Femme, annee) : {corr_year:.3f}")
print(f"Corr(Femme, mois)  : {corr_month:.3f}")

report_path = analysis_dir / "women_men_correlations.txt"
report_lines = [
    "CORRELATIONS (tous canaux, mensuel)",
    f"Corr(Femme, annee) : {corr_year:.3f}",
    f"Corr(Femme, mois)  : {corr_month:.3f}",
    "",
]

print("\nREGRESSION LINEAIRE (Femme ~ annee + mois + chaines)")
df_reg = df_pivot.copy()
df_reg = df_reg.dropna(subset=["Femme", "year", "month_num", "channel"])

try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    model = smf.ols("Femme ~ year + C(month_num) + C(channel)", data=df_reg).fit()
    print(model.summary().tables[1])
    print(f"R2 ajusté : {model.rsquared_adj:.3f}")
    report_lines.append("REGRESSION LINEAIRE (statsmodels)")
    report_lines.append(model.summary().as_text())
    report_lines.append(f"R2 ajuste : {model.rsquared_adj:.3f}")
except Exception as exc:
    try:
        from sklearn.linear_model import LinearRegression

        X = pd.get_dummies(df_reg[["year", "month_num", "channel"]], drop_first=True)
        y = df_reg["Femme"].values
        reg = LinearRegression().fit(X, y)
        r2 = reg.score(X, y)
        print("statsmodels non disponible, fallback sklearn")
        print(f"R2 : {r2:.3f}")
        report_lines.append("REGRESSION LINEAIRE (sklearn)")
        report_lines.append(f"R2 : {r2:.3f}")
    except Exception as exc2:
        print("statsmodels et sklearn non disponibles")
        print(f"Erreur : {exc}")
        print(f"Erreur sklearn : {exc2}")
        report_lines.append("REGRESSION LINEAIRE")
        report_lines.append(f"Erreur : {exc}")
        report_lines.append(f"Erreur sklearn : {exc2}")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"✅ Rapport correlations/regression enregistre : {report_path.name}")

# ==========================================
# 8. REGRESSION LOGISTIQUE (GLM BINOMIAL)
# ==========================================
logit_report = analysis_dir / "women_men_logistic_regression.txt"
logit_lines = [
    "REGRESSION LOGISTIQUE (GLM binomial sur proportion Femme)",
    "Note: la cible est une proportion (0-1), pas un binaire.",
    "Le modele est une approximation logit sur proportion.",
    "",
]

try:
    import statsmodels.formula.api as smf

    df_log = df_reg.copy()
    df_log["Femme_clip"] = df_log["Femme"].clip(1e-6, 1 - 1e-6)
    model_log = smf.glm(
        "Femme_clip ~ year + C(month_num) + C(channel)",
        data=df_log,
        family=sm.families.Binomial(),
    ).fit()
    logit_lines.append(model_log.summary().as_text())
    logit_lines.append(f"AIC : {model_log.aic:.3f}")
    logit_lines.append(f"BIC : {model_log.bic:.3f}")
    print("✅ Regression logistique calculee (statsmodels)")
except Exception as exc:
    logit_lines.append("statsmodels non disponible, regression logistique non calculee.")
    logit_lines.append(f"Erreur : {exc}")
    print("statsmodels non disponible, regression logistique non calculee.")

with open(logit_report, "w", encoding="utf-8") as f:
    f.write("\n".join(logit_lines))
print(f"✅ Rapport regression logistique enregistre : {logit_report.name}")
