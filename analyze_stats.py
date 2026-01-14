import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import statsmodels.formula.api as smf
from sklearn.metrics import confusion_matrix

# 1) Paths
current_dir = Path(__file__).resolve().parent
file_path = current_dir / "ressources" / "20190308-stats(1).csv"

if not file_path.exists():
    print(f"ERREUR : Fichier CSV introuvable : {file_path}")
    raise SystemExit(1)

graph_dir = current_dir / "graphes" / "stats"
graph_dir.mkdir(parents=True, exist_ok=True)
analysis_dir = current_dir / "analyses"
analysis_dir.mkdir(parents=True, exist_ok=True)

# 2) Load + features
df = pd.read_csv(file_path)
df["is_public_channel"] = (
    df["is_public_channel"].astype(str).str.strip().str.lower()
)
df["female_share"] = df["female_duration"] / (
    df["female_duration"] + df["male_duration"]
)
df["total_speech"] = df["female_duration"] + df["male_duration"]

df = df.replace([pd.NA, float("inf"), float("-inf")], pd.NA)
df = df.dropna(subset=["female_share", "total_speech"])

sns.set_theme(style="whitegrid")

# 3) Graphs
plt.figure(figsize=(8, 5))
sns.histplot(df["female_share"], bins=30, kde=True, color="royalblue")
plt.title("Distribution de la part de parole féminine")
plt.xlabel("female_share")
plt.ylabel("Fréquence")
plt.savefig(graph_dir / "stats_dist_female_share.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(9, 6))
sns.boxplot(
    data=df,
    x="media_type",
    y="female_share",
    hue="media_type",
    palette="Set2",
    legend=False,
)
plt.title("Part de parole féminine par media_type")
plt.xlabel("media_type")
plt.ylabel("female_share")
plt.savefig(graph_dir / "stats_box_media_type.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(9, 6))
sns.boxplot(
    data=df,
    x="is_public_channel",
    y="female_share",
    hue="is_public_channel",
    palette="Set3",
    legend=False,
)
plt.title("Part de parole féminine par is_public_channel")
plt.xlabel("is_public_channel")
plt.ylabel("female_share")
plt.savefig(graph_dir / "stats_box_public_channel.png", dpi=300, bbox_inches="tight")
plt.show()

hour_mean = (
    df.groupby(["hour", "media_type"])["female_share"]
    .mean()
    .reset_index()
)
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=hour_mean,
    x="hour",
    y="female_share",
    hue="media_type",
    marker="o",
)
plt.title("Part de parole féminine par heure")
plt.xlabel("Heure")
plt.ylabel("female_share")
plt.savefig(graph_dir / "stats_line_hour_media_type.png", dpi=300, bbox_inches="tight")
plt.show()

weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
plt.figure(figsize=(9, 6))
sns.barplot(
    data=df,
    x="week_day",
    y="female_share",
    order=[d for d in weekday_order if d in df["week_day"].unique()],
    palette="viridis",
)
plt.title("Part de parole féminine par jour de la semaine")
plt.xlabel("week_day")
plt.ylabel("female_share")
plt.savefig(graph_dir / "stats_bar_week_day.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(9, 6))
sns.boxplot(
    data=df,
    x="civil_holyday",
    y="female_share",
    hue="civil_holyday",
    palette="coolwarm",
    legend=False,
)
plt.title("Part de parole féminine : jours fériés")
plt.xlabel("civil_holyday")
plt.ylabel("female_share")
plt.savefig(graph_dir / "stats_box_civil_holyday.png", dpi=300, bbox_inches="tight")
plt.show()

if "school_holiday_zones" in df.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x="school_holiday_zones",
        y="female_share",
        palette="Set1",
    )
    plt.title("Part de parole féminine : zones de vacances scolaires")
    plt.xlabel("school_holiday_zones")
    plt.ylabel("female_share")
    plt.savefig(graph_dir / "stats_box_school_holiday_zones.png", dpi=300, bbox_inches="tight")
    plt.show()

top_channels = (
    df.groupby("channel_name")["female_share"].mean().nlargest(10).reset_index()
)
plt.figure(figsize=(10, 6))
sns.barplot(
    data=top_channels,
    y="channel_name",
    x="female_share",
    palette="mako",
)
plt.title("Top 10 chaînes par part de parole féminine")
plt.xlabel("female_share")
plt.ylabel("channel_name")
plt.savefig(graph_dir / "stats_top_channels_female_share.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="total_speech",
    y="female_share",
    alpha=0.4,
)
plt.title("Part féminine vs volume de parole")
plt.xlabel("total_speech")
plt.ylabel("female_share")
plt.savefig(graph_dir / "stats_scatter_total_speech.png", dpi=300, bbox_inches="tight")
plt.show()

print("✅ Graphes enregistrés dans /graphes/stats")

# 4) Régression multiple (female_share)
# Regroupe les chaines rares pour limiter le nombre de variables.
top_channel_names = (
    df["channel_name"].value_counts().nlargest(20).index
)
df["channel_group"] = df["channel_name"].where(
    df["channel_name"].isin(top_channel_names), "Other"
)

reg_cols = [
    "female_share",
    "hour",
    "week_day",
    "civil_holyday",
    "school_holiday_zones",
    "music_duration",
    "total_speech",
    "channel_group",
]
df_reg = df[reg_cols].dropna()

model = smf.ols(
    "female_share ~ hour + music_duration + total_speech + "
    "C(week_day) + C(civil_holyday) + C(school_holiday_zones) + C(channel_group)",
    data=df_reg,
).fit()

with open(analysis_dir / "stats_regression_female_share.txt", "w", encoding="utf-8") as f:
    f.write("=== OLS COMPLET -> female_share ===\n")
    f.write(model.summary().as_text())
    f.write("\n")

# 5) Régression logistique (parité)
df_reg["parity"] = (df_reg["female_share"] >= 0.5).astype(int)

logit_model = smf.logit(
    "parity ~ hour + music_duration + total_speech + "
    "C(week_day) + C(civil_holyday) + C(school_holiday_zones) + C(channel_group)",
    data=df_reg,
).fit(disp=False)

probs = logit_model.predict(df_reg)
threshold = 0.3
preds = (probs >= threshold).astype(int)
cm = confusion_matrix(df_reg["parity"], preds)

with open(analysis_dir / "stats_logistic_parity.txt", "w", encoding="utf-8") as f:
    f.write("=== LOGIT -> parity (female_share >= 0.5) ===\n")
    f.write(logit_model.summary().as_text())
    f.write("\n")
    f.write(f"Confusion matrix (threshold={threshold}):\n")
    f.write(str(cm))
    f.write("\n")
