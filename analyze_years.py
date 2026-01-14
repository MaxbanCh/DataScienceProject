import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression

# ==========================================
# 1. CHEMINS
# ==========================================
current_dir = Path(__file__).resolve().parent
file_path = current_dir / "ressources" / "20190308-years(2).csv"

if not file_path.exists():
    print(f"ERREUR : Fichier CSV introuvable : {file_path}")
    raise SystemExit(1)

graph_dir = current_dir / "graphes" / "years"
graph_dir.mkdir(exist_ok=True)
analysis_dir = current_dir / "analyses"
analysis_dir.mkdir(exist_ok=True)

# ==========================================
# 2. CHARGEMENT
# ==========================================
df = pd.read_csv(file_path)
df["is_public_channel"] = (
    df["is_public_channel"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"true": 1, "false": 0})
)

# ==========================================
# 3. GRAPHIQUES D'ANALYSE
# ==========================================
sns.set_theme(style="whitegrid")

# Distribution globale
plt.figure(figsize=(8, 5))
sns.histplot(df["women_expression_rate"], bins=30, kde=True, color="royalblue")
plt.title("Distribution du taux d'expression des femmes")
plt.xlabel("women_expression_rate")
plt.ylabel("Fréquence")
plt.savefig(graph_dir / "years_dist_women_expression_rate.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df["speech_rate"], bins=30, kde=True, color="darkorange")
plt.title("Distribution du speech_rate")
plt.xlabel("speech_rate")
plt.ylabel("Fréquence")
plt.savefig(graph_dir / "years_dist_speech_rate.png", dpi=300, bbox_inches="tight")
plt.show()

# Evolution temporelle
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="year", y="women_expression_rate", hue="media_type", marker="o")
plt.title("Évolution du taux d'expression des femmes par media_type")
plt.xlabel("Année")
plt.ylabel("women_expression_rate")
plt.savefig(graph_dir / "years_trend_women_expression_rate.png", dpi=300, bbox_inches="tight")
plt.show()

# Boxplots par groupes
plt.figure(figsize=(9, 6))
sns.boxplot(
    data=df,
    x="media_type",
    y="women_expression_rate",
    hue="media_type",
    palette="Set2",
    legend=False,
)
plt.title("women_expression_rate par media_type")
plt.xlabel("media_type")
plt.ylabel("women_expression_rate")
plt.savefig(graph_dir / "years_box_media_type.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure(figsize=(9, 6))
sns.boxplot(
    data=df,
    x="is_public_channel",
    y="women_expression_rate",
    hue="is_public_channel",
    palette="Set3",
    legend=False,
)
plt.title("women_expression_rate par is_public_channel")
plt.xlabel("is_public_channel")
plt.ylabel("women_expression_rate")
plt.savefig(graph_dir / "years_box_public_channel.png", dpi=300, bbox_inches="tight")
plt.show()

# Interaction simple (moyennes)
group_mean = (
    df.groupby(["media_type", "is_public_channel"])["women_expression_rate"]
    .mean()
    .reset_index()
)
plt.figure(figsize=(9, 6))
sns.barplot(
    data=group_mean,
    x="media_type",
    y="women_expression_rate",
    hue="is_public_channel",
    palette="viridis",
)
plt.title("Moyenne women_expression_rate par media_type et is_public_channel")
plt.xlabel("media_type")
plt.ylabel("women_expression_rate")
plt.savefig(graph_dir / "years_bar_media_public.png", dpi=300, bbox_inches="tight")
plt.show()

# ==========================================
# 4. RÉGRESSION LINÉAIRE (OLS COMPLET)
# ==========================================
X = pd.get_dummies(df[["media_type", "is_public_channel"]], drop_first=True)
X = sm.add_constant(X).astype(float)
y = pd.to_numeric(df["women_expression_rate"], errors="coerce").astype(float)

mask = X.notnull().all(axis=1) & y.notnull()
X = X.loc[mask]
y = y.loc[mask]

model = sm.OLS(y.values, X.values).fit()

with open(analysis_dir / "linear_regression_women_expression_years.txt", "w", encoding="utf-8") as f:
    f.write("=== OLS COMPLET -> women_expression_rate ~ media_type + is_public_channel ===\n")
    f.write(model.summary().as_text())
    f.write("\n")

print("✅ Graphes enregistrés dans /graphes")
print("✅ Résumé régression enregistré dans /analyses/linear_regression_women_expression_years.txt")

# ==========================================
# 5. CLASSIFICATION : is_public_channel
# ==========================================
features = ["media_type", "year", "women_expression_rate", "speech_rate", "nb_hours_analyzed"]
target = "is_public_channel"

df_clf = df[features + [target]].dropna()
X = df_clf[features]
y = df_clf[target].astype(int)

cat_cols = ["media_type"]
num_cols = [c for c in features if c not in cat_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = Pipeline(
    steps=[
        ("prep", preprocess),
        ("logreg", LogisticRegression(max_iter=1000)),
    ]
)

clf.fit(X_train, y_train)
preds = clf.predict(X_test)

report = classification_report(y_test, preds, digits=4)
cm = confusion_matrix(y_test, preds)

with open(analysis_dir / "classification_is_public_channel.txt", "w", encoding="utf-8") as f:
    f.write("=== CLASSIFICATION -> is_public_channel ===\n")
    f.write(report)
    f.write("\nConfusion matrix:\n")
    f.write(str(cm))
    f.write("\n")
