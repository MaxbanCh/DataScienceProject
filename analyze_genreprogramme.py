import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# 1. GESTION DES CHEMINS ET DOSSIERS
# ==========================================
current_dir = Path(__file__).resolve().parent
file_path = current_dir / "ressources" / "ina-csa-parole-femmes-genreprogramme.csv"

if not file_path.exists():
    file_path = current_dir / "ressources" / "ina-csa-parole-femmes-chaines.csv"

if not file_path.exists():
    print(f"ERREUR : Fichier CSV introuvable dans {current_dir}/ressources/")
    exit()

output_dir = current_dir / "graphes"
output_dir.mkdir(exist_ok=True)

# ==========================================
# 2. CHARGEMENT ET PRÉPARATION
# ==========================================
df = pd.read_csv(file_path)

def categoriser(genre):
    genre = str(genre).lower()
    if 'information' in genre or 'journal' in genre: return 'Information'
    if 'sport' in genre: return 'Sport'
    if 'divertissement' in genre or 'jeux' in genre: return 'Divertissement'
    if 'musique' in genre or 'spectacle' in genre: return 'Culture'
    return 'Autres (Fiction/Doc)'

df['cat'] = df['genre'].apply(categoriser)

# Statistiques de taux et d'évolution
cols = ['women_expression_rate_2019', 'women_expression_rate_2020']
df_stats = df.groupby('cat')[cols].mean().reset_index()
df_stats['Evolution_Points'] = (df_stats['women_expression_rate_2020'] - df_stats['women_expression_rate_2019']) * 100

# Statistiques de volume (Durée totale convertie en heures pour 2020)
df['duree_h_2020'] = df['total_declarations_duration_2020'] / 3600
df_volume = df.groupby('cat')['duree_h_2020'].sum().reset_index()
df_stats = df_stats.merge(df_volume, on='cat')

# ==========================================
# 3. ANALYSE 1 : TAUX PAR CATÉGORIE (BARPLOT)
# ==========================================
df_plot = df_stats.melt(id_vars='cat', value_vars=cols, var_name='Année', value_name='Taux')
df_plot['Année'] = df_plot['Année'].str.extract(r'(\d+)')

plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")
ax1 = sns.barplot(data=df_plot, x='cat', y='Taux', hue='Année', palette='viridis')
plt.axhline(0.5, color='red', linestyle='--', alpha=0.6, label='Parité (0.5)')

plt.title('Répartition du temps de parole féminin (2019-2020)', fontsize=14, fontweight='bold')
plt.ylim(0, 0.6)
plt.legend(title='Année', bbox_to_anchor=(1, 1))

for p in ax1.patches:
    if p.get_height() > 0:
        ax1.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=9)

plt.savefig(output_dir / "1_comparaison_taux_genres.png", dpi=300, bbox_inches='tight')
print("✅ Graphe 1 enregistré : 1_comparaison_taux_genres.png")
plt.show()

# ==========================================
# 4. ANALYSE 2 : FOCUS SUR L'ÉVOLUTION (DOT PLOT)
# ==========================================
plt.figure(figsize=(10, 6))
df_stats_sorted = df_stats.sort_values('Evolution_Points', ascending=False)

sns.scatterplot(data=df_stats_sorted, x='Evolution_Points', y='cat', s=200, color='royalblue', edgecolor='black')
plt.axvline(0, color='black', linestyle='-', linewidth=1)

plt.title('Progression ou Régression du taux de parole (en points %)', fontsize=13)
plt.xlabel('Évolution (2020 vs 2019)')
plt.ylabel('Thématique')
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig(output_dir / "2_evolution_detaillee.png", dpi=300, bbox_inches='tight')
print("✅ Graphe 2 enregistré : 2_evolution_detaillee.png")
plt.show()

# ==========================================
# 5. ANALYSE 3 : VOLUME D'ANTENNE VS PARITÉ (SCATTER)
# ==========================================
plt.figure(figsize=(11, 7))
# La taille des bulles dépend du volume d'heures
sns.scatterplot(data=df_stats, x='duree_h_2020', y='women_expression_rate_2020', 
                size='duree_h_2020', sizes=(100, 1000), hue='cat', palette='Set1')

plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Parité')
plt.title('Lien entre Volume d\'Antenne et Taux de Parole (2020)', fontsize=13, fontweight='bold')
plt.xlabel('Volume total de parole (Heures cumulées)')
plt.ylabel('Taux d\'expression des femmes (2020)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Catégories')

# Ajout des étiquettes de texte pour chaque bulle
for i in range(df_stats.shape[0]):
    plt.text(df_stats.duree_h_2020[i], df_stats.women_expression_rate_2020[i] + 0.01, 
             df_stats.cat[i], fontsize=10, ha='center', fontweight='semibold')

plt.savefig(output_dir / "3_volume_vs_parite.png", dpi=300, bbox_inches='tight')
print("✅ Graphe 3 enregistré : 3_volume_vs_parite.png")
plt.show()

# ==========================================
# 6. RÉSUMÉ CONSOLE
# ==========================================
print("\n" + "="*65)
print(f"{'THÉMATIQUE':<20} | {'TAUX 2020':<10} | {'EVOL (pts)':<10} | {'VOLUME (h)'}")
print("-" * 65)
for _, row in df_stats.sort_values('women_expression_rate_2020', ascending=False).iterrows():
    print(f"{row['cat']:<20} | {row['women_expression_rate_2020']:.3f}     | {row['Evolution_Points']:+6.2f}    | {row['duree_h_2020']:.1f} h")