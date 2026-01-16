import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuration du style matplotlib
sns.set_theme(style="white")
plt.rcParams['font.family'] = 'sans-serif'


def get_pure_linear(row):
    """Fonction de correction pour la TV linéaire pure"""
    if row['annee'] >= 2016:
        correction = (row['taux_replay_pourcent'] / 100) * 0.15
        return row['duree_tv_globale_min'] * (1 - correction)
    return row['duree_tv_globale_min']


@st.cache_data
def load_audience_data():
    """Charge les données d'audience"""
    try:
        df = pd.read_csv('data/processed/master_file_annuel.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Fichier 'data/processed/master_file_annuel.csv' introuvable!")
        return None


def show_vad_impact_chart(df):
    """Graphique: Impact VaD sur TV traditionnelle"""
    st.subheader("L'impact de la Vidéo à la Demande sur la consommation TV traditionnelle")

    # Préparation des données
    df['duree_tv_lineaire_pure'] = df.apply(get_pure_linear, axis=1)
    df_subst = df[df['annee'] >= 2010].copy()

    # Création du graphique
    fig, ax1 = plt.subplots(figsize=(13, 7))

    color1 = '#2c3e50'
    color2 = '#e74c3c'

    # Courbe 1: Durée TV Linéaire Pure
    sns.lineplot(data=df_subst, x='annee', y='duree_tv_lineaire_pure',
                 ax=ax1, color=color1, linewidth=3, marker='o', label='Durée TV Linéaire Pure')

    ax1.set_ylabel("Durée d'écoute quotidienne (min/jour)", fontsize=12, fontweight='bold', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xlabel('Année', fontsize=12)

    # Courbe 2: CA VaD
    ax2 = ax1.twinx()
    sns.lineplot(data=df_subst, x='annee', y='ca_vad_millions',
                 ax=ax2, color=color2, linewidth=3, marker='s', label='CA VaD (M€)')

    ax2.set_ylabel("Chiffre d'Affaires VaD (Millions €)", fontsize=12, fontweight='bold', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    years = df_subst['annee'].astype(int).unique()
    ax1.set_xticks(np.arange(min(years), max(years) + 1, 1))

    plt.title("L'impact de la Vidéo à la Demande sur la consommation TV traditionnelle (Données Corrigées)",
              fontsize=16, fontweight='bold', pad=20)

    ax1.grid(True, axis='y', linestyle=':', alpha=0.7)
    sns.despine(right=False)
    ax1.get_legend().remove()
    ax2.get_legend().remove()

    plt.tight_layout()
    st.pyplot(fig)

    st.info(
        "📊 Ce graphique montre la relation inverse entre l'essor de la VaD et la consommation de TV traditionnelle.")


def show_regression_chart(df):
    """Graphique: Régression VaD"""
    st.subheader("Régression : Impact de l'essor de la VaD sur le temps d'écoute TV")

    # Préparation des données
    df['duree_tv_lineaire_pure'] = df.apply(get_pure_linear, axis=1)
    df_plot = df.dropna(subset=['duree_tv_lineaire_pure', 'ca_vad_millions']).copy()

    # Création du graphique
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.regplot(data=df_plot, x='ca_vad_millions', y='duree_tv_lineaire_pure',
                scatter=True, color='blue', label='Tendance (Régression)', ax=ax)

    plt.title("Régression : Impact de l'essor de la VaD sur le temps d'écoute TV", fontsize=14)
    plt.xlabel("Chiffre d'Affaires de la VaD (en Millions €)", fontsize=12)
    plt.ylabel("Durée TV Linéaire Pure (minutes/jour)", fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    st.success(
        "📉 La régression linéaire montre une corrélation négative entre le CA de la VaD et le temps d'écoute TV.")


def show_age_evolution_chart(df):
    """Graphique: Évolution par tranche d'âge"""
    st.subheader("Évolution de la durée d'écoute quotidienne par tranche d'âge")

    # Préparation des données
    df['annee'] = df['annee'].astype(int)
    df_periode = df[(df['annee'] >= 2009) & (df['annee'] <= 2023)].copy()

    # Création du graphique
    fig, ax = plt.subplots(figsize=(13, 7))

    color_seniors = '#e67e22'
    color_moyenne = '#7f8c8d'
    color_actifs = '#2c3e50'

    sns.lineplot(data=df_periode, x='annee', y='duree_tv_50_plus_min',
                 ax=ax, color=color_seniors, linewidth=3, marker='s', label='50+ ans')

    sns.lineplot(data=df_periode, x='annee', y='duree_tv_globale_min',
                 ax=ax, color=color_moyenne, linewidth=2, linestyle='--', marker='x',
                 alpha=0.6, label='Moyenne Nationale')

    sns.lineplot(data=df_periode, x='annee', y='duree_tv_15_49_min',
                 ax=ax, color=color_actifs, linewidth=3, marker='o', label='15-49 ans')

    years = df_periode['annee'].unique()
    ax.set_xticks(np.arange(min(years), max(years) + 1, 1))

    plt.title("Évolution de la durée d'écoute quotidienne de la télévision par tranche d'âge (2009-2023)",
              fontsize=16, fontweight='bold', pad=20)

    ax.set_ylabel("Durée d'écoute quotidienne par personne (en minute)", fontsize=12, fontweight='bold')
    ax.set_xlabel('Année', fontsize=12)
    ax.grid(True, axis='y', linestyle=':', alpha=0.7)
    sns.despine()
    ax.set_ylim(100, 400)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=True)

    plt.tight_layout()
    st.pyplot(fig)

    st.warning(
        "👥 Les seniors (50+) restent fidèles à la TV, tandis que les actifs (15-49 ans) décrochent progressivement.")


def show_info_thematic_chart(df):
    """Graphique: Info vs Thématiques"""
    st.subheader("Évolution croisée de l'offre d'information et de l'audience thématique")

    # Préparation des données
    df_plot = df.dropna(subset=['offre_info_pourcent', 'pda_thematiques']).copy()
    df_plot['annee'] = df_plot['annee'].astype(int)

    # Création du graphique
    fig, ax1 = plt.subplots(figsize=(13, 7))

    color_info = '#2c3e50'
    color_them = '#e74c3c'

    ax1.plot(df_plot['annee'], df_plot['offre_info_pourcent'],
             color=color_info, marker='o', linewidth=3, label="Part de l'Information (%)")
    ax1.set_xlabel('Année', fontsize=12)
    ax1.set_ylabel('Offre Information (%)', color=color_info, fontweight='bold', fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color_info)

    ax2 = ax1.twinx()
    ax2.plot(df_plot['annee'], df_plot['pda_thematiques'],
             color=color_them, marker='s', linewidth=3, label="PDA Thématiques (%)")
    ax2.set_ylabel('Audience Thématique (%)', color=color_them, fontweight='bold', fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color_them)

    plt.title("Évolution croisée de l'offre d'information et de la part d'audience des chaînes thématiques (2007-2024)",
              fontsize=16, fontweight='bold', pad=20)
    ax1.grid(True, axis='y', linestyle=':', alpha=0.7)
    ax1.set_xticks(np.arange(df_plot['annee'].min(), df_plot['annee'].max() + 1, 2))
    sns.despine(right=False)

    plt.tight_layout()
    st.pyplot(fig)

    st.info("📰 L'augmentation de l'offre d'information coïncide avec la montée des chaînes thématiques.")


def show_clustering_chart(df):
    """Graphique: Clustering K-Means"""
    st.subheader("Clustering K-Means : Rupture entre l'offre Info et l'Audience Thématique")

    # Préparation des données
    df_ml = df.dropna(subset=['offre_info_pourcent', 'pda_thematiques']).copy()
    X = df_ml[['offre_info_pourcent', 'pda_thematiques']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df_ml['cluster'] = kmeans.fit_predict(X_scaled)

    noms_clusters = {
        0: "Ancien Modèle (Généraliste)",
        1: "Nouveau Modèle (Info & Thématiques)"
    }

    df_ml['Nom_Era'] = df_ml['cluster'].map(noms_clusters)

    # Création du graphique
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.scatterplot(
        data=df_ml,
        x='offre_info_pourcent',
        y='pda_thematiques',
        hue='Nom_Era',
        palette=['#2ecc71', '#e74c3c'],
        s=150,
        style='Nom_Era',
        ax=ax
    )

    for i in range(df_ml.shape[0]):
        plt.text(df_ml.offre_info_pourcent.iloc[i] + 0.2, df_ml.pda_thematiques.iloc[i],
                 int(df_ml.annee.iloc[i]), fontsize=10, alpha=0.8)

    plt.legend(title="Classification des Ères Médiatiques", title_fontsize='13', fontsize='11')
    plt.title("Clustering K-Means : Rupture entre l'offre Info et l'Audience Thématique", fontsize=15)
    plt.xlabel("Part de l'Information dans l'Offre (%)")
    plt.ylabel("Part d'Audience Thématique (%)")
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    st.pyplot(fig)

    st.success("🎯 Le clustering K-Means identifie deux ères distinctes dans l'évolution de l'audiovisuel français.")


def show_audience_analysis():
    """Fonction principale pour afficher toutes les analyses d'audience"""
    st.header("Audiences des chaînes")

    # Chargement des données
    df = load_audience_data()

    if df is None:
        st.error("Impossible de charger les données. Vérifiez que le fichier existe.")
        return

    # Création des sous-onglets
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Impact VaD",
        "Régression VaD",
        "Évolution par Âge",
        "Info vs Thématiques",
        "Clustering K-Means"
    ])

    with tab1:
        show_vad_impact_chart(df)

    with tab2:
        show_regression_chart(df)

    with tab3:
        show_age_evolution_chart(df)

    with tab4:
        show_info_thematic_chart(df)

    with tab5:
        show_clustering_chart(df)