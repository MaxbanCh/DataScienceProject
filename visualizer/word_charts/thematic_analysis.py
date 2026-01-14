import os
import glob
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

THEMES = {
    'Droite': [
        'droite', 'LR', 'Les Républicains', 'LREM', 'La République En Marche',
        'centristes',
    ],
    'Extreme_Droite': [
        'Extrême droite', 'FN', 'Front National', 'RN', 'Rassemblement National',
        'Identitaire', 'Souverainiste', 'Nationaliste', 'Ultra droite', "Union des droites",
        'Insécurité', 'Immigration', 'Islamophobie', 'Violences policières',
        'Violences urbaines', 'Délinquance', 'Criminalité', 'Identité',
        'Islamisation', 'Laïcité', 'Séparatisme', 'Voile', 'Remigration',
        'Grand Remplacement'
    ],
    'Gauche': [
        'gauche', 'PS', 'Parti Socialiste', 'LFI', 'La France Insoumise',
        'NUPES', 'NFP', 'Nouveau Front Populaire', 'EELV', 'Europe Écologie Les Verts',
        'Les Écologistes'
    ],
    'Économie': [
        'Economie', 'Économie', 'Finance', 'Budget', 'Impôt',
        'Chômage', 'Inflation', "Richesse", 'Pauvreté', 'Inégalité',
        'Redistribution'
    ],
    'Social': [
        'Santé', 'Hôpital', 'Médical', 'Éducation', 'Enseignement',
        'Université', 'Vaccin', 'COVID'
    ],
    'Environnement': [
        'Environnement', 'écologie', 'Climat', 'Climatique',
        'Transition écologique', 'Pollution'
    ],
    'Sécurité & Défense': [
        'Sécurité', 'Défense', 'Armée', 'Guerre', 'Conflit'
    ],
    'International': [
        'Europe', 'Russie', 'Ukraine', 'Israël', 'Palestine', 'Gaza'
    ],
    'Discriminations': [
        'Discrimination', 'Racisme', 'Antisémitisme', 'Xénophobie',
        'Homophobie', 'Transphobie', 'Sexisme', 'Féminisme',
        'Haine', 'Intolérance'
    ],
    'Religion': [
        'Religion', 'Islam', 'Judaïsme', 'Catholicisme'
    ],
    'LGBT+': [
        'LGBT', 'Trans', 'Transgenre', 'Non-binaire'
    ],
    'Technologie': [
        'IA', 'Intelligence Artificielle'
    ],
    'Feminisme': [
        'Féminisme', 'Féminicide', 'Viol', 'Violence conjugale', 'Harcèlement',
        'Consentement', 'Violences sexuelles', 'Avortement', 'IVG', 'PMA'
    ],
}

@st.cache_data
def load_all_word_data(data_path="../ina-api/data/INA/Words"):
    """Charge et combine tous les fichiers CSV des mots"""
    all_data = []
    errors = []
    
    if not os.path.exists(data_path):
        st.error(f"Le chemin {data_path} n'existe pas.")
        return pd.DataFrame()
    
    word_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    
    if not word_folders:
        st.warning(f"Aucun dossier trouvé dans {data_path}")
        return pd.DataFrame()
    
    for word_folder in word_folders:
        word_path = os.path.join(data_path, word_folder)
        csv_files = glob.glob(os.path.join(word_path, "*.csv"))
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                if df.empty:
                    continue
                
                columns_to_keep = ['type', 'value', 'date', 'word']
                missing_columns = [col for col in columns_to_keep if col not in df.columns]
                
                if missing_columns:
                    continue
                
                df = df[columns_to_keep].dropna()
                
                if not df.empty:
                    all_data.append(df)
                    
            except Exception:
                continue
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        combined_df['value'] = pd.to_numeric(combined_df['value'], errors='coerce')
        combined_df = combined_df.dropna(subset=['value'])
        return combined_df
    
    return pd.DataFrame()

def assign_theme(word):
    """Assign a theme to a word"""
    for theme, words in THEMES.items():
        if word in words:
            return theme
    return 'Autre'

def create_theme_channel_matrix(df):
    """Create a matrix: channels x themes with occurrence counts"""
    df = df.copy()
    df['theme'] = df['word'].apply(assign_theme)
    df = df[df['theme'] != 'Autre']
    
    theme_channel = df.groupby(['type', 'theme'])['value'].sum().reset_index()
    matrix = theme_channel.pivot(index='type', columns='theme', values='value').fillna(0)
    
    return matrix

def analyze_channel_theme_relationship(matrix):
    """Perform ML analyses"""
    
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)
    
    # PCA
    pca = PCA(n_components=min(3, matrix.shape[1]))
    pca_result = pca.fit_transform(matrix_scaled)
    
    # Clustering
    n_clusters = min(4, len(matrix))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(matrix_scaled)
    
    # Feature importance
    feature_importance = None
    if len(set(clusters)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            matrix_scaled, clusters, test_size=0.3, random_state=42
        )
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        feature_importance = pd.DataFrame({
            'theme': matrix.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
    
    # Correlation
    correlation_matrix = matrix.T.corr()
    
    return {
        'pca_result': pca_result,
        'pca': pca,
        'clusters': clusters,
        'matrix': matrix,
        'feature_importance': feature_importance,
        'correlation_matrix': correlation_matrix
    }

def show_thematic_analysis():
    """Affiche l'interface Streamlit pour l'analyse thématique"""
    
    st.header("Analyse Thématique par Chaîne")
    
    # Chargement des données
    with st.spinner("Chargement des données..."):
        data = load_all_word_data()
    
    if data.empty:
        st.error("Aucune donnée disponible")
        return
    
    # Filtres
    col1, col2, col3 = st.columns(3)
    
    with col1:
        channels = st.multiselect(
            "Chaînes:",
            sorted(data['type'].unique()),
            default=sorted(data['type'].unique())[:5]
        )
    
    with col2:
        min_date = data['date'].min().date()
        max_date = data['date'].max().date()
        start_date = st.date_input("Date de début:", min_date, key="start_date_theme", max_value=max_date)
    
    with col3:
        end_date = st.date_input("Date de fin:", max_date, key="end_date_theme", min_value=start_date)
    
    # Filtrer les données
    filtered_data = data[
        (data['type'].isin(channels)) &
        (data['date'].dt.date >= start_date) &
        (data['date'].dt.date <= end_date)
    ]
    
    if filtered_data.empty:
        st.warning("Aucune donnée pour ces critères")
        return
    
    st.success(f"{len(filtered_data)} enregistrements chargés")
    
    # Créer la matrice et l'analyse
    matrix = create_theme_channel_matrix(filtered_data)
    results = analyze_channel_theme_relationship(matrix)
    
    # Onglets pour les différentes visualisations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Heatmap",
        "PCA",
        "Clustering",
        "Importance",
        "Corrélation"
    ])
    
    # TAB 1: Heatmap
    with tab1:
        st.subheader("Occurrences des thèmes par chaîne")
        
        # Heatmap interactive
        fig = px.imshow(
            matrix,
            labels=dict(x="Thème", y="Chaîne", color="Occurrences"),
            color_continuous_scale="YlOrRd",
            aspect="auto",
            title="Heatmap des thèmes par chaîne"
        )
        fig.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total occurrences", int(matrix.sum().sum()))
        with col2:
            st.metric("Nombre de chaînes", len(matrix))
        with col3:
            st.metric("Nombre de thèmes", len(matrix.columns))
        
        # Thèmes les plus représentés
        st.subheader("Top 10 thèmes")
        theme_totals = matrix.sum().sort_values(ascending=False).head(10)
        fig = px.bar(
            x=theme_totals.values,
            y=theme_totals.index,
            orientation='h',
            labels={'x': 'Occurrences', 'y': 'Thème'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: PCA
    with tab2:
        st.subheader("Analyse PCA - Projection des chaînes")
        
        pca_result = results['pca_result']
        pca = results['pca']
        clusters = results['clusters']
        
        if pca_result.shape[1] >= 2:
            df_pca = pd.DataFrame({
                'PC1': pca_result[:, 0],
                'PC2': pca_result[:, 1],
                'Chaîne': matrix.index,
                'Cluster': clusters.astype(str)
            })
            
            fig = px.scatter(
                df_pca,
                x='PC1',
                y='PC2',
                color='Cluster',
                text='Chaîne',
                title='Positions des chaînes en espace PCA',
                labels={
                    'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                    'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'
                }
            )
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)
            
            # Variance expliquée
            st.subheader("Variance expliquée")
            variance_df = pd.DataFrame({
                'Composante': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                'Variance': pca.explained_variance_ratio_
            })
            fig = px.bar(
                variance_df,
                x='Composante',
                y='Variance',
                labels={'Variance': 'Variance expliquée (%)'},
                title='Variance expliquée par composante'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: Clustering
    with tab3:
        st.subheader("Analyse de clustering")
        
        clusters = results['clusters']
        
        # Regrouper par cluster
        cluster_info = []
        for i in range(len(set(clusters))):
            channels = matrix.index[clusters == i].tolist()
            cluster_info.append({
                'Cluster': i + 1,
                'Chaînes': ', '.join(channels),
                'Nombre': len(channels)
            })
        
        cluster_df = pd.DataFrame(cluster_info)
        st.dataframe(cluster_df, use_container_width=True)
        
        # Visualisation des clusters
        if results['pca_result'].shape[1] >= 2:
            pca_result = results['pca_result']
            df_clusters = pd.DataFrame({
                'PC1': pca_result[:, 0],
                'PC2': pca_result[:, 1],
                'Chaîne': matrix.index,
                'Cluster': clusters.astype(str)
            })
            
            fig = px.scatter(
                df_clusters,
                x='PC1',
                y='PC2',
                color='Cluster',
                text='Chaîne',
                size_max=50,
                title='Visualisation des clusters'
            )
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: Feature Importance
    with tab4:
        st.subheader("Importance des thèmes")
        
        if results['feature_importance'] is not None:
            feat_imp = results['feature_importance'].head(15)
            
            fig = px.bar(
                feat_imp,
                x='importance',
                y='theme',
                orientation='h',
                labels={'importance': 'Importance', 'theme': 'Thème'},
                title='Top 15 thèmes discriminants'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tableau
            st.subheader("Détails")
            st.dataframe(feat_imp, use_container_width=True)
        else:
            st.info("Pas assez de clusters pour calculer l'importance")
    
    # TAB 5: Corrélation
    with tab5:
        st.subheader("Similitude entre chaînes")
        
        corr_matrix = results['correlation_matrix']
        
        fig = px.imshow(
            corr_matrix,
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            labels=dict(color='Corrélation'),
            title='Matrice de corrélation entre chaînes',
            aspect='auto'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Paires les plus similaires
        st.subheader("Chaînes les plus similaires")
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                correlations.append({
                    'Chaîne 1': corr_matrix.columns[i],
                    'Chaîne 2': corr_matrix.columns[j],
                    'Corrélation': corr_matrix.iloc[i, j]
                })
        
        corr_df = pd.DataFrame(correlations).sort_values('Corrélation', ascending=False)
        st.dataframe(corr_df, use_container_width=True)