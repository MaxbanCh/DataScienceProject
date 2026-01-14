import os
import glob
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


THEMES = {
    # 'Politique': [
    #     'Politique', 'Gouvernement', 'Présidence', 'président', 'Présidentielle',
    #     'député', 'sénateur', 'maire', 'ministre', 'Vote', 'Election', 'Reforme',
    #     'Loi', 'Manifestation', 'Grève', 'Revendication'
    # ],
    # 'Partis Politiques': [
    #     'droite', 'gauche', 'extrême droite', 'extrême gauche', 'centristes',
    #     'FN', 'Front National', 'RN', 'Rassemblement National',
    #     'PS', 'Parti Socialiste', 'LR', 'Les Républicains',
    #     'LREM', 'La République En Marche', 'EELV', 'Europe Écologie Les Verts',
    #     'LFI', 'La France Insoumise', 'NUPES', 'NFP', 'Nouveau Front Populaire',
    #     'Les Écologistes', 'union des droites'
    # ],
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
    'Gauche' : [
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
def load_all_word_data(data_path="../../ina-api/data/INA/Words"):
    """Charge et combine tous les fichiers CSV des mots"""
    all_data = []
    errors = []
    empty_files = []
    
    # Vérifier si le chemin existe
    if not os.path.exists(data_path):
        st.error(f"Le chemin {data_path} n'existe pas.")
        return pd.DataFrame()
    
    # Parcourir tous les dossiers de mots
    word_folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    
    if not word_folders:
        st.warning(f"Aucun dossier trouvé dans {data_path}")
        return pd.DataFrame()
    
    for word_folder in word_folders:
        word_path = os.path.join(data_path, word_folder)
        
        # Parcourir tous les fichiers CSV du mot
        csv_files = glob.glob(os.path.join(word_path, "*.csv"))
        
        if not csv_files:
            st.warning(f"Aucun fichier CSV trouvé dans {word_folder}")
            continue
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Vérifier si le fichier contient des données
                if df.empty:
                    empty_files.append(csv_file)
                    continue
                
                # Vérifier que les colonnes nécessaires existent
                columns_to_keep = ['type', 'value', 'date', 'word']
                missing_columns = [col for col in columns_to_keep if col not in df.columns]
                
                if missing_columns:
                    errors.append(f"{os.path.basename(csv_file)}: colonnes manquantes {missing_columns}")
                    continue
                
                # Ne garder que les colonnes nécessaires
                df = df[columns_to_keep]
                
                # Supprimer les lignes avec des valeurs manquantes
                df = df.dropna()
                
                if not df.empty:
                    all_data.append(df)
                else:
                    empty_files.append(csv_file)
                    
            except pd.errors.EmptyDataError:
                empty_files.append(csv_file)
            except Exception as e:
                errors.append(f"{os.path.basename(csv_file)}: {str(e)}")
    
    if errors:
        with st.expander(f"{len(errors)} erreur(s) de chargement"):
            for error in errors[:10]:  # Limiter l'affichage
                st.warning(error)
            if len(errors) > 10:
                st.info(f"... et {len(errors) - 10} autres erreurs")
        
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Conversion de la date
        try:
            combined_df['date'] = pd.to_datetime(combined_df['date'])
        except Exception as e:
            st.error(f"Erreur lors de la conversion des dates: {e}")
            return pd.DataFrame()
        
        # Vérifier que les valeurs sont numériques
        combined_df['value'] = pd.to_numeric(combined_df['value'], errors='coerce')
        combined_df = combined_df.dropna(subset=['value'])
        
        return combined_df
    else:
        st.error("Aucune donnée valide n'a pu être chargée.")
        return pd.DataFrame()

no_theme = []

def assign_theme(word):
    """Assign a theme to a word"""
    for theme, words in THEMES.items():
        if word in words:
            return theme
    if word not in no_theme:
        no_theme.append(word)
    # print(f"No theme found for word: {word}. Checked themes: {', '.join(no_theme)}")
    return 'Autre'

def filter_channel(df, channels):
    """Filter dataframe for specific channels"""
    return df[df['type'].isin(channels)]

def filter_date_range(df, start_date, end_date):
    """Filter dataframe for a specific date range"""
    return df[(df['date'] >= start_date) & (df['date'] <= end_date)]


def create_theme_channel_matrix(df):
    """Create a matrix: channels x themes with occurrence counts"""
    df['theme'] = df['word'].apply(assign_theme)
    df = df[df['theme'] != 'Autre']

    print(f"No theesme words: {no_theme}")
    
    # Aggregate by channel and theme
    theme_channel = df.groupby(['type', 'theme'])['value'].sum().reset_index()
    
    # Pivot to create matrix
    matrix = theme_channel.pivot(index='type', columns='theme', values='value').fillna(0)
    
    return matrix


def analyze_channel_theme_relationship(matrix):
    """Perform various ML analyses"""
    
    # Normalize data
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)
    
    # 1. PCA - Dimensionality reduction
    print("\n=== PCA Analysis ===")
    pca = PCA(n_components=min(3, matrix.shape[1]))
    pca_result = pca.fit_transform(matrix_scaled)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")
    
    # 2. Clustering - Group similar channels
    print("\n=== Clustering Analysis ===")
    n_clusters = min(4, len(matrix))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(matrix_scaled)
    
    print("\nChannel clusters:")
    for i in range(n_clusters):
        channels_in_cluster = matrix.index[clusters == i].tolist()
        print(f"Cluster {i+1}: {', '.join(channels_in_cluster)}")
    
    # 3. Feature importance - Which themes differentiate channels most?
    print("\n=== Feature Importance Analysis ===")
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
        
        print("\nTop 5 discriminating themes:")
        print(feature_importance.head())
    
    # 4. Correlation analysis
    print("\n=== Correlation Analysis ===")
    correlation_matrix = matrix.T.corr()
    
    # Find most correlated channel pairs
    correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            correlations.append({
                'channel1': correlation_matrix.columns[i],
                'channel2': correlation_matrix.columns[j],
                'correlation': correlation_matrix.iloc[i, j]
            })
    
    correlations_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
    print("\nTop 5 most similar channel pairs:")
    print(correlations_df.head())
    
    return {
        'pca_result': pca_result,
        'pca': pca,
        'clusters': clusters,
        'feature_importance': feature_importance if len(set(clusters)) > 1 else None,
        'correlation_matrix': correlation_matrix
    }

def visualize_results(matrix, results):
    """Create visualizations"""
    
    # 1. Heatmap of themes by channel
    plt.figure(figsize=(14, 8))
    sns.heatmap(matrix, cmap='YlOrRd', annot=True, fmt='g')
    plt.title('Theme Occurrences by Channel')
    plt.xlabel('Theme')
    plt.ylabel('Channel')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('theme_channel_heatmap.png', dpi=300, bbox_inches='tight')
    print("\nSaved: theme_channel_heatmap.png")
    
    # 2. PCA visualization
    if results['pca_result'].shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        plt.scatter(results['pca_result'][:, 0], results['pca_result'][:, 1], 
                   c=results['clusters'], cmap='viridis', s=200, alpha=0.6)
        
        for i, channel in enumerate(matrix.index):
            plt.annotate(channel, (results['pca_result'][i, 0], results['pca_result'][i, 1]),
                        fontsize=9, ha='center')
        
        plt.xlabel(f'PC1 ({results["pca"].explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({results["pca"].explained_variance_ratio_[1]:.1%})')
        plt.title('Channels in PCA Space (colored by cluster)')
        plt.colorbar(label='Cluster')
        plt.tight_layout()
        plt.savefig('pca_channels.png', dpi=300, bbox_inches='tight')
        print("Saved: pca_channels.png")
    
    # 3. Feature importance
    if results['feature_importance'] is not None:
        plt.figure(figsize=(10, 6))
        top_features = results['feature_importance'].head(10)
        plt.barh(top_features['theme'], top_features['importance'])
        plt.xlabel('Importance')
        plt.title('Top 10 Themes that Differentiate Channels')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("Saved: feature_importance.png")
    
    # 4. Channel correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(results['correlation_matrix'], annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, vmin=-1, vmax=1)
    plt.title('Channel Similarity (Correlation)')
    plt.tight_layout()
    plt.savefig('channel_correlation.png', dpi=300, bbox_inches='tight')
    print("Saved: channel_correlation.png")


def main():
    print("Loading data...")
    df = load_all_word_data()
    df = filter_channel(df, ['BFM TV', 'CNews', 'franceinfo:', 'LCI'])
    # df = filter_channel(df, ['TF1', 'France 2', 'France 3', 'Arte', 'M6'])

    df = filter_date_range(df, '2022-01-01', '2025-11-30')
    
    if df.empty:
        print("No data found!")
        return
    
    print(f"Loaded {len(df)} records")
    print(f"Unique words: {df['word'].nunique()}")
    print(f"Unique channels: {df['type'].nunique()}")
    
    print("\nCreating theme-channel matrix...")
    matrix = create_theme_channel_matrix(df)
    
    print(f"\nMatrix shape: {matrix.shape}")
    print(f"Channels: {list(matrix.index)}")
    print(f"Themes: {list(matrix.columns)}")
    
    print("\nTheme totals across all channels:")
    print(matrix.sum().sort_values(ascending=False))
    
    print("\nPerforming ML analysis...")
    results = analyze_channel_theme_relationship(matrix)
    
    print("\nGenerating visualizations...")
    visualize_results(matrix, results)
    
    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    main()