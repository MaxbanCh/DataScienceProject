import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def analyze_channel_theme_relationship(matrix):
    """Perform ML analyses"""
    
    scaler = StandardScaler()
    matrix_scaled = scaler.fit_transform(matrix)
    
    clusters = None
    pca_result = None
    pca = None
    feature_importance = None
    # PCA
    if matrix.shape[1] > 2:
        try:
            pca = PCA(n_components=min(3, matrix.shape[1]))
            pca_result = pca.fit_transform(matrix_scaled)
        
            # Clustering
            n_clusters = min(3, len(matrix))
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
        except Exception as e:
            print(f"An error occurred during analysis: {e}")
    
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
