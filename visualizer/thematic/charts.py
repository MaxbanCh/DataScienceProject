import numpy as np
from sklearn.linear_model import LinearRegression
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from thematic.load_data import THEMES, assign_theme
from thematic.load_data import load_all_word_data, create_theme_channel_matrix
from thematic.analysis import analyze_channel_theme_relationship


def show_thematic_analysis():
    """Affiche l'interface Streamlit pour l'analyse thématique"""

    st.header("Analyse Thématique par Chaîne")
    print(THEMES.keys())

    # Chargement des données
    with st.spinner("Chargement des données..."):
        data = load_all_word_data()
        data = data.copy()
        data["theme"] = data["word"].apply(assign_theme)
        data = data[data["theme"] != "Autre"]

    if data.empty:
        st.error("Aucune donnée disponible")
        return

    # Filtres
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        channels = st.multiselect(
            "Chaînes:",
            sorted(data["type"].unique()),
            default=["BFM TV", "CNews", "LCI", "franceinfo:"],
        )

    with col2:
        min_date = data["date"].min().date()
        max_date = data["date"].max().date()
        start_date = st.date_input(
            "Date de début:", min_date, key="start_date_theme", min_value=min_date, max_value=max_date
        )

    with col3:
        end_date = st.date_input(
            "Date de fin:", max_date, key="end_date_theme", min_value=start_date, max_value=max_date
        )

    with col4:
        selected_themes = st.multiselect(
            "Thèmes:",
            sorted(set().union(list(THEMES.keys()))),
            default=[theme for theme in sorted(set().union(list(THEMES.keys()))) if theme != "Vie Institutionnelle"],
        )

    if not selected_themes:
        st.warning("Veuillez sélectionner au moins un thème")
        return

    # Filtrer les données
    filtered_data = data[
        (data["type"].isin(channels))
        & (data["date"].dt.date >= start_date)
        & (data["date"].dt.date <= end_date)
        & (data["theme"].isin(selected_themes))
    ]

    if filtered_data.empty:
        st.warning("Aucune donnée pour ces critères")
        return

    st.success(f"{len(filtered_data)} enregistrements chargés")

    # Créer la matrice et l'analyse
    matrix = create_theme_channel_matrix(filtered_data)
    results = analyze_channel_theme_relationship(matrix)

    # Onglets pour les différentes visualisations
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "Heatmap",
            "ACP",
            "Clustering",
            "Importance",
            "Corrélation",
            "Analyse Temporelle",
            "Méthodologie",
        ]
    )

    # TAB 1: Heatmap
    with tab1:
        st.subheader("Occurrences des thèmes par chaîne")

        # Heatmap interactive
        fig = px.imshow(
            matrix,
            labels=dict(x="Thème", y="Chaîne", color="Occurrences"),
            color_continuous_scale="YlOrRd",
            aspect="auto",
            title="Heatmap des thèmes par chaîne",
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
            orientation="h",
            labels={"x": "Occurrences", "y": "Thème"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # TAB 2: PCA
    with tab2:
        st.subheader("Analyse ACP - Projection des chaînes")

        pca_result = results["pca_result"]
        pca = results["pca"]
        clusters = results["clusters"]

        if pca_result.shape[1] >= 2:
            df_pca = pd.DataFrame(
                {
                    "CP1": pca_result[:, 0],
                    "CP2": pca_result[:, 1],
                    "Chaîne": matrix.index,
                    "Cluster": clusters.astype(str),
                }
            )

            fig = px.scatter(
                df_pca,
                x="CP1",
                y="CP2",
                color="Cluster",
                text="Chaîne",
                title="Positions des chaînes en espace ACP",
                labels={
                    "CP1": f"CP1 ({pca.explained_variance_ratio_[0]:.1%})",
                    "CP2": f"CP2 ({pca.explained_variance_ratio_[1]:.1%})",
                },
            )
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)

            # Variance expliquée
            st.subheader("Variance expliquée")
            variance_df = pd.DataFrame(
                {
                    "Composante": [
                        f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))
                    ],
                    "Variance": pca.explained_variance_ratio_,
                }
            )
            fig = px.bar(
                variance_df,
                x="Composante",
                y="Variance",
                labels={"Variance": "Variance expliquée (%)"},
                title="Variance expliquée par composante",
            )
            st.plotly_chart(fig, use_container_width=True)

    # TAB 3: Clustering
    with tab3:
        st.subheader("Analyse de clustering")

        clusters = results["clusters"]

        # Regrouper par cluster
        cluster_info = []
        for i in range(len(set(clusters))):
            channels = matrix.index[clusters == i].tolist()
            cluster_info.append(
                {
                    "Cluster": i + 1,
                    "Chaînes": ", ".join(channels),
                    "Nombre": len(channels),
                }
            )

        cluster_df = pd.DataFrame(cluster_info)
        st.dataframe(cluster_df, use_container_width=True)

        # Visualisation des clusters
        if results["pca_result"].shape[1] >= 2:
            pca_result = results["pca_result"]
            df_clusters = pd.DataFrame(
                {
                    "PC1": pca_result[:, 0],
                    "PC2": pca_result[:, 1],
                    "Chaîne": matrix.index,
                    "Cluster": clusters.astype(str),
                }
            )

            fig = px.scatter(
                df_clusters,
                x="PC1",
                y="PC2",
                color="Cluster",
                text="Chaîne",
                size_max=50,
                title="Visualisation des clusters",
            )
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)

    # TAB 4: Feature Importance
    with tab4:
        st.subheader("Importance des thèmes")

        if results["feature_importance"] is not None:
            feat_imp = results["feature_importance"].head(15)

            fig = px.bar(
                feat_imp,
                x="importance",
                y="theme",
                orientation="h",
                labels={"importance": "Importance", "theme": "Thème"},
                title="Top 15 thèmes discriminants",
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

        corr_matrix = results["correlation_matrix"]

        fig = px.imshow(
            corr_matrix,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            labels=dict(color="Corrélation"),
            title="Matrice de corrélation entre chaînes",
            aspect="auto",
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Paires les plus similaires
        st.subheader("Chaînes les plus similaires")
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                correlations.append(
                    {
                        "Chaîne 1": corr_matrix.columns[i],
                        "Chaîne 2": corr_matrix.columns[j],
                        "Corrélation": corr_matrix.iloc[i, j],
                    }
                )

        corr_df = pd.DataFrame(correlations).sort_values("Corrélation", ascending=False)
        st.dataframe(corr_df, use_container_width=True)

    # TAB 6: Analyse Temporelle
    with tab6:
        st.subheader("Analyse temporelle des thèmes")

        # Assigner les thèmes aux données filtrées
        temporal_data = filtered_data.copy()
        temporal_data["theme"] = temporal_data["word"].apply(assign_theme)
        temporal_data = temporal_data[temporal_data["theme"] != "Autre"]

        # Options de visualisation
        freq = st.selectbox(
            "Fréquence d'agrégation:",
            ["D", "W", "M", "Q"],
            format_func=lambda x: {
                "D": "Jour",
                "W": "Semaine",
                "M": "Mois",
                "Q": "Trimestre",
            }[x],
            index=2,
        )

        # Option pour afficher la régression
        show_regression = st.checkbox("Afficher les lignes de régression", value=False)

        # Filtrer par thèmes sélectionnés
        temporal_filtered = temporal_data[temporal_data["theme"].isin(selected_themes)]

        # Graphique 1: Évolution par thème (toutes chaînes confondues)
        st.subheader("Évolution globale des thèmes")
        theme_evolution = (
            temporal_filtered.groupby([pd.Grouper(key="date", freq=freq), "theme"])[
                "value"
            ]
            .sum()
            .reset_index()
        )

        fig = px.line(
            theme_evolution,
            x="date",
            y="value",
            color="theme",
            labels={"date": "Date", "value": "Occurrences", "theme": "Thème"},
            title="Évolution temporelle des thèmes",
        )

        # Ajouter les régressions si activé
        if show_regression:
            for theme in selected_themes:
                theme_data = theme_evolution[theme_evolution["theme"] == theme].copy()
                if len(theme_data) > 1:
                    # Convertir les dates en nombres pour la régression
                    X = np.arange(len(theme_data)).reshape(-1, 1)
                    y = theme_data["value"].values
                    
                    # Calculer la régression
                    reg = LinearRegression().fit(X, y)
                    y_pred = reg.predict(X)
                    
                    # Ajouter la ligne de régression
                    fig.add_trace(go.Scatter(
                        x=theme_data["date"],
                        y=y_pred,
                        mode="lines",
                        name=f"{theme} (régression)",
                        line=dict(dash="dash"),
                        hovertemplate="%{y:.0f}<extra></extra>"
                    ))
            
            st.subheader("Statistiques de régression par thème")
            show_regression_by_theme(theme_evolution)

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Graphique 2: Évolution par chaîne
        st.subheader("Évolution par chaîne")

        selected_theme_for_channel = st.selectbox(
            "Choisir un thème pour voir par chaîne:", selected_themes
        )

        channel_theme_evolution = (
            temporal_filtered[temporal_filtered["theme"] == selected_theme_for_channel]
            .groupby([pd.Grouper(key="date", freq=freq), "type"])["value"]
            .sum()
            .reset_index()
        )

        fig = px.line(
            channel_theme_evolution,
            x="date",
            y="value",
            color="type",
            labels={"date": "Date", "value": "Occurrences", "type": "Chaîne"},
            title=f'Évolution du thème "{selected_theme_for_channel}" par chaîne',
        )

        # Ajouter les régressions par chaîne si activé
        if show_regression:
            for channel in channel_theme_evolution["type"].unique():
                channel_data = channel_theme_evolution[channel_theme_evolution["type"] == channel].copy()
                if len(channel_data) > 1:
                    X = np.arange(len(channel_data)).reshape(-1, 1)
                    y = channel_data["value"].values
                    
                    reg = LinearRegression().fit(X, y)
                    y_pred = reg.predict(X)
                    
                    fig.add_trace(go.Scatter(
                        x=channel_data["date"],
                        y=y_pred,
                        mode="lines",
                        name=f"{channel} (régression)",
                        line=dict(dash="dash"),
                        hovertemplate="%{y:.0f}<extra></extra>"
                    ))
            
            st.subheader("Statistiques de régression par chaîne")
            show_regression_by_channel(channel_theme_evolution)

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Graphique 3: Heatmap temporelle
        st.subheader("Heatmap temporelle")

        # Créer une matrice temps x thème
        temporal_matrix = (
            temporal_filtered.groupby([pd.Grouper(key="date", freq=freq), "theme"])[
                "value"
            ]
            .sum()
            .reset_index()
        )

        temporal_pivot = temporal_matrix.pivot(
            index="date", columns="theme", values="value"
        ).fillna(0)

        fig = px.imshow(
            temporal_pivot.T,
            labels=dict(x="Date", y="Thème", color="Occurrences"),
            color_continuous_scale="YlOrRd",
            aspect="auto",
            title="Heatmap temporelle des thèmes",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Graphique 4: Parts de marché des thèmes (stacked area)
        st.subheader("Parts relatives des thèmes")

        # Normaliser pour avoir des pourcentages
        temporal_percentage = temporal_matrix.copy()
        total_by_date = temporal_percentage.groupby("date")["value"].transform("sum")
        temporal_percentage["percentage"] = (
            temporal_percentage["value"] / total_by_date
        ) * 100

        fig = px.area(
            temporal_percentage,
            x="date",
            y="percentage",
            color="theme",
            labels={"date": "Date", "percentage": "Part (%)", "theme": "Thème"},
            title="Parts relatives des thèmes au fil du temps",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    with tab7:
        st.subheader("Méthodologie")
        st.markdown("""
            Les données proviennent d'un site de l'INA, qui analysent les flux audios des principales chaines de télévision avec une IA, afin de ressortir des données.  
            On a ensuite regroupés les différents mots par thèmes afin des les analyser de manière groupée et observer des tendances par chaines.
            
            Les thèmes sont regroupés ainsi :
        """)
        for theme, words in THEMES.items():
            if theme != "Autre":
                st.markdown(f"**{theme}** : {', '.join(words)}")
        
        st.markdown("""
            On a pu ensuite réaliser plusieurs analyses :
            - Une heatmap des occurrences des thèmes par chaîne
            - Une ACP pour visualiser les similarités entre chaînes
            - Un clustering des chaînes basé sur les thèmes (à l'aide des K-Moyennes)
            - L'importance des thèmes pour discriminer les chaînes (à l'aide de Random Forest)
            - La corrélation entre chaînes basée sur les thèmes
            - Une analyse temporelle de l'évolution des thèmes
        """)

def show_regression_by_theme(data):
    """Affiche les résultats de régression par theme"""
    results = {}
    
    for theme_val in data['theme'].unique():
        theme_data = data[data['theme'] == theme_val].copy().sort_values('date')
        
        if len(theme_data) < 2:
            continue
        
        # Convertir les dates en nombres
        X = np.arange(len(theme_data)).reshape(-1, 1)
        y = theme_data['value'].values
        
        # Fit du modèle
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculs statistiques
        y_pred = model.predict(X)
        residuals = y - y_pred
        mse = np.mean(residuals ** 2)
        r_squared = model.score(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        results[theme_val] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'mse': mse,
            'trend': "Hausse" if slope > 0 else "Baisse"
        }
        
        # Afficher les résultats
        with st.expander(f"{theme_val}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pente (variation/période)", f"{slope:.4f}%")
            with col2:
                st.metric("R² (qualité d'ajustement)", f"{r_squared:.4f}")
            with col3:
                st.metric("Tendance", results[theme_val]['trend'])
            
            st.write(f"**Équation:** y = {slope:.4f}x + {intercept:.2f}")
            st.write(f"**Erreur quadratique moyenne:** {mse:.4f}")
    
    return results

def show_regression_by_channel(data):
    """Affiche les résultats de régression par chaîne"""
    results = {}
    
    for channel_val in data['type'].unique():
        channel_data = data[data['type'] == channel_val].copy().sort_values('date')
        
        if len(channel_data) < 2:
            continue
        
        # Convertir les dates en nombres
        X = np.arange(len(channel_data)).reshape(-1, 1)
        y = channel_data['value'].values
        
        # Fit du modèle
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculs statistiques
        y_pred = model.predict(X)
        residuals = y - y_pred
        mse = np.mean(residuals ** 2)
        r_squared = model.score(X, y)
        slope = model.coef_[0]
        intercept = model.intercept_
        
        results[channel_val] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'mse': mse,
            'trend': "Hausse" if slope > 0 else "Baisse"
        }
        
        # Afficher les résultats
        with st.expander(f"{channel_val}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pente (variation/période)", f"{slope:.4f}%")
            with col2:
                st.metric("R² (qualité d'ajustement)", f"{r_squared:.4f}")
            with col3:
                st.metric("Tendance", results[channel_val]['trend'])
            
            st.write(f"**Équation:** y = {slope:.4f}x + {intercept:.2f}")
            st.write(f"**Erreur quadratique moyenne:** {mse:.4f}")
    
    return results