import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
from women_proportion.load_data import aggregate_data, load_proportion_data

def show_women_proportion_charts():
    st.header("Visualisation des données INA - Proportion Femmes/Hommes")

    with st.spinner("Chargement des données..."):
        data = load_proportion_data()

    if data.empty:
        st.error("Aucune donnée trouvée. Vérifiez le chemin vers les fichiers CSV.")
        st.stop()

    st.success(f"Données chargées : {len(data)} enregistrements")

    # Créer des colonnes pour les filtres
    st.subheader("Filtres")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # Filtre par type (Femme/Homme)
        available_types = sorted(data['type'].unique())
        selected_types = st.multiselect(
            "Type:",
            options=available_types,
            default=["Femme", "Homme"],
            key="types_select"
        )

    with col2:
        # Filtre par chaîne
        available_channels = sorted(data['channel'].unique())
        selected_channels = st.multiselect(
            "Chaînes:",
            options=available_channels,
            default=available_channels[:1] if len(available_channels) >= 1 else available_channels,
            key="channels_select_prop"
        )

    with col3:
        min_date = data['date'].min().date()
        max_date = data['date'].max().date()
        start_date = st.date_input("Date de début:", min_date, key="start_date_prop", max_value=max_date)
    
    with col4:
        end_date = st.date_input("Date de fin:", max_date, key="end_date_prop", min_value=start_date)

    with col5:
        # Période d'agrégation
        aggregation_period = st.selectbox(
            "Agrégation:",
            options=['daily', 'monthly', 'yearly'],
            format_func=lambda x: {'monthly': 'Mensuel', 'daily': 'Quotidien', 'yearly': 'Annuel'}[x],
            key="agg_period_prop",
        )

    # Filtrage des données
    filtered_data = data.copy()

    if not selected_types:
        st.warning("Sélectionnez au moins un type (Femme/Homme)")
        return

    if selected_types:
        filtered_data = filtered_data[filtered_data['type'].isin(selected_types)]

    if selected_channels:
        filtered_data = filtered_data[filtered_data['channel'].isin(selected_channels)]

    filtered_data = filtered_data[
        (filtered_data['date'].dt.date >= start_date) &
        (filtered_data['date'].dt.date <= end_date)
    ]
    
    if filtered_data.empty:
        st.warning("Aucune donnée ne correspond aux critères sélectionnés.")
        return

    # Agrégation des données
    aggregated_data = aggregate_data(filtered_data, aggregation_period)

    # Création du graphique
    if not aggregated_data.empty:
        st.header("Graphique")
        
        # Colonnes pour les métriques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pourcentage moyen", f"{aggregated_data['value'].mean():.1f}%")
        with col2:
            st.metric("Nombre de types", len(selected_types))
        with col3:
            st.metric("Nombre de chaînes", len(selected_channels))
        
        # Graphique principal - Ligne avec régression
        fig = create_chart_with_regression(aggregated_data, aggregation_period)
        st.plotly_chart(fig, use_container_width=True)
        
        # Graphique secondaire - Bar chart
        if len(selected_channels) > 1 or len(selected_types) > 1:
            fig_bar = px.bar(
                aggregated_data,
                x='date',
                y='value',
                color='type',
                facet_col='channel',
                title=f"Proportion par chaîne ({aggregation_period})"
            )
            
            fig_bar.update_layout(height=500)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Tableau des données
        with st.expander("Voir les données détaillées"):
            st.dataframe(aggregated_data.sort_values('date', ascending=False))
        
        # Statistiques par type
        if len(selected_types) > 1:
            st.header("Comparaison par type")
            type_stats = aggregated_data.groupby('type')['value'].agg(['mean', 'min', 'max', 'std']).round(2)
            type_stats.columns = ['Moyenne (%)', 'Min (%)', 'Max (%)', 'Écart-type']
            st.dataframe(type_stats)
        
        # Statistiques par chaîne
        if len(selected_channels) > 1:
            st.header("Comparaison par chaîne")
            channel_stats = aggregated_data.groupby('channel')['value'].agg(['mean', 'min', 'max', 'std']).round(2)
            channel_stats.columns = ['Moyenne (%)', 'Min (%)', 'Max (%)', 'Écart-type']
            st.dataframe(channel_stats)
        
        # Analyse de régression par type
        st.header("Analyse de tendance par type")
        regression_results = show_regression_by_type(aggregated_data)

    else:
        st.warning("Aucune donnée ne correspond aux critères sélectionnés.")


def create_chart_with_regression(data, aggregation_period):
    """Crée un graphique avec ligne de régression par type et chaîne"""
    fig = go.Figure()
    
    # Ajouter les points réels
    for type_val in data['type'].unique():
        for channel_val in data['channel'].unique():
            subset = data[(data['type'] == type_val) & (data['channel'] == channel_val)].copy()
            
            if subset.empty:
                continue
            
            # Convertir les dates en nombres pour la régression
            subset = subset.sort_values('date')
            X = np.arange(len(subset)).reshape(-1, 1)
            y = subset['value'].values
            
            # Tracer les points
            fig.add_trace(go.Scatter(
                x=subset['date'],
                y=y,
                mode='markers',
                name=f"{type_val} - {channel_val}",
                marker=dict(size=6)
            ))
            
            # Calculer et tracer la ligne de régression
            if len(subset) > 1:
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                
                slope = model.coef_[0]
                trend = "Hausse" if slope > 0 else "Baisse"
                
                fig.add_trace(go.Scatter(
                    x=subset['date'],
                    y=y_pred,
                    mode='lines',
                    name=f"Tendance {type_val} - {channel_val} ({trend})",
                    line=dict(dash='dash'),
                    hovertemplate='<b>Tendance</b><br>Date: %{x}<br>Valeur: %{y:.2f}%<extra></extra>'
                ))
    
    fig.update_layout(
        title=f"Évolution et tendances de la proportion Femmes/Hommes ({aggregation_period})",
        xaxis_title="Date",
        yaxis_title="Pourcentage (%)",
        hovermode='x unified',
        legend_title="Légende",
        height=500
    )
    
    return fig


def show_regression_by_type(data):
    """Affiche les résultats de régression par type"""
    results = {}
    
    for type_val in data['type'].unique():
        type_data = data[data['type'] == type_val].copy().sort_values('date')
        
        if len(type_data) < 2:
            continue
        
        # Convertir les dates en nombres
        X = np.arange(len(type_data)).reshape(-1, 1)
        y = type_data['value'].values
        
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
        
        results[type_val] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'mse': mse,
            'trend': "Hausse" if slope > 0 else "Baisse"
        }
        
        # Afficher les résultats
        with st.expander(f"{type_val}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pente (variation/période)", f"{slope:.4f}%")
            with col2:
                st.metric("R² (qualité d'ajustement)", f"{r_squared:.4f}")
            with col3:
                st.metric("Tendance", results[type_val]['trend'])
            
            st.write(f"**Équation:** y = {slope:.4f}x + {intercept:.2f}")
            st.write(f"**Erreur quadratique moyenne:** {mse:.4f}")
    
    return results