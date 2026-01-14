import streamlit as st
import plotly.express as px
from word_charts.load_data import aggregate_data, load_word_data

def show_word_charts():
    st.header("Visualisation des données INA - Mots")

    with st.spinner("Chargement des données..."):
        data = load_word_data()

    if data.empty:
        st.error("Aucune donnée trouvée. Vérifiez le chemin vers les fichiers CSV.")
        st.stop()

    st.success(f"Données chargées : {len(data)} enregistrements")

    # Créer des colonnes pour les filtres (pas de sidebar)
    st.subheader("Filtres")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # Filtre par mot
        available_words = sorted(data['word'].unique())
        selected_words = st.multiselect(
            "Mots:",
            options=available_words,
            default=available_words[:3] if len(available_words) >= 3 else available_words,
            key="words_select"
        )

    with col2:
        # Filtre par chaîne (type)
        available_channels = sorted(data['type'].unique())
        selected_channels = st.multiselect(
            "Chaînes:",
            options=available_channels,
            default=available_channels,
            key="channels_select"
        )

    with col3:
        min_date = data['date'].min().date()
        max_date = data['date'].max().date()
        start_date = st.date_input("Date de début:", min_date, key="start_date_word", max_value=max_date)
    
    with col4:
        end_date = st.date_input("Date de fin:", max_date, key="end_date_word", min_value=start_date)

    with col5:
        # Période d'agrégation
        aggregation_period = st.selectbox(
            "Agrégation:",
            options=['daily', 'monthly', 'yearly'],
            format_func=lambda x: {'daily': 'Quotidien', 'monthly': 'Mensuel', 'yearly': 'Annuel'}[x],
            key="agg_period"
        )

    with col5:
        # Type de graphique
        chart_type = st.selectbox(
            "Graphique:",
            options=['line', 'bar', 'area'],
            format_func=lambda x: {'line': 'Ligne', 'bar': 'Barres', 'area': 'Aires'}[x],
            key="chart_type"
        )

    # Filtrage des données
    filtered_data = data.copy()

    if selected_words:
        filtered_data = filtered_data[filtered_data['word'].isin(selected_words)]

    if selected_channels:
        filtered_data = filtered_data[filtered_data['type'].isin(selected_channels)]

    filtered_data = filtered_data[
        (filtered_data['date'].dt.date >= start_date) &
        (filtered_data['date'].dt.date <= end_date)
    ]
    
    # Agrégation des données
    aggregated_data = aggregate_data(filtered_data, aggregation_period)

    # Création du graphique
    if not aggregated_data.empty:
        st.header("Graphique")
        
        # Colonnes pour les métriques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre total d'occurrences", int(aggregated_data['value'].sum()))
        with col2:
            st.metric("Nombre de mots sélectionnés", len(selected_words))
        with col3:
            st.metric("Nombre de chaînes", len(selected_channels))
        
        # Graphique principal
        if chart_type == 'line':
            fig = px.line(
                aggregated_data, 
                x='date', 
                y='value', 
                color='word',
                line_dash='type',
                title=f"Évolution du nombre d'occurrences ({aggregation_period})"
            )
        elif chart_type == 'bar':
            fig = px.bar(
                aggregated_data, 
                x='date', 
                y='value', 
                color='word',
                pattern_shape='type',
                title=f"Nombre d'occurrences par période ({aggregation_period})"
            )
        else:  # area
            fig = px.area(
                aggregated_data, 
                x='date', 
                y='value', 
                color='word',
                line_group='type',
                title=f"Aires cumulées des occurrences ({aggregation_period})"
            )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Nombre d'occurrences",
            hovermode='x unified',
            legend_title="Légende"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tableau des données
        if st.expander("Voir les données détaillées"):
            st.dataframe(aggregated_data.sort_values('date', ascending=False))
        
        # Statistiques par mot
        if len(selected_words) > 1:
            st.header("Comparaison par mot")
            word_stats = aggregated_data.groupby('word')['value'].agg(['sum', 'mean', 'std']).round(2)
            word_stats.columns = ['Total', 'Moyenne', 'Écart-type']
            st.dataframe(word_stats)
        
        # Statistiques par chaîne
        if len(selected_channels) > 1:
            st.header("Comparaison par chaîne")
            channel_stats = aggregated_data.groupby('type')['value'].agg(['sum', 'mean', 'std']).round(2)
            channel_stats.columns = ['Total', 'Moyenne', 'Écart-type']
            st.dataframe(channel_stats)

    else:
        st.warning("Aucune donnée ne correspond aux critères sélectionnés.")