import streamlit as st
import word_charts.charts as word_charts
import word_charts.thematic_analysis as thematic_analysis
# import people_charts.charts as people_charts

st.title("Projet Data Science - Les Français.es et les médias")
st.markdown(
    """ 
     Travail réalisé par :
        - Max Chateau
        - Rasim Erben
        - Myndie Ferrandez
        - Quentin Jacquot
    """
)

# Navigation principale
main_tab1, main_tab2, main_tab3 = st.tabs(["Mots", "Thématiques", "Personnalités"])

with main_tab1:
    word_charts.show_word_charts()

with main_tab2:
    thematic_analysis.show_thematic_analysis()

with main_tab3:
    st.header("Visualisation des données INA - Personnalités")
    