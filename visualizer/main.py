import streamlit as st
import word_charts.charts as word_charts
import thematic.charts as thematic_analysis
import Rapport_francais_audiovisuel.rapport_fr_tv as rapport_fr_tv
# import people_charts.charts as people_charts

st.set_page_config(layout="wide")

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
main_tab1, main_tab2, main_tab3, main_tab4, main_tab5 = st.tabs(["Mots", "Thématiques", "Personnalités", "Proportion Femmes/Hommes", "Audiences"])

with main_tab1:
    word_charts.show_word_charts()

with main_tab2:
    thematic_analysis.show_thematic_analysis()

with main_tab3:
    st.header("Visualisation des données INA - Personnalités")

with main_tab4:
    st.header("Proportion Femmes/Hommes dans les données INA")

with main_tab5:
    rapport_fr_tv.show_audience_analysis()