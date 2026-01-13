import streamlit as st
import word_charts.charts as word_charts


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

word_charts.show_word_charts()