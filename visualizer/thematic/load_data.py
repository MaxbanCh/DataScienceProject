import os
import glob
import pandas as pd
import streamlit as st

# THEMES = {
#     "Politique": [
#         "Politique",
#         "Gouvernement",
#         "Présidence",
#         "président",
#         "Présidentielle",
#         "député",
#         "sénateur",
#         "maire",
#         "ministre",
#         "Vote",
#         "Election",
#         "Reforme",
#         "Loi",
#         "Manifestation",
#         "Grève",
#         "Revendication",
#     ],
#     "Droite": [
#         "droite",
#         "LR",
#         "Les Républicains",
#         "LREM",
#         "La République En Marche",
#         "centristes",
#     ],
#     "Extreme_Droite": [
#         "Extrême droite",
#         "FN",
#         "Front National",
#         "RN",
#         "Rassemblement National",
#         "Identitaire",
#         "Souverainiste",
#         "Nationaliste",
#         "Ultra droite",
#         "Union des droites",
#         "Insécurité",
#         "Immigration",
#         "Islamophobie",
#         "Violences policières",
#         "Violences urbaines",
#         "Délinquance",
#         "Criminalité",
#         "Identité",
#         "Islamisation",
#         "Laïcité",
#         "Séparatisme",
#         "Voile",
#         "Remigration",
#         "Grand Remplacement",
#     ],
#     "Gauche": [
#         "gauche",
#         "PS",
#         "Parti Socialiste",
#         "LFI",
#         "La France Insoumise",
#         "NUPES",
#         "NFP",
#         "Nouveau Front Populaire",
#         "EELV",
#         "Europe Écologie Les Verts",
#         "Les Écologistes",
#     ],
#     "Économie": [
#         "Economie",
#         "Économie",
#         "Finance",
#         "Budget",
#         "Impôt",
#         "Chômage",
#         "Inflation",
#         "Richesse",
#         "Pauvreté",
#         "Inégalité",
#         "Redistribution",
#     ],
#     "Social": [
#         "Santé",
#         "Hôpital",
#         "Médical",
#         "Éducation",
#         "Enseignement",
#         "Université",
#         "Vaccin",
#         "COVID",
#     ],
#     "Environnement": [
#         "Environnement",
#         "écologie",
#         "Climat",
#         "Climatique",
#         "Transition écologique",
#         "Pollution",
#     ],
#     "Sécurité & Défense": ["Sécurité", "Défense", "Armée", "Guerre", "Conflit"],
#     "International": ["Europe", "Russie", "Ukraine", "Israël", "Palestine", "Gaza"],
#     "Discriminations": [
#         "Discrimination",
#         "Racisme",
#         "Antisémitisme",
#         "Xénophobie",
#         "Homophobie",
#         "Transphobie",
#         "Sexisme",
#         "Haine",
#         "Intolérance",
#     ],
#     "Religion": ["Religion", "Islam", "Judaïsme", "Catholicisme"],
#     "LGBT+": ["LGBT", "Trans", "Transgenre", "Non-binaire"],
#     "Technologie": ["IA", "Intelligence Artificielle"],
#     "Feminisme": [
#         "Féminisme",
#         "Féminicide",
#         "Viol",
#         "Violence conjugale",
#         "Harcèlement",
#         "Consentement",
#         "Violences sexuelles",
#         "Avortement",
#         "IVG",
#         "PMA",
#     ],
# }

THEMES = {
    "Vie Institutionnelle": [
        "Politique", "Gouvernement", "Présidence", "président", "Présidentielle",
        "député", "sénateur", "maire", "ministre", "Vote", "Election", 
        "Reforme", "Loi", "Manifestation", "Grève", "Revendication", "49.3"
    ],
    "Economie Travail": [
        "Economie", "Économie", "Finance", "Budget", "Impôt", "Chômage", 
        "Inflation", "Richesse", "Pauvreté", "Inégalité", "Redistribution",
        "Pouvoir d'achat", "SMIC", "Salaire", "Dette", "Logement",
        "IA", "Intelligence Artificielle", "Technologie"
    ],
    "Securite Justice": [
        "Sécurité", "Défense", "Armée", "Police", "Justice", "Délinquance",
        "Récidive", "Execution provisoire", "Narcotrafic", "Ensauvagement",
        "Violences policières", "Violences urbaines", "Criminalité", "Haine"
    ],
    "Extrême Droite": [
        "Extrême droite", "FN", "Front National", "RN", "Rassemblement National",
        "Ultra droite", "Union des droites", "Reconquête"
    ],
    "Identité et Immigration": [
        "Immigration", "OQTF", "Frontière", "Identité", "Nationaliste", 
        "Souverainiste", "Grand Remplacement", "Remigration", "Xénophobie",
        "Identitaire"
    ],
    "Social Education": [
        "Santé", "Hôpital", "Médical", "Éducation", "Enseignement", 
        "Université", "Vaccin", "COVID", "Retraite", "Baccalauréat", "Ecole"
    ],
    "Environnement Energie": [
        "Environnement", "écologie", "Climat", "Climatique", 
        "Transition écologique", "Pollution", "Nucléaire", "Énergie"
    ],
    "Société Droits": [
        "Discrimination", "Racisme", "Antisémitisme", "Homophobie", 
        "Transphobie", "Sexisme", "Féminisme", "Féminicide", "Viol", 
        "Violence conjugale", "Harcèlement", "Consentement", 
        "Violences sexuelles", "Avortement", "IVG", "PMA", 
        "LGBT", "Trans", "Transgenre", "Non-binaire", "Woke", "Cancel culture",
        "Pride", "Lesbienne", "Gay"
    ],
    "Religions Laicite": [
        "Religion", "Islam", "Judaïsme", "Catholicisme", "Laïcité", 
        "Islamisation", "Séparatisme", "Voile", "Islamophobie"
    ],
    "International": [
        "Europe", "Russie", "Ukraine", "Israël", "Palestine", "Gaza", 
        "International", "Guerre", "Conflit", "OTAN", "USA", "Chine"
    ],
    "Partis Gauche": [
        "gauche", "PS", "Parti Socialiste", "LFI", "La France Insoumise", 
        "NUPES", "NFP", "Nouveau Front Populaire", "EELV", 
        "Europe Écologie Les Verts", "Les Écologistes"
    ],
    "Partis Droite": [
        "droite", "LR", "Les Républicains", "LREM", 
        "La République En Marche", "centristes"
    ]
}


@st.cache_data
def load_all_word_data(data_path="../ina-api/data/INA/Words"):
    """Charge et combine tous les fichiers CSV des mots"""
    all_data = []
    errors = []

    if not os.path.exists(data_path):
        st.error(f"Le chemin {data_path} n'existe pas.")
        return pd.DataFrame()

    word_folders = [
        f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))
    ]

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

                columns_to_keep = ["type", "value", "date", "word"]
                missing_columns = [
                    col for col in columns_to_keep if col not in df.columns
                ]

                if missing_columns:
                    continue

                df = df[columns_to_keep].dropna()

                if not df.empty:
                    all_data.append(df)

            except Exception:
                continue

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df["date"] = pd.to_datetime(combined_df["date"])
        combined_df["value"] = pd.to_numeric(combined_df["value"], errors="coerce")
        combined_df = combined_df.dropna(subset=["value"])
        return combined_df

    return pd.DataFrame()


def assign_theme(word):
    """Assign a theme to a word"""
    for theme, words in THEMES.items():
        if word in words:
            return theme
    return "Autre"


def create_theme_channel_matrix(df):
    """Create a matrix: channels x themes with occurrence counts"""
    df = df.copy()
    df["theme"] = df["word"].apply(assign_theme)
    df = df[df["theme"] != "Autre"]

    theme_channel = df.groupby(["type", "theme"])["value"].sum().reset_index()
    matrix = theme_channel.pivot(index="type", columns="theme", values="value").fillna(
        0
    )

    return matrix
