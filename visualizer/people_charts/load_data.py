import os
import glob
import pandas as pd
import streamlit as st

@st.cache_data
def load_people_data(data_path="../ina-api/data/INA/People"):
    """Charge et combine tous les fichiers CSV des top personnalités mentionnées."""
    all_data = []
    errors = []
    empty_files = []

    # Vérifier si le chemin existe
    if not os.path.exists(data_path):
        st.error(f"Le chemin {data_path} n'existe pas.")
        return pd.DataFrame()

    # Parcourir tous les fichiers CSV des top personnalités
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))

    if not csv_files:
        st.warning(f"Aucun fichier CSV trouvé dans {data_path}")
        return pd.DataFrame()

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)

            # Vérifier si le fichier contient des données
            if df.empty:
                empty_files.append(csv_file)
                continue

            # Vérifier que les colonnes nécessaires existent
            columns_to_keep = ["type", "value", "date", "channel"]
            missing_columns = [
                col for col in columns_to_keep if col not in df.columns
            ]

            if missing_columns:
                errors.append(
                    f"{os.path.basename(csv_file)}: colonnes manquantes {missing_columns}"
                )
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
            combined_df["date"] = pd.to_datetime(combined_df["date"])
        except Exception as e:
            st.error(f"Erreur lors de la conversion des dates: {e}")
            return pd.DataFrame()

        # Vérifier que les valeurs sont numériques
        combined_df["value"] = pd.to_numeric(combined_df["value"], errors="coerce")
        combined_df = combined_df.dropna(subset=["value"])

        return combined_df
    else:
        st.error("Aucune donnée valide n'a pu être chargée.")
        return pd.DataFrame()

def aggregate_data(df, period='daily'):
    """Agrège les données par période"""
    if df.empty:
        return df
    
    if period == 'daily':
        grouped = df.groupby(['channel', 'type', 'date']).agg({
            'value': 'sum'
        }).reset_index()

    if period == 'monthly':
        df['period'] = df['date'].dt.to_period('M')
        grouped = df.groupby(['channel', 'type', 'period']).agg({
            'value': 'sum'
        }).reset_index()
        grouped['date'] = grouped['period'].dt.start_time
        grouped = grouped.drop('period', axis=1)

    if period == 'yearly':
        df['period'] = df['date'].dt.to_period('Y')
        grouped = df.groupby(['channel', 'type', 'period']).agg({
            'value': 'sum'
        }).reset_index()
        grouped['date'] = grouped['period'].dt.start_time
        grouped = grouped.drop('period', axis=1)
    
    return grouped