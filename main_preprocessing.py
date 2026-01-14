import pandas as pd
import os
import re


def clean_value(val):
    if pd.isna(val) or str(val).lower() in ['-', 'nd', 'nc', '']: return None
    val = str(val).replace(',', '.').replace(' ', '').replace('\xa0', '')
    val = re.sub(r'[^\d.]', '', val)
    try:
        return float(val) if val else None
    except:
        return None


def to_minutes(val):
    if pd.isna(val) or val in ['-', 'nd', '']: return None
    s = str(val).strip()
    if ':' in s:
        try:
            parts = s.split(':')
            return int(parts[0]) * 60 + int(parts[1])
        except:
            return None
    match = re.search(r'(\d+)h(\d+)', s)
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))
    return None


def fix_year(df):
    """Force le nettoyage de la colonne année et la convertit en entier."""
    df = df.copy()
    df.rename(columns={df.columns[0]: 'annee'}, inplace=True)
    df['annee'] = df['annee'].astype(str).str.extract(r'(\d{4})')[0]
    df = df.dropna(subset=['annee'])
    df['annee'] = df['annee'].astype(int)
    return df


def main():
    path = 'data/processed'

    # --- 1. STRUCTURE OFFRE (Fiction & Info) ---
    path_off = os.path.join(path, 'Audience_de_television', 'Structure_de_loffre_TV_par_genre_de_programmes.csv')
    df_struct = pd.read_csv(path_off, header=1)
    y_cols = [c for c in df_struct.columns if any(char.isdigit() for char in str(c))]

    # Fiction
    df_fic = df_struct[df_struct.iloc[:, 0].astype(str).str.contains('fictions', case=False)].copy()
    df_fic_m = df_fic.melt(value_vars=y_cols, var_name='annee', value_name='offre_fiction_pourcent')
    df_fic_m = fix_year(df_fic_m)
    df_fic_m['offre_fiction_pourcent'] = df_fic_m['offre_fiction_pourcent'].apply(clean_value)
    df_fic_final = df_fic_m.groupby('annee')['offre_fiction_pourcent'].mean().reset_index()

    # Information (On s'assure de bien prendre la ligne "Information")
    df_info = df_struct[df_struct.iloc[:, 0].astype(str).str.contains('journaux télévisés', case=False)].copy()
    df_info_m = df_info.melt(value_vars=y_cols, var_name='annee', value_name='offre_info_pourcent')
    df_info_m = fix_year(df_info_m)
    df_info_m['offre_info_pourcent'] = df_info_m['offre_info_pourcent'].apply(clean_value)
    df_info_final = df_info_m.groupby('annee')['offre_info_pourcent'].mean().reset_index()

    # --- 2. PARTS D'AUDIENCE (Somme des chaînes TNT/Thématiques) ---
    path_aud = os.path.join(path, 'Audience_de_television', 'Parts_daudience_des_chaines_de_televison.csv')
    df_aud = pd.read_csv(path_aud, header=1)
    df_aud = fix_year(df_aud)

    # On identifie les chaînes historiques à exclure de la somme "Thématiques"
    hist_cols = ['TF1', 'France 2', 'France 3', 'Canal+', 'La 5', 'France 5', 'M6 ', 'Arte']
    # Toutes les autres colonnes (C8, W9, BFM, etc.) sont considérées comme "Thématiques/TNT"
    them_cols = [c for c in df_aud.columns if c not in hist_cols and c != 'annee']

    # On nettoie et on somme
    df_aud_clean = df_aud.copy()
    for c in them_cols:
        df_aud_clean[c] = df_aud_clean[c].apply(clean_value).fillna(0)

    df_aud_clean['pda_thematiques'] = df_aud_clean[them_cols].sum(axis=1)
    # Si la somme est 0 (ex: avant 2005), on remet None pour la régression
    df_aud_clean.loc[df_aud_clean['pda_thematiques'] == 0, 'pda_thematiques'] = None
    df_pda_final = df_aud_clean[['annee', 'pda_thematiques']]

    # --- 3. DURÉE ÉCOUTE PAR ÂGE ---
    path_age = os.path.join(path, 'Audience_de_television', 'Duree_decoute_quotidienne_de_la_television_selon_lage.csv')
    df_age = pd.read_csv(path_age, header=1)
    df_age = fix_year(df_age)
    df_age['duree_tv_15_49_min'] = df_age['15-49 ans'].apply(to_minutes)
    df_age['duree_tv_50_plus_min'] = df_age['50 ans et plus'].apply(to_minutes)
    df_age['duree_tv_15_24_min'] = df_age['15-24 ans'].apply(to_minutes)
    df_age_final = df_age[['annee', 'duree_tv_15_49_min', 'duree_tv_50_plus_min', 'duree_tv_15_24_min']]

    # --- 4. DURÉE ÉCOUTE GLOBALE ---
    path_mois = os.path.join(path, 'Audience_de_television', 'Duree_decoute_quotidienne_de_la_television_par_mois.csv')
    df_glob = pd.read_csv(path_mois, header=1)
    df_glob = fix_year(df_glob)
    col_m = [c for c in df_glob.columns if 'année' in c.lower() or 'annee' in c.lower()][-1]
    df_glob['duree_tv_globale_min'] = df_glob[col_m].apply(to_minutes)
    df_glob_final = df_glob[['annee', 'duree_tv_globale_min']]

    # --- 5. CA VAD ---
    path_vad = os.path.join(path, 'Consommation_VaD',
                            'Estimation_100%_du_chiffre_daffaire_de_la_VaD_payante_selon_le_type_dachat.csv')
    df_vad = pd.read_csv(path_vad, header=1)
    y_cols_v = [c for c in df_vad.columns if any(char.isdigit() for char in str(c))]
    df_vad_m = df_vad.melt(value_vars=y_cols_v, var_name='annee', value_name='ca')
    df_vad_m = fix_year(df_vad_m)
    df_vad_m['ca_vad_millions'] = df_vad_m['ca'].apply(clean_value).divide(1000000)
    df_vad_final = df_vad_m.groupby('annee')['ca_vad_millions'].sum().reset_index()

    # --- FUSION FINALE ---
    master = pd.DataFrame({'annee': range(1990, 2025)})
    master = master.merge(df_fic_final, on='annee', how='left') \
        .merge(df_info_final, on='annee', how='left') \
        .merge(df_pda_final, on='annee', how='left') \
        .merge(df_age_final, on='annee', how='left') \
        .merge(df_glob_final, on='annee', how='left') \
        .merge(df_vad_final, on='annee', how='left')

    master = master.dropna(thresh=2).sort_values('annee').reset_index(drop=True)

    output_file = os.path.join(path, 'master_file_annuel.csv')
    master.to_csv(output_file, index=False)

    print(f"✅ Master File mis à jour avec succès.")
    print("\n--- Diagnostic des données (Valeurs non-nulles) ---")
    print(master.notna().sum())


if __name__ == "__main__":
    main()