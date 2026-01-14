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
    """Gère les formats 02:30:00 et 2h30'"""
    if pd.isna(val) or val in ['-', 'nd', '']: return None
    s = str(val).strip()

    # Format HH:MM:SS ou HH:MM
    if ':' in s:
        try:
            parts = s.split(':')
            return int(parts[0]) * 60 + int(parts[1])
        except:
            return None

    # Format 2h30' ou 0h58'
    match = re.search(r'(\d+)h(\d+)', s)
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))
    return None


def main():
    # Configuration des chemins (à ajuster selon ton environnement local)
    path = 'data/processed'

    # 1. STRUCTURE OFFRE (Fiction)
    # Mesure l'effort de programmation des chaînes
    df_struct = pd.read_csv(
        os.path.join(path, 'Audience_de_television', 'Structure_de_loffre_TV_par_genre_de_programmes.csv'), header=1)
    df_fic = df_struct[df_struct.iloc[:, 0].astype(str).str.contains('fictions', case=False)].copy()
    y_cols = [c for c in df_fic.columns if any(char.isdigit() for char in str(c))]
    df_fic = df_fic.melt(value_vars=y_cols, var_name='annee', value_name='offre_fiction_pourcent')
    df_fic['annee'] = df_fic['annee'].str.extract(r'(\d{4})')[0]
    df_fic['offre_fiction_pourcent'] = df_fic['offre_fiction_pourcent'].apply(clean_value)
    df_fic = df_fic.groupby('annee')['offre_fiction_pourcent'].mean().reset_index()

    # 2. DURÉE ÉCOUTE PAR ÂGE (Cibles : 15-49 ans vs 50+)
    # C'est ici que se joue la démonstration "Jeunes/Actifs" vs "Seniors"
    df_age = pd.read_csv(
        os.path.join(path, 'Audience_de_television', 'Duree_decoute_quotidienne_de_la_television_selon_lage.csv'),
        header=1)
    df_age.rename(columns={df_age.columns[0]: 'annee'}, inplace=True)
    df_age['annee'] = df_age['annee'].astype(str).str.extract(r'(\d{4})')[0]

    # Extraction des deux colonnes piliers pour la régression
    df_age['duree_tv_15_49_min'] = df_age['15-49 ans'].apply(to_minutes)
    df_age['duree_tv_50_plus_min'] = df_age['50 ans et plus'].apply(to_minutes)

    # On conserve aussi les 15-24 pour tes graphiques descriptifs si besoin
    df_age['duree_tv_15_24_min'] = df_age['15-24 ans'].apply(to_minutes)

    df_age = df_age[['annee', 'duree_tv_15_49_min', 'duree_tv_50_plus_min', 'duree_tv_15_24_min']].dropna(
        subset=['annee'])

    # 3. DURÉE ÉCOUTE GLOBALE (Moyenne France)
    df_mois = pd.read_csv(
        os.path.join(path, 'Audience_de_television', 'Duree_decoute_quotidienne_de_la_television_par_mois.csv'),
        header=1)
    df_mois.rename(columns={df_mois.columns[0]: 'annee'}, inplace=True)
    df_mois['annee'] = df_mois['annee'].astype(str).str.extract(r'(\d{4})')[0]
    col_m = [c for c in df_mois.columns if 'année' in c.lower() or 'annee' in c.lower()][-1]
    df_mois['duree_tv_globale_min'] = df_mois[col_m].apply(to_minutes)
    df_mois = df_mois[['annee', 'duree_tv_globale_min']].dropna()

    # 4. CA VaD (Substitut à la TV)
    df_vad = pd.read_csv(os.path.join(path, 'Consommation_VaD',
                                      'Estimation_100%_du_chiffre_daffaire_de_la_VaD_payante_selon_le_type_dachat.csv'),
                         header=1)
    y_cols_v = [c for c in df_vad.columns if any(char.isdigit() for char in str(c))]
    df_vad = df_vad.melt(value_vars=y_cols_v, var_name='annee', value_name='ca')
    df_vad['annee'] = df_vad['annee'].str.extract(r'(\d{4})')[0]
    df_vad['ca_vad_millions'] = df_vad['ca'].apply(clean_value).divide(1000000)
    df_vad = df_vad.groupby('annee')['ca_vad_millions'].sum().reset_index()

    # 5. RATTRAPAGE (Replay)
    df_rat = pd.read_csv(
        os.path.join(path, 'Televison_de_rattrapage', 'Penetration_de_la_television_de_rattrapage.csv'))
    df_rat.columns = ['label', 'taux']
    df_rat['annee'] = df_rat['label'].astype(str).str.extract(r'(\d{4})')[0]
    df_rat['taux_replay_pourcent'] = df_rat['taux'].apply(clean_value)
    df_rat = df_rat.groupby('annee')['taux_replay_pourcent'].mean().reset_index()

    # --- FUSION FINALE ---
    master = df_fic.merge(df_age, on='annee', how='outer') \
        .merge(df_mois, on='annee', how='outer') \
        .merge(df_vad, on='annee', how='outer') \
        .merge(df_rat, on='annee', how='outer')

    # Tri et nettoyage final
    master['annee'] = master['annee'].astype(int)
    master = master.sort_values('annee').drop_duplicates('annee').reset_index(drop=True)

    # Sauvegarde
    output_file = os.path.join(path, 'master_file_annuel.csv')
    master.to_csv(output_file, index=False)

    print("✅ Master File mis à jour avec succès.")
    print(f"Période couverte : {master['annee'].min()} à {master['annee'].max()}")
    print(
        f"Colonnes pour la régression : {['duree_tv_15_49_min', 'duree_tv_50_plus_min', 'ca_vad_millions', 'offre_fiction_pourcent']}")


if __name__ == "__main__":
    main()