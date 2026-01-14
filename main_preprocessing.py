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


def to_minutes(time_str):
    if pd.isna(time_str) or ':' not in str(time_str): return None
    try:
        parts = str(time_str).split(':')
        return int(parts[0]) * 60 + int(parts[1])
    except:
        return None


def main():
    path = 'data/processed'

    # 1. STRUCTURE OFFRE (Fiction)
    df_struct = pd.read_csv(
        os.path.join(path, 'Audience_de_television', 'Structure_de_loffre_TV_par_genre_de_programmes.csv'), header=1)
    df_fic = df_struct[df_struct.iloc[:, 0].astype(str).str.contains('fictions', case=False)].copy()
    y_cols = [c for c in df_fic.columns if any(char.isdigit() for char in str(c))]
    df_fic = df_fic.melt(value_vars=y_cols, var_name='annee', value_name='offre_fiction_pourcent')
    df_fic['annee'] = df_fic['annee'].str.extract(r'(\d{4})')[0]
    df_fic['offre_fiction_pourcent'] = df_fic['offre_fiction_pourcent'].apply(clean_value)
    df_fic = df_fic.groupby('annee')['offre_fiction_pourcent'].mean().reset_index()

    # 2. PARTS D'AUDIENCE
    df_parts = pd.read_csv(os.path.join(path, 'Audience_de_television', 'Parts_daudience_des_chaines_de_televison.csv'),
                           header=1)
    df_parts.rename(columns={df_parts.columns[0]: 'annee'}, inplace=True)
    df_parts['annee'] = df_parts['annee'].astype(str).str.extract(r'(\d{4})')[0]
    cols_p = [c for c in df_parts.columns if c != 'annee' and 'unnamed' not in c.lower()]
    for c in cols_p: df_parts[c] = df_parts[c].apply(clean_value)
    df_parts['part_audience_moyenne'] = df_parts[cols_p].mean(axis=1)
    df_parts = df_parts[['annee', 'part_audience_moyenne']]

    # 3. DURÉE ÉCOUTE PAR ÂGE (15-49 ans)
    df_age = pd.read_csv(
        os.path.join(path, 'Audience_de_television', 'Duree_decoute_quotidienne_de_la_television_selon_lage.csv'),
        header=1)
    df_age.rename(columns={df_age.columns[0]: 'annee'}, inplace=True)
    df_age['annee'] = df_age['annee'].astype(str).str.extract(r'(\d{4})')[0]
    col_a = [c for c in df_age.columns if '15-49' in c or '25-59' in c][0]
    df_age['duree_tv_jeunes_min'] = df_age[col_a].apply(to_minutes)
    df_age = df_age.groupby('annee')['duree_tv_jeunes_min'].mean().reset_index()

    # 4. DURÉE ÉCOUTE GLOBALE (Moyenne annuelle)
    df_mois = pd.read_csv(
        os.path.join(path, 'Audience_de_television', 'Duree_decoute_quotidienne_de_la_television_par_mois.csv'),
        header=1)
    df_mois.rename(columns={df_mois.columns[0]: 'annee'}, inplace=True)
    df_mois['annee'] = df_mois['annee'].astype(str).str.extract(r'(\d{4})')[0]
    col_m = [c for c in df_mois.columns if 'année' in c.lower() or 'annee' in c.lower()][-1]
    df_mois['duree_tv_globale_min'] = df_mois[col_m].apply(to_minutes)
    df_mois = df_mois[['annee', 'duree_tv_globale_min']].dropna()

    # 5. CA VaD (Total annuel)
    df_vad = pd.read_csv(os.path.join(path, 'Consommation_VaD',
                                      'Estimation_100%_du_chiffre_daffaire_de_la_VaD_payante_selon_le_type_dachat.csv'),
                         header=1)
    y_cols_v = [c for c in df_vad.columns if any(char.isdigit() for char in str(c))]
    df_vad = df_vad.melt(value_vars=y_cols_v, var_name='annee', value_name='ca')
    df_vad['annee'] = df_vad['annee'].str.extract(r'(\d{4})')[0]
    df_vad['ca_vad_millions'] = df_vad['ca'].apply(clean_value).divide(1000000)
    df_vad = df_vad.groupby('annee')['ca_vad_millions'].sum().reset_index()

    # 6. RATTRAPAGE (REPLAY)
    df_rat = pd.read_csv(
        os.path.join(path, 'Televison_de_rattrapage', 'Penetration_de_la_television_de_rattrapage.csv'))
    df_rat.columns = ['label', 'taux']
    df_rat['annee'] = df_rat['label'].astype(str).str.extract(r'(\d{4})')[0]
    df_rat['taux_replay_pourcent'] = df_rat['taux'].apply(clean_value)
    df_rat = df_rat.groupby('annee')['taux_replay_pourcent'].mean().reset_index()

    # --- FUSION TOTALE ---
    master = df_fic.merge(df_parts, on='annee', how='outer') \
        .merge(df_age, on='annee', how='outer') \
        .merge(df_mois, on='annee', how='outer') \
        .merge(df_vad, on='annee', how='outer') \
        .merge(df_rat, on='annee', how='outer')

    master = master.dropna(subset=['annee']).drop_duplicates('annee').sort_values('annee').reset_index(drop=True)
    master.to_csv(os.path.join(path, 'master_file_annuel.csv'), index=False)
    print(f"✅ Succès ! Le fichier contient {len(master.columns)} colonnes.")


if __name__ == "__main__": main()