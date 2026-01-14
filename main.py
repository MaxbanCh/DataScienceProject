from babel.messages.extract import extract

from src.data_loader import DataLoader
import os

def main():

    """
    Uniquement ces data a changé pour chaque tableau
    """

    input_file = 'data/raw/Consommation VàD - données statistiques.xlsx'
    sheet_name = 'CA 100% Support'
    first_row = 6
    last_row = 11
    table_name = "Estimation 100% du chiffre d'affaires de la VàD payante selon le support d'achat"
    output_dir = 'data/processed/Consommation_VaD'


    # On s'assure que le dossier existe
    os.makedirs(output_dir, exist_ok=True)

    # Init de l'extracteur
    extractor = DataLoader(input_file)

    # On extrait le tableau donné de la page donnée
    df_file = extractor.extract_sheets(sheet_name, skip_rows=first_row)
    df_file = df_file.head(last_row - first_row)

    # On retire les colonnes et lignes vides
    df_file = df_file.dropna(axis=0, how='all')
    df_file = df_file.dropna(axis=1, how='all')

    # On sauvegarde le fichier avec un nom normalisé
    output_file = (
        str(table_name).strip()
        .replace(' ', '_')
        .replace("'", '')
        .replace('é', 'e')
        .replace('è', 'e')
        .replace('à', 'a')
        .replace('(', '_')
        .replace(')', '_')
    )
    save_path = os.path.join(output_dir, f"{output_file}.csv")
    df_file.to_csv(save_path, index=False)

if __name__ == '__main__':
    main()

