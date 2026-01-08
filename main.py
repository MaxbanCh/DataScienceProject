from babel.messages.extract import extract

from src.data_loader import DataLoader
import os

def main():

    """
    Uniquement ces data a changé pour chaque tableau
    """

    input_file = 'data/raw/Télévision de rattrapage.xlsx'
    output_file = 'Consommation_de_télévision_de_rattrapage.csv'
    sheet_name = 'Consommation'
    first_row = 5
    last_row = 150


    output_dir = 'data/processed'
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

    # On sauvegarde le fichier
    save_path = os.path.join(output_dir, output_file)
    df_file.to_csv(save_path, index=False)

if __name__ == '__main__':
    main()

