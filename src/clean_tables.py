import os
import csv
import re

PROCESSED_DIR = "../data/processed"

def clean_channel_name(name):
    # Supprime tous les chiffres inutiles
    return re.sub(r"\d+", "", name).strip()

for filename in os.listdir(PROCESSED_DIR):

    # Ne traiter que les fichiers qui commencent par "Nombre"
    if not filename.startswith("Nombre"):
        continue

    if not filename.lower().endswith(".csv"):
        continue

    file_path = os.path.join(PROCESSED_DIR, filename)
    rows = []

    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and len(row) > 0:
                row[0] = clean_channel_name(row[0])
            rows.append(row)

    # Réécriture du fichier corrigé
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"Corrigé : {filename}")
