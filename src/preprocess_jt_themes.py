import pandas as pd
from config import DATA_RAW, DATA_PROCESSED
from utils_io import save_df

RAW_FILE = next((p for p in DATA_RAW.glob("ina-barometre-jt-tv*csv")), None)
OUT_FILE = DATA_PROCESSED / "jt_themes_clean.parquet"

def main():
    if RAW_FILE is None:
        raise FileNotFoundError("Fichier ina-barometre-jt-tv introuvable dans data/raw")

    df = pd.read_csv(
        RAW_FILE,
        sep=";",
        header=None,
        encoding="cp1252",
        low_memory=False
    )

    df.columns = [
        "date",
        "chaine",
        "jt_code",       
        "theme",
        "nb_sujets",
        "duree_sec"
    ]

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df["nb_sujets"] = pd.to_numeric(df["nb_sujets"], errors="coerce")
    df["duree_sec"] = pd.to_numeric(df["duree_sec"], errors="coerce")

    df = df.dropna(subset=["date", "chaine", "theme", "nb_sujets", "duree_sec"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dow"] = df["date"].dt.dayofweek  

    df["chaine"] = df["chaine"].astype("category")
    df["theme"] = df["theme"].astype("category")

    df = df[
        ["date", "chaine", "theme", "nb_sujets", "duree_sec", "year", "month", "dow"]
    ].copy()

    save_df(df, OUT_FILE)
    print(f"[OK] Sauvé: {OUT_FILE} | shape={df.shape}")
    print("Exemples:")
    print(df.head())

if __name__ == "__main__":
    main()
