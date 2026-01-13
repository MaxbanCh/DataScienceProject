import pandas as pd
from config import DATA_RAW, DATA_PROCESSED
from utils_io import save_df

RAW_FILE = next((p for p in DATA_RAW.glob("*arcom*2024*xlsx")), None)
OUT_FILE = DATA_PROCESSED / "arcom2024_clean.parquet"

def main():
    if RAW_FILE is None:
        raise FileNotFoundError("Fichier ARCOM 2024 introuvable dans data/raw (*arcom*2024*xlsx).")

    df = pd.read_excel(RAW_FILE, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]

    if "CONF1_R" in df.columns:
        df["CONF1_R"] = pd.to_numeric(df["CONF1_R"], errors="coerce")

    save_df(df, OUT_FILE)
    print(f"[OK] Sauvé: {OUT_FILE} | shape={df.shape}")

if __name__ == "__main__":
    main()
