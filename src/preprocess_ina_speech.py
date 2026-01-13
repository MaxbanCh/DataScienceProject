import pandas as pd
import numpy as np
from config import DATA_RAW, DATA_PROCESSED
from utils_io import save_df

RAW_FILE = DATA_RAW / "20190308-stats.csv"
OUT_FILE = DATA_PROCESSED / "ina_speech_clean.parquet"

def main():
    df = pd.read_csv(RAW_FILE, sep=",", encoding="utf-8", low_memory=False)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    for c in ["hour", "male_duration", "female_duration", "music_duration"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["total_speech"] = df["male_duration"].fillna(0) + df["female_duration"].fillna(0)
    df["female_share"] = np.where(df["total_speech"] > 0,
                                  df["female_duration"] / df["total_speech"],
                                  np.nan)

    for c in ["media_type", "channel_code", "channel_name", "week_day", "school_holiday_zones"]:
        df[c] = df[c].astype("category")

    save_df(df, OUT_FILE)
    print(f"[OK] Sauvé: {OUT_FILE} | shape={df.shape}")
    print("Colonnes:", df.columns.tolist())

if __name__ == "__main__":
    main()
