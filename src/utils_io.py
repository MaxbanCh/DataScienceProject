import pandas as pd
from pathlib import Path

def safe_read_csv(path: Path, sep=";", encoding="utf-8", low_memory=False) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=sep, encoding=encoding, low_memory=low_memory)
    except UnicodeDecodeError:
        # fréquent sur des exports FR
        return pd.read_csv(path, sep=sep, encoding="cp1252", low_memory=low_memory)

def save_df(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
