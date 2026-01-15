
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
from config import DATA_PROCESSED, RESULTS

IN_FILE = DATA_PROCESSED / "jt_themes_clean.parquet"
OUT_DIR = RESULTS / "jt_theme_time_regression_linear"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K_THEMES = 10
N_THEMES_TO_MODEL = 5
START_YEAR = 2000
END_YEAR = 2020
MAKE_PLOTS = True


def fit_theme_trend(df: pd.DataFrame, theme_name: str):
    """
    Régression linéaire simple :
    duree_annuelle = b0 + b1 * year + epsilon
    """
    d = df[df["theme"].astype(str) == str(theme_name)].copy()
    g = (
        d.groupby("year", as_index=False)["duree_sec"]
         .sum()
         .sort_values("year")
    )

    g = g[(g["year"] >= START_YEAR) & (g["year"] <= END_YEAR)].copy()

    X = sm.add_constant(g["year"].astype(float))
    y = g["duree_sec"].astype(float)

    model = sm.OLS(y, X).fit()
    return g, model


def main():
    df = pd.read_parquet(IN_FILE)
    df = df.dropna(subset=["theme", "year", "duree_sec"]).copy()
    df = df[df["duree_sec"] > 0].copy()

    # Top 10 thèmes par durée cumulée
    top = (
        df.groupby("theme")["duree_sec"]
          .sum()
          .sort_values(ascending=False)
          .head(TOP_K_THEMES)
    )

    themes = list(top.index[:N_THEMES_TO_MODEL])

    rows = []
    for th in themes:
        g, model = fit_theme_trend(df, th)

        b0 = model.params.get("const", np.nan)
        b1 = model.params.get("year", np.nan)
        pval = model.pvalues.get("year", np.nan)
        r2 = model.rsquared

        rows.append({
            "theme": str(th),
            "n_years": int(g["year"].nunique()),
            "b0_const": float(b0),
            "b1_year": float(b1),
            "pvalue_year": float(pval),
            "R2": float(r2),
        })

        # Graphique observé vs ajusté
        if MAKE_PLOTS and len(g) >= 3:
            pred = model.predict(sm.add_constant(g["year"].astype(float)))

            plt.figure()
            plt.plot(g["year"], g["duree_sec"], marker="o", label="observé")
            plt.plot(g["year"], pred, marker="o", label="ajusté (régression)")
            plt.xlabel("Année")
            plt.ylabel("Durée totale annuelle (secondes)")
            plt.title(f"Évolution temporelle — {th}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                OUT_DIR / f"fig_trend_linear_{str(th).replace(' ', '_')}.png",
                dpi=200
            )
            plt.close()

        # Résumé du modèle 
        with open(
            OUT_DIR / f"ols_summary_linear_{str(th).replace(' ', '_')}.txt",
            "w",
            encoding="utf-8"
        ) as f:
            f.write(model.summary().as_text())

    res = pd.DataFrame(rows).sort_values("pvalue_year")
    res.to_csv(OUT_DIR / "theme_time_trend_linear_regressions.csv", index=False)

    print("[OK] Résultats sauvegardés dans :", OUT_DIR)
    print(res)


if __name__ == "__main__":
    main()
