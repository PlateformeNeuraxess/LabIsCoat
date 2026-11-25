# -*- coding: utf-8 -*-
"""
12b - ANOVA a mesures repetees sur la puissance temps-frequence - Lab Is Coat

Pour chaque region (central, occipital) et chaque bande, ANOVA
Reality (REAL / VIRT) x Valence (APP / DISG) x Hemisphere (gauche / droit)
sur la puissance en dB de la fenetre principale (0.3-1.75 s).
C'est l'ANOVA decrite en 2.7 ("Statistics") ; les F cites en 3.3
(bande beta basse : Reality, Reality x Hemisphere) en sortent.

Entree : outputs_tfr/tfr_roi_band_windows.csv (sortie du script 12)
Sortie : outputs_tfr/tfr_anova.csv
"""

import os

import pandas as pd
from statsmodels.stats.anova import AnovaRM

IN_FILE = os.path.join("outputs_tfr", "tfr_roi_band_windows.csv")
OUT_FILE = os.path.join("outputs_tfr", "tfr_anova.csv")
WINDOW = "main"

df = pd.read_csv(IN_FILE)
df = df[df.window == WINDOW].copy()
df["region"] = df.roi.str.split().str[0]                       # Central / Occipital
df["hemisphere"] = df.roi.str.split().str[1]                   # gauche / droit
df["reality"] = df.condition.str.split("_").str[0]             # REAL / VIRT
df["valence"] = df.condition.str.split("_").str[1]             # APP / DISG

rows = []
for (region, band), d in df.groupby(["region", "band"]):
    res = AnovaRM(d, depvar="power_db", subject="subject",
                  within=["reality", "valence", "hemisphere"]).fit()
    t = res.anova_table
    for effect, r in t.iterrows():
        rows.append({"region": region, "band": band, "effect": effect,
                     "F": r["F Value"], "df_num": r["Num DF"], "df_den": r["Den DF"],
                     "p": r["Pr > F"]})

out = pd.DataFrame(rows)
out.to_csv(OUT_FILE, index=False)
pd.set_option("display.width", 160)
print(f"{df.subject.nunique()} sujets, fenetre {WINDOW}")
print(out.round(3).to_string(index=False))
print(f"\nSauvegarde : {OUT_FILE}")
