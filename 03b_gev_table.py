# -*- coding: utf-8 -*-
"""
03b - GEV de chaque carte (Tableau 2) - Lab Is Coat

Le script 02 ne donne que la GEV globale des 5 cartes aux pics de GFP
(73.8 %). Le Tableau 2 donne la GEV de CHAQUE carte dans les ERP
individuels : elle est dans la colonne GEV de microstate_parameters.csv
(script 03, une ligne par sujet x condition x microetat).

Calcul : moyenne des 4 conditions pour chaque sujet, puis moyenne et
ecart-type entre sujets. La somme des 5 cartes = GEV totale du
back-fitting dans les ERP individuels.

Entree : outputs_microstates/microstate_parameters.csv
"""

import os
import pandas as pd

IN_FILE = os.path.join("outputs_microstates", "microstate_parameters.csv")
EXCLUDED = []          # meme liste que le script 04 si des sujets y sont exclus

df = pd.read_csv(IN_FILE)
df = df[~df.subject.isin(EXCLUDED)]

per_subject = df.groupby(["subject", "microstate"]).GEV.mean().unstack()
print(f"{len(per_subject)} sujets\n")
print("GEV par carte (moyenne des 4 conditions, puis entre sujets) :")
for ms in ["MS3", "MS1", "MS4", "MS5", "MS2"]:
    g = per_subject[ms]
    print(f"  {ms} : {g.mean():.3f} ± {g.std(ddof=1):.3f}  ({100 * g.mean():.1f} %)")
tot = per_subject.sum(axis=1)
print(f"  Total : {tot.mean():.3f} ± {tot.std(ddof=1):.3f}")
