# -*- coding: utf-8 -*-
"""
11a - Chronologie des marqueurs autour des stimuli VIS - Lab Is Coat

Script de diagnostic RAPIDE (ne lit que les annotations, pas le signal) à lancer
AVANT 11_tfr_extract.py pour choisir l'époque et la baseline.

Pour chaque sujet et chaque essai VIS, relève tous les autres marqueurs situés
dans [-WINDOW, +WINDOW] s autour du VIS (appui bouton, HID, VIS précédent...).

Questions auxquelles il répond :
  1) À quel délai (médian, min, max) tombent l'appui bouton, le HID, le VIS
     précédent... par rapport au VIS ?  -> où placer une baseline "au repos"
  2) Le délai appui -> VIS diffère-t-il entre conditions ?  -> s'il diffère,
     le sujet anticipe la condition (confusion possible avec l'effet du réel)
  3) Intervalle réel entre deux VIS successifs.

Sorties :
  timeline_markers_long.csv     un marqueur par ligne (sujet, essai, condition, délai)
  timeline_summary.csv          résumé par sujet et par type de marqueur
  timeline_press_by_condition.csv  délai appui -> VIS par sujet et condition
  + résumé dans la console
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import mne

warnings.filterwarnings("ignore", message=".*does not conform to MNE naming conventions.*")

# ==========================================================
# PARAMÈTRES
# ==========================================================

RAW_ICA_DIR = r"D:/LIC/TousapresICA"
RAW_PATTERN = "*-ICA.fif"
OUTPUT_DIR = "outputs_tfr"

WINDOW = 6.0          # s, de part et d'autre du VIS

CONDITIONS = {
    "REAL_APP": ["VIS1", "VIS2"],
    "VIRT_APP": ["VIS3", "VIS4"],
    "REAL_DISG": ["VIS5", "VIS6"],
    "VIRT_DISG": ["VIS7", "VIS8"],
}

# Nom du marqueur d'appui bouton. None = détection automatique : le marqueur
# non-VIS le plus fréquent dans les 2 s qui précèdent le VIS.
PRESS_MARKER = None

# ==========================================================

vis2cond = {v: c for c, vis in CONDITIONS.items() for v in vis}
os.makedirs(OUTPUT_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(RAW_ICA_DIR, RAW_PATTERN)))
print(len(files), "fichier(s) trouvé(s)\n")

rows = []
for f in files:
    subject = os.path.basename(f).replace("-ICA.fif", "")
    raw = mne.io.read_raw_fif(f, preload=False, verbose=False)
    ann = raw.annotations
    onsets = np.asarray(ann.onset)
    descs = np.array([str(d) for d in ann.description], dtype=object)

    vis_idx = [i for i, d in enumerate(descs) if d in vis2cond]
    for trial, i in enumerate(vis_idx):
        d = onsets - onsets[i]
        near = np.where((np.abs(d) <= WINDOW) & (np.arange(len(d)) != i))[0]
        for j in near:
            name = descs[j]
            if name in vis2cond:
                name = "VIS_suivant" if d[j] > 0 else "VIS_precedent"
            rows.append({
                "subject": subject, "trial": trial, "vis": descs[i],
                "condition": vis2cond[descs[i]], "marker": name,
                "marker_raw": descs[j], "delay_s": round(float(d[j]), 4),
            })

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("Aucun marqueur trouvé autour des VIS.")
df.to_csv(os.path.join(OUTPUT_DIR, "timeline_markers_long.csv"), index=False)

# --- Pour chaque essai, on ne garde que l'occurrence la plus proche de chaque type
#     de marqueur avant le VIS et après le VIS (évite de mélanger les essais voisins)
df["side"] = np.where(df.delay_s < 0, "avant", "apres")
df["absd"] = df.delay_s.abs()
nearest = df.sort_values("absd").groupby(["subject", "trial", "marker", "side"]).head(1)

summary = (nearest.groupby(["subject", "marker", "side"]).delay_s
           .agg(n="count", median="median", min="min", max="max", sd="std")
           .reset_index().round(3))
summary.to_csv(os.path.join(OUTPUT_DIR, "timeline_summary.csv"), index=False)

# --- Chronologie type (tous sujets) : médiane des médianes individuelles
print("=" * 72)
print("CHRONOLOGIE TYPE D'UN ESSAI (0 = VIS), médiane des sujets")
print("=" * 72)
glob_tl = (summary.groupby(["marker", "side"])
           .agg(n_sujets=("subject", "nunique"), median=("median", "median"),
                min=("min", "min"), max=("max", "max"))
           .reset_index().sort_values("median"))
print(glob_tl.to_string(index=False))

# --- Appui bouton
press = PRESS_MARKER
if press is None:
    cand = nearest[(nearest.side == "avant") & (nearest.delay_s >= -2.0)
                   & ~nearest.marker.str.startswith("VIS")]
    if len(cand):
        press = cand.marker.value_counts().idxmax()
        print(f"\nMarqueur d'appui bouton détecté automatiquement : '{press}'")
if press is not None:
    p = nearest[(nearest.marker == press) & (nearest.side == "avant")]
    by_cond = (p.groupby(["subject", "condition"]).delay_s.median()
               .unstack().round(3))
    by_cond.to_csv(os.path.join(OUTPUT_DIR, "timeline_press_by_condition.csv"))

    print("\nDÉLAI APPUI -> VIS PAR SUJET (s)")
    per_subj = p.groupby("subject").delay_s.agg(["median", "min", "max", "std"]).round(3)
    print(per_subj.to_string())

    # Test : le délai dépend-il de la condition ? (sujets x 4 conditions)
    print("\nDÉLAI APPUI -> VIS PAR CONDITION (médiane des sujets, s)")
    print(by_cond.median().round(3).to_string())
    try:
        from scipy.stats import friedmanchisquare
        bc = by_cond.dropna()
        if len(bc) >= 3:
            stat, pval = friedmanchisquare(*[bc[c] for c in bc.columns])
            print(f"Friedman (effet condition sur le délai) : chi2 = {stat:.2f}, "
                  f"p = {pval:.4f}, n = {len(bc)} sujets")
            print("  -> p < 0.05 : le sujet pourrait anticiper la condition, à discuter")
    except ImportError:
        pass

# --- Intervalle entre VIS successifs
ivis = nearest[(nearest.marker == "VIS_precedent")]
if len(ivis):
    print("\nINTERVALLE VIS précédent -> VIS (s) : médiane "
          f"{-ivis.delay_s.median():.2f}, min {-ivis.delay_s.max():.2f}, "
          f"max {-ivis.delay_s.min():.2f}")
    print(f"(essais sans VIS précédent dans les {WINDOW:.0f} s : premier essai de "
          "bloc ou intervalle plus long)")

print(f"\nFichiers écrits dans {OUTPUT_DIR}/ : timeline_markers_long.csv, "
      "timeline_summary.csv, timeline_press_by_condition.csv")
