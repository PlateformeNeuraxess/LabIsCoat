# -*- coding: utf-8 -*-

"""
Analyse GFP en fenetres temporelles FIXES par carte de micro-etat
Lab Is Coat

Objectif :
    Tester les effets de Reality / Valence sur l'amplitude GFP
    dans une fenetre temporelle FIXE (memes bornes pour toutes
    les conditions et tous les sujets), definie a priori a partir
    de l'inspection visuelle du grand-moyennage de segmentation.

    Contrairement aux parametres par micro-etat de
    04_microstate_parameters_statistics.py (fenetre ADAPTATIVE,
    definie individuellement par le backfitting de chaque sujet/
    condition), cette analyse n'utilise PAS le resultat du
    backfitting : la fenetre est fixee une fois pour toutes,
    independamment du contraste Reality/Valence teste ensuite.
    Ca evite le "double dipping" (choisir la fenetre en regardant
    les donnees qu'on va ensuite tester dedans) et ca decorrele la
    question "y a-t-il un effet d'amplitude dans cette periode ?"
    de la question "cette carte est-elle plus/moins souvent
    assignee par le backfitting ?" (deja testee en 05).

Fenetres utilisees (a ajuster si besoin, cf MAP_WINDOWS) :
    MS1 : 150-200 ms
    MS3 : 210-280 ms
    MS4 : 210-540 ms

Entrees :
    - epochs individuels (*-epo.fif)   [PAS besoin de
      microstate_segmentation.csv ni de microstate_templates.npy]

Sorties :
    - microstate_parameters_fixed_window.csv
    - results_fixed_window.csv
    - figures/
        heatmap_statistics_fixed_window.png
        significant_effects_fixed_window.png
"""

import os
import glob

import numpy as np
import pandas as pd
import mne

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf

import config
from statsmodels.stats.multitest import multipletests

# ==========================================================
# CODAGE SOMME (-0.5 / +0.5) DES FACTEURS
# ==========================================================
#
# Avec un terme d'interaction, un codage de traitement (0/1, niveau de
# reference) fait des coefficients "Reality" et "Valence" des effets
# SIMPLES (Reality chez APP seulement, Valence chez VIRTUAL seulement).
# Le codage somme -0.5/+0.5 donne de vrais effets principaux, moyennes
# sur l'autre facteur :
#   real_c  : REAL = +0.5, VIRTUAL = -0.5  -> beta = REAL - VIRTUAL
#   disg_c  : DISG = +0.5, APP     = -0.5  -> beta = DISG - APP
#   real_c:disg_c -> (REAL: DISG-APP) - (VIRTUAL: DISG-APP)
# Les noms d'effets en sortie sont conserves (Reality[T.REAL], ...)
# pour que les figures et les CSV restent compatibles.

EFFECT_TERMS = {
    "real_c": "Reality[T.REAL]",
    "disg_c": "Valence[T.DISG]",
    "real_c:disg_c": "Reality[T.REAL]:Valence[T.DISG]",
}


def add_sum_coding(d):
    """Ajoute les colonnes real_c / disg_c (codage -0.5 / +0.5)."""
    d = d.copy()
    d["real_c"] = np.where(d["Reality"] == "REAL", 0.5, -0.5)
    d["disg_c"] = np.where(d["Valence"] == "DISG", 0.5, -0.5)
    return d



# ==========================================================
# PARAMETRES
# ==========================================================

EPOCHS_DIR = config.EPOCHS_DIR

OUTPUT_PARAMETERS = "microstate_parameters_fixed_window.csv"
OUTPUT_RESULTS = "results_fixed_window.csv"

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

TMIN = config.TMIN   # 80 ms : voir config.py (correction de l'artefact)
TMAX = config.TMAX

CONDITIONS = {
    "REAL_APP": ["VIS1", "VIS2"],
    "VIRT_APP": ["VIS3", "VIS4"],
    "REAL_DISG": ["VIS5", "VIS6"],
    "VIRT_DISG": ["VIS7", "VIS8"],
}

# Fenetres fixes par carte (en ms), definies a priori sur le
# grand-moyennage TOUTES conditions confondues (Reality x Valence)
# -> surtout ne pas les recaler en regardant REAL vs VIRT
# separement, ca reintroduirait du double dipping.
# Fenetres fixes = segments de la grande moyenne toutes conditions
# confondues (script 03, grand_average_segments.csv). Definies sans
# regarder REAL vs VIRT ni APP vs DISG -> pas de double dipping.
MAP_WINDOWS = config.load_ga_windows()
print("Fenetres fixes :", MAP_WINDOWS)

METRICS = [
    "mean_gfp_fixed_uv",
    "peak_gfp_fixed_uv",
]


# ==========================================================
# FACTEURS
# ==========================================================

def reality(x):
    return "REAL" if "REAL" in x else "VIRTUAL"


def valence(x):
    return "APP" if "APP" in x else "DISG"


# ==========================================================
# EXTRACTION GFP EN FENETRE FIXE
# ==========================================================

parameters = []

files = sorted(
    glob.glob(os.path.join(EPOCHS_DIR, "*-epo.fif"))
)

# memes canaux / meme sfreq que le pipeline micro-etats (01-05)
common_channels, target_sfreq = config.get_dataset_specs(files)

print(len(files), "sujets detectes")

for file in files:

    subject = os.path.basename(file).replace("-epo.fif", "")

    print("\nSujet :", subject)

    epochs = config.prepare_epochs(file, common_channels, target_sfreq)

    for condition, vis in CONDITIONS.items():

        try:
            ep = mne.concatenate_epochs(
                [epochs[vis[0]], epochs[vis[1]]]
            )
        except Exception:
            continue

        evoked = ep.average()
        evoked.crop(TMIN, TMAX)
        # reference moyenne : necessaire pour que sqrt(mean(V^2)) soit
        # bien le GFP (ecart-type spatial), comme dans 01-05
        evoked.set_eeg_reference("average", projection=False, verbose=False)

        data = evoked.data
        times = evoked.times * 1000  # ms

        gfp = np.sqrt(np.mean(data ** 2, axis=0))  # volts

        for microstate_name, (t_start, t_end) in MAP_WINDOWS.items():

            mask = (times >= t_start) & (times <= t_end)

            if not np.any(mask):
                print(
                    "Fenetre vide :",
                    subject, condition, microstate_name,
                )
                continue

            gfp_window = gfp[mask]
            time_window = times[mask]

            mean_gfp = np.mean(gfp_window)

            peak_index = np.argmax(gfp_window)
            peak_gfp = gfp_window[peak_index]
            peak_latency = time_window[peak_index]

            parameters.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "microstate": microstate_name,
                    "window_start_ms": t_start,
                    "window_end_ms": t_end,
                    "mean_gfp_fixed_uv": mean_gfp * 1e6,
                    "peak_gfp_fixed_uv": peak_gfp * 1e6,
                    "peak_gfp_fixed_latency_ms": peak_latency,
                }
            )

parameters_df = pd.DataFrame(parameters)

parameters_df["Reality"] = parameters_df.condition.apply(reality)
parameters_df["Valence"] = parameters_df.condition.apply(valence)
parameters_df = add_sum_coding(parameters_df)

parameters_df.to_csv(OUTPUT_PARAMETERS, index=False)

print("\nParametres en fenetre fixe sauvegardes :", OUTPUT_PARAMETERS)


# ==========================================================
# MODELES MIXTES : Reality * Valence, par carte et par metrique
# ==========================================================

results = []

for microstate_name in MAP_WINDOWS:

    data_ms = parameters_df[
        parameters_df.microstate == microstate_name
    ]

    for metric in METRICS:

        data = data_ms[
            ["subject", "real_c", "disg_c", metric]
        ].dropna()

        if len(data) < 20:
            continue

        try:
            model = smf.mixedlm(
                f"{metric} ~ real_c * disg_c",
                data,
                groups=data.subject,
            )

            fit = model.fit(reml=False)

            for term, effect in EFFECT_TERMS.items():
                if term in fit.params.index:

                    beta = fit.params[term]
                    p = fit.pvalues[term]

                    sd = data[metric].std()
                    effect_size = beta / sd if sd > 0 else np.nan

                    results.append(
                        {
                            "microstate": microstate_name,
                            "metric": metric,
                            "effect": effect,
                            "beta": beta,
                            "effect_size": effect_size,
                            "p": p,
                        }
                    )

        except Exception as e:
            print(e)

results = pd.DataFrame(results)


# ==========================================================
# FDR : par metrique, sur 3 cartes x 3 effets = 9 tests
# (famille beaucoup plus petite que dans 05, cf discussion sur
# le decoupage des familles de correction)
# ==========================================================

results["p_FDR"] = np.nan

for metric in METRICS:

    idx = results.metric == metric

    if idx.sum() > 0:
        results.loc[idx, "p_FDR"] = multipletests(
            results.loc[idx, "p"], method="fdr_bh"
        )[1]

results["significant"] = results.p_FDR < 0.05

results.to_csv(OUTPUT_RESULTS, index=False)

print("\nResultats sauvegardes :", OUTPUT_RESULTS)


# ==========================================================
# FIGURE 1 : HEATMAP STATISTIQUE (FDR)
# ==========================================================

pivot = results.pivot_table(
    index=["metric", "effect"],
    columns="microstate",
    values="p_FDR",
)
# fenetres dans l'ordre chronologique (et non alphabetique)
pivot = pivot[[w for w in MAP_WINDOWS if w in pivot.columns]]

plt.figure(figsize=(8, 6))

sns.heatmap(
    -np.log10(pivot),
    annot=True,
    cmap="viridis",
    cbar_kws={"label": "-log10(FDR)"},
)

plt.title("Fenetres fixes - Statistiques (FDR)")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTDIR, "heatmap_statistics_fixed_window.png"),
    dpi=300,
)

plt.close()


# ==========================================================
# FIGURE 2 : EFFETS SIGNIFICATIFS
# ==========================================================

sig = results[results.significant]

if len(sig) > 0:

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=sig,
        x="microstate",
        y="effect_size",
        hue="metric",
    )

    plt.axhline(0, color="black")
    plt.title("Fenetres fixes - Effets significatifs")
    plt.tight_layout()

    plt.savefig(
        os.path.join(OUTDIR, "significant_effects_fixed_window.png"),
        dpi=300,
    )

    plt.close()

print("\nTermine.")
