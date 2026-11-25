# -*- coding: utf-8 -*-

"""
Backfitting + parametres + transitions

- Backfitting (microstate_utils.backfit) : chaque instant de l'ERP
  individuel (sujet x condition) est attribue au template de plus
  forte correlation spatiale. Polarite prise en compte si
  config.IGNORE_POLARITY = False (correlation signee). Lissage de
  Pascual-Marqui et al. (1995) puis fusion des segments < 20 ms.

- Parametres par etat (memes definitions que pycrostates) : duree
  moyenne, occurrence, couverture, GEV, correlation moyenne ; plus
  l'amplitude GFP moyenne pendant l'etat (mean_gfp_uv). Le pic de GFP
  global (latence + amplitude, propriete de l'ERP entier) est calcule
  a part.

- Transitions : comptage des passages i -> j entre segments
  successifs.

Entrees :
    outputs_microstates/microstate_model.npz
    *-epo.fif (config.EPOCHS_DIR)

Sorties :
    outputs_microstates/microstate_parameters.csv
    outputs_microstates/microstate_transitions.csv
    outputs_microstates/gfp_peak_latency.csv
    outputs_microstates/microstate_segmentation.png
"""

import os
import glob

import numpy as np
import pandas as pd
import mne

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import config
import microstate_utils as msu


OUTPUT_PARAMETERS = os.path.join(config.OUTPUT_DIR, "microstate_parameters.csv")
OUTPUT_TRANSITIONS = os.path.join(config.OUTPUT_DIR, "microstate_transitions.csv")
OUTPUT_GFP_PEAK = os.path.join(config.OUTPUT_DIR, "gfp_peak_latency.csv")
OUTPUT_FIGURE = os.path.join(config.OUTPUT_DIR, "microstate_segmentation.png")
OUTPUT_GA_SEGMENTS = config.GA_SEGMENTS_FILE

MODEL_FILE = os.path.join(config.OUTPUT_DIR, "microstate_model.npz")


# ==========================================================
# CHARGEMENT DU MODELE
# ==========================================================

model = msu.load_model(MODEL_FILE)
templates = model["templates"]
cluster_names = model["names"]
n_states = len(cluster_names)
ignore_polarity = model["ignore_polarity"]

print("Micro-etats charges :", cluster_names,
      "| polarite", "ignoree" if ignore_polarity else "prise en compte")


def segment(evoked_data, sfreq):
    """Backfitting + lissage + duree minimale (parametres de config)."""
    min_segment_length = max(
        1, int(round(config.MIN_SEGMENT_DURATION_MS * sfreq / 1000))
    )
    return msu.backfit(
        evoked_data,
        templates,
        factor=config.SMOOTHING_FACTOR,
        half_window_size=config.SMOOTHING_HALF_WINDOW,
        tol=config.SMOOTHING_TOL,
        min_segment_length=min_segment_length,
        ignore_polarity=ignore_polarity,
    )


# ==========================================================
# TRAITEMENT PAR SUJET / CONDITION
# ==========================================================

parameter_rows = []
transition_rows = []
gfp_peak_rows = []
evoked_conditions = {}

files = sorted(glob.glob(os.path.join(config.EPOCHS_DIR, "*-epo.fif")))
if config.DEBUG_MAX_SUBJECTS is not None:
    files = files[: config.DEBUG_MAX_SUBJECTS]
    print(f"[mode test] limite a {len(files)} sujet(s) -> config.DEBUG_MAX_SUBJECTS")
print(len(files), "sujets")

# meme verification canaux/sfreq que dans 01_extract_gfp_peaks.py
# (voir config.get_dataset_specs) -> evite un plantage numpy plus loin
# si les sujets n'ont pas exactement la meme forme de donnees
common_channels, target_sfreq = config.get_dataset_specs(files)
print(f"Canaux communs retenus : {len(common_channels)}  |  sfreq cible : {target_sfreq} Hz")

for file in files:

    subject = os.path.basename(file).replace("-epo.fif", "")
    print("\n", subject)

    epochs = config.prepare_epochs(file, common_channels, target_sfreq)

    for condition, vis in config.CONDITIONS.items():

        try:
            ep = mne.concatenate_epochs([epochs[vis[0]], epochs[vis[1]]])
        except KeyError:
            continue

        evoked = ep.average()
        evoked.crop(config.TMIN, config.TMAX)
        evoked.set_eeg_reference("average", projection=False, verbose=False)

        evoked_conditions.setdefault(condition, []).append(evoked)

        # ----------------------------------------------
        # BACKFIT
        # ----------------------------------------------

        sfreq = evoked.info["sfreq"]
        seg_labels = segment(evoked.data, sfreq)

        # ----------------------------------------------
        # PARAMETRES
        # ----------------------------------------------

        params = msu.compute_parameters(
            evoked.data, seg_labels, templates, sfreq, ignore_polarity
        )

        # GFP reel (uV), echantillon par echantillon -> amplitude GFP
        # moyenne pendant chaque etat, et pic de GFP global.
        gfp_uv = np.sqrt(np.mean(evoked.data**2, axis=0)) * 1e6

        for state_idx, state in enumerate(cluster_names):
            state_mask = seg_labels == state_idx
            mean_gfp_uv = gfp_uv[state_mask].mean() if state_mask.any() else np.nan
            p = params[state_idx]
            parameter_rows.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "microstate": state,
                    "mean_corr": p["mean_corr"],
                    "GEV": p["gev"],
                    "coverage_percent": p["timecov"] * 100,
                    "mean_duration_ms": p["meandurs"] * 1000,
                    "occurrence_per_sec": p["occurrences"],
                    "mean_gfp_uv": mean_gfp_uv,
                }
            )

        # ----------------------------------------------
        # PIC DE GFP GLOBAL (latence + amplitude) : une seule
        # valeur par sujet/condition (propriete de l'ERP entier,
        # pas d'un micro-etat en particulier) -> fichier separe.
        # ----------------------------------------------

        peak_idx = int(np.argmax(gfp_uv))
        gfp_peak_rows.append(
            {
                "subject": subject,
                "condition": condition,
                "peak_gfp_latency_ms": evoked.times[peak_idx] * 1000,
                "peak_gfp_uv": gfp_uv[peak_idx],
            }
        )

        # ----------------------------------------------
        # TRANSITIONS (comptages bruts -> stats en 04)
        # ----------------------------------------------

        T_counts = msu.transition_counts(seg_labels, n_states)

        for i, from_state in enumerate(cluster_names):
            for j, to_state in enumerate(cluster_names):
                if i == j:
                    continue
                transition_rows.append(
                    {
                        "subject": subject,
                        "condition": condition,
                        "from": from_state,
                        "to": to_state,
                        "count": int(T_counts[i, j]),
                    }
                )


parameters_df = pd.DataFrame(parameter_rows)
parameters_df.to_csv(OUTPUT_PARAMETERS, index=False)

transitions_df = pd.DataFrame(transition_rows)
transitions_df.to_csv(OUTPUT_TRANSITIONS, index=False)

gfp_peak_df = pd.DataFrame(gfp_peak_rows)
gfp_peak_df.to_csv(OUTPUT_GFP_PEAK, index=False)

print("\nFichiers crees :")
print(" ", OUTPUT_PARAMETERS)
print(" ", OUTPUT_TRANSITIONS)
print(" ", OUTPUT_GFP_PEAK)


# ==========================================================
# FIGURE : GFP REEL (pas un proxy) + SEGMENTATION MOYENNE
# ==========================================================

condition_order = list(config.CONDITIONS.keys())
colors = plt.cm.tab10(np.arange(n_states))

# panneaux : un par condition, + un dernier qui combine TOUS les
# sujets et TOUTES les conditions (demande explicite : vue d'ensemble
# du GFP moyen et de la segmentation, tous fichiers confondus)
all_evoked_pooled = [e for evs in evoked_conditions.values() for e in evs]
panels = [(cond, evoked_conditions.get(cond, [])) for cond in condition_order]
panels.append(("TOUTES CONDITIONS (grand-average)", all_evoked_pooled))

def _contiguous_runs(labels):
    """Renvoie une liste de (start_idx, end_idx_inclus, label) pour
    chaque segment contigu de meme etat."""
    runs = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start]:
            runs.append((start, i - 1, labels[start]))
            start = i
    runs.append((start, len(labels) - 1, labels[start]))
    return runs


# panneaux empiles verticalement (1 colonne), axe temporel partage
# -> comparaison directe des latences d'une condition a l'autre.
# Plus de bandeau separe : le remplissage colore de la courbe suffit
# a montrer la segmentation, et chaque panneau garde son propre axe
# des temps (graduations + grille) pour lire les latences directement.
n_panels = len(panels)
fig = plt.figure(figsize=(9, 1.8 * n_panels))
gs = gridspec.GridSpec(n_panels, 1, hspace=0.35)

shared_ax = None
gfp_max = 0.0

# pre-calcul pour fixer une echelle Y commune -> comparaison honnete
precomputed = []
for label, evs in panels:
    if not evs:
        precomputed.append(None)
        continue
    mean_data = np.mean([e.data for e in evs], axis=0)
    gfp = np.sqrt(np.mean(mean_data**2, axis=0)) * 1e6
    times = evs[0].times * 1000
    gfp_max = max(gfp_max, gfp.max())

    seg_mean = segment(mean_data, evs[0].info["sfreq"])
    precomputed.append((times, gfp, seg_mean))

# ----------------------------------------------------------
# Segments de la grande moyenne TOUTES CONDITIONS confondues
# -> fenetres fixes des scripts 07 / 08 / 10. Definies sans regarder
# les contrastes Reality / Valence (orthogonal au test -> pas de
# double dipping).
# ----------------------------------------------------------
if precomputed[-1] is not None:
    ga_times, _, ga_seg = precomputed[-1]
    ga_rows = []
    start = 0
    for i in range(1, len(ga_seg) + 1):
        if i == len(ga_seg) or ga_seg[i] != ga_seg[start]:
            ga_rows.append(
                {
                    "microstate": cluster_names[ga_seg[start]],
                    "start_ms": round(float(ga_times[start]), 1),
                    "end_ms": round(float(ga_times[i - 1]), 1),
                }
            )
            start = i
    pd.DataFrame(ga_rows).to_csv(OUTPUT_GA_SEGMENTS, index=False)
    print("\nSegments de la grande moyenne (fenetres fixes 07/08/10) :")
    for r in ga_rows:
        print(f"  {r['microstate']} : {r['start_ms']:.0f}-{r['end_ms']:.0f} ms")
    print(" ", OUTPUT_GA_SEGMENTS)

for i, ((label, evs), pre) in enumerate(zip(panels, precomputed)):

    if pre is None:
        continue

    times, gfp, seg_labels = pre
    runs = _contiguous_runs(seg_labels)

    # --- GFP avec l'interieur colore par etat ---
    ax = fig.add_subplot(gs[i, 0], sharex=shared_ax)
    if shared_ax is None:
        shared_ax = ax

    for start, end, lab in runs:
        color = colors[lab] if lab >= 0 else (0.8, 0.8, 0.8, 1.0)
        # +1 pour inclure le point de jonction avec le segment suivant
        # (evite un mini-trou visuel a chaque frontiere de segment)
        stop = min(end + 2, len(times))
        ax.fill_between(
            times[start:stop], 0, gfp[start:stop], color=color, alpha=0.75, linewidth=0
        )

    ax.plot(times, gfp, color="black", linewidth=1)
    ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
    ax.set_ylim(0, gfp_max * 1.05)
    ax.set_xlim(times[0], times[-1])

    # graduations + grille verticale tous les 100 ms, visibles sur
    # chaque panneau pour lire les latences sans avoir a se referer
    # a un seul axe partage en bas de la figure
    ax.xaxis.set_major_locator(plt.MultipleLocator(100))
    ax.grid(axis="x", color="black", alpha=0.15, linewidth=0.8)
    ax.tick_params(labelbottom=True, labelsize=7)
    ax.set_xlabel("ms", fontsize=7)

# legende des couleurs -> etats
handles = [
    plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(n_states)
]
fig.legend(handles, cluster_names, loc="lower center", ncol=n_states, fontsize=8)

plt.suptitle("GFP reel et segmentation micro-etats (par condition + toutes conditions confondues)")
plt.tight_layout(rect=[0.08, 0.05, 1, 0.97])
plt.savefig(OUTPUT_FIGURE, dpi=300)
plt.close()

print(" ", OUTPUT_FIGURE)
print("\nTermine")
