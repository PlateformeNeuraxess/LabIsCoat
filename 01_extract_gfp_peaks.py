# -*- coding: utf-8 -*-

"""
Extraction des cartes GFP peak pour le clustering micro-etats


Ce que ce script corrige par rapport a la version precedente :

- Reference moyenne (average reference) appliquee explicitement avant
  le calcul du GFP. La normalisation manuelle "topo - mean(topo)" de
  l'ancienne version n'est pas equivalente a un vrai re-referencement
  EEG (elle ignore par ex. les canaux exclus du pick, et n'est pas la
  convention utilisee par pycrostates / Cartool).
- Les cartes ne sont plus normalisees "a la main" avant clustering :
  le k-means (microstate_utils.fit_kmeans) s'en charge, et la polarite
  est conservee (voir 02_cluster_microstates.py). Ici on ne fait qu'extraire les pics GFP
  sur l'ERP moyen par sujet/condition, tel que dans la version
  originale (design "micro-etats sur ERP", coherent avec un
  paradigme visuel verrouille au stimulus).

Sorties :
    outputs_microstates/gfp_peak_maps.npy       (n_peaks, n_channels)
    outputs_microstates/gfp_peak_metadata.csv
    outputs_microstates/reference_info.fif       (infos capteurs, reutilisees
                                                    par les scripts suivants)
"""

import os
import glob

import numpy as np
import pandas as pd
import mne

from scipy.signal import find_peaks

import config


os.makedirs(config.OUTPUT_DIR, exist_ok=True)

OUTPUT_MAPS = os.path.join(config.OUTPUT_DIR, "gfp_peak_maps.npy")
OUTPUT_METADATA = os.path.join(config.OUTPUT_DIR, "gfp_peak_metadata.csv")
OUTPUT_INFO = os.path.join(config.OUTPUT_DIR, "reference_info.fif")


def extract_gfp_peaks(evoked, n_peaks, min_distance):
    """Renvoie les indices temporels des n_peaks plus grands maxima locaux du GFP."""

    data = evoked.data

    gfp = np.sqrt(np.mean(data**2, axis=0))

    peaks, _ = find_peaks(gfp, distance=min_distance)

    if len(peaks) == 0:
        return np.array([], dtype=int)

    # tri par amplitude GFP decroissante, on garde les meilleurs
    order = np.argsort(gfp[peaks])[::-1]
    peaks = peaks[order][:n_peaks]

    return peaks


# ==========================================================
# BOUCLE SUJETS / CONDITIONS
# ==========================================================

all_maps = []
metadata = []
reference_info = None

files = sorted(glob.glob(os.path.join(config.EPOCHS_DIR, "*-epo.fif")))
if config.DEBUG_MAX_SUBJECTS is not None:
    files = files[: config.DEBUG_MAX_SUBJECTS]
    print(f"[mode test] limite a {len(files)} sujet(s) -> config.DEBUG_MAX_SUBJECTS")

print(len(files), "sujets detectes")

if len(files) == 0:
    raise FileNotFoundError(
        f"Aucun fichier *-epo.fif trouve dans config.EPOCHS_DIR = "
        f"'{config.EPOCHS_DIR}'. Verifie ce chemin dans config.py."
    )

# --------------------------------------------------------------
# Verification prealable : tous les sujets doivent avoir le meme
# nombre de canaux EEG et la meme frequence d'echantillonnage pour
# pouvoir empiler leurs cartes ensemble. Fonction partagee avec les
# scripts 03 et 05 (voir config.py) pour appliquer la meme regle
# partout plutot que de planter plus loin avec une erreur numpy.
# --------------------------------------------------------------

common_channels, target_sfreq = config.get_dataset_specs(files)
print(f"Canaux communs retenus : {len(common_channels)}  |  sfreq cible : {target_sfreq} Hz")

for file in files:

    subject = os.path.basename(file).replace("-epo.fif", "")
    print("\nSujet :", subject)

    epochs = config.prepare_epochs(file, common_channels, target_sfreq)

    for condition, vis in config.CONDITIONS.items():

        try:
            ep = mne.concatenate_epochs([epochs[vis[0]], epochs[vis[1]]])
        except KeyError:
            print(" ", condition, ": condition absente, sujet ignore pour cette condition")
            continue

        evoked = ep.average()
        evoked.crop(config.TMIN, config.TMAX)

        # reference moyenne explicite -> requis par la
        # litterature micro-etats (Michel & Koenig, 2018)
        evoked.set_eeg_reference("average", projection=False, verbose=False)

        if reference_info is None:
            reference_info = evoked.info.copy()

        peaks = extract_gfp_peaks(
            evoked,
            n_peaks=config.N_PEAKS_PER_CONDITION,
            min_distance=config.MIN_PEAK_DISTANCE,
        )

        gfp = np.sqrt(np.mean(evoked.data**2, axis=0))

        for p in peaks:

            # carte brute, polarite conservee -> pas de normalisation
            # manuelle ici, le k-means la gere au clustering
            topo = evoked.data[:, p]

            all_maps.append(topo)

            metadata.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "time_ms": evoked.times[p] * 1000,
                    "gfp_uv": gfp[p] * 1e6,
                }
            )

maps = np.array(all_maps)
metadata = pd.DataFrame(metadata)

print("\nCartes extraites :", maps.shape)

np.save(OUTPUT_MAPS, maps)
metadata.to_csv(OUTPUT_METADATA, index=False)
mne.io.write_info(OUTPUT_INFO, reference_info, overwrite=True)

print("\nFichiers crees :")
print(" ", OUTPUT_MAPS)
print(" ", OUTPUT_METADATA)
print(" ", OUTPUT_INFO)
