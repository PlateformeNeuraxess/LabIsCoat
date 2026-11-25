# -*- coding: utf-8 -*-
"""
11b - Sauvegarde des epoques "temps-frequence" nettoyees - Lab Is Coat
=====================================================================

Pour la localisation de source de l'effet mu (script 18), il faut les
epoques longues (-2.5 a +2.5 s) telles qu'elles entrent dans la TFR du
script 11, mais AVANT le Laplacien de surface (CSD) : un modele de source
s'applique au potentiel en reference moyenne, pas a la CSD.

Ce script refait exactement les etapes 1 a 5 du script 11, avec les memes
parametres et les memes fonctions (copiees a l'identique) :
    1) epoquage -2.5 / +2.5 s autour des VIS, sans baseline ;
    3) correction PCA de l'artefact d'ecran a l'ouverture et a la fermeture ;
    4) rejet crete-a-crete 150 uV sur -2.0 / +1.95 s ;
    5) reechantillonnage a 250 Hz ;
puis sauvegarde les epoques. Il ne touche a aucune sortie du script 11
(tfr/*.npz, tfr_qc.csv restent tels quels).

Entree  : G:/LIC/TousapresICA/*-ICA.fif
Sortie  : outputs_tfr/epochs/<sujet>-tfr-epo.fif   (~150 Mo par sujet)
"""

import os
import glob
import warnings

import numpy as np
import mne
from scipy.signal.windows import tukey
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", message=".*does not conform to MNE naming conventions.*")

# ---------------- parametres : identiques au script 11 ----------------
RAW_ICA_DIR = r"D:/LIC/TousapresICA"
RAW_PATTERN = "*-ICA.fif"
OUTPUT_DIR = os.path.join("outputs_tfr", "epochs")

CONDITIONS = {
    "REAL_APP": ["VIS1", "VIS2"],
    "VIRT_APP": ["VIS3", "VIS4"],
    "REAL_DISG": ["VIS5", "VIS6"],
    "VIRT_DISG": ["VIS7", "VIS8"],
}
EPOCH_TMIN, EPOCH_TMAX = -2.5, 2.5
BASELINE = (-2.0, -1.5)
HID_SEARCH = (1.5, 2.5)
REJECT_WINDOW = (BASELINE[0], 1.95)
ARTIFACT_WINDOW = (-0.03, 0.05)
ARTIFACT_OFFSET_WINDOW = (-0.10, -0.03)
HID_PREFIX = "HID"
TUKEY_ALPHA = 0.5
REJECT_UV = 150.0
RESAMPLE_SFREQ = 250.0
MIN_TRIALS_PER_CONDITION = 2
# ---------------------------------------------------------------------


def subject_id(path):
    parent = os.path.dirname(os.path.abspath(path))
    if os.path.normcase(parent) != os.path.normcase(os.path.abspath(RAW_ICA_DIR)):
        return os.path.basename(parent)
    return os.path.basename(path).replace("-ICA.fif", "")


def fit_artifact_topography(raw, events, hid_ids):
    ep = mne.Epochs(raw, events, event_id=hid_ids, tmin=-0.1, tmax=0.6,
                    baseline=(-0.1, 0), preload=True, picks="eeg", verbose=False)
    if len(ep) == 0:
        return None, np.nan
    mask = (ep.times >= ARTIFACT_WINDOW[0]) & (ep.times <= ARTIFACT_WINDOW[1])
    x = ep.get_data()[:, :, mask]
    x = x.transpose(0, 2, 1).reshape(-1, x.shape[1])
    pca = PCA(n_components=1).fit(x)
    comp = pca.components_[0]
    return comp / np.linalg.norm(comp), float(pca.explained_variance_ratio_[0])


def hid_latencies(epochs, events, hid_codes, sfreq):
    out = []
    hid_samples = events[np.isin(events[:, 2], list(hid_codes)), 0]
    for s in epochs.events[:, 0]:
        d = (hid_samples - s) / sfreq
        ok = d[(d >= HID_SEARCH[0]) & (d <= HID_SEARCH[1])]
        out.append(float(ok.min()) if ok.size else None)
    return out


def remove_artifact(epochs, comp, extra_onsets=None):
    times = epochs.times
    data = epochs.get_data()
    for i in range(data.shape[0]):
        onsets = [0.0]
        if extra_onsets is not None and extra_onsets[i] is not None:
            onsets.append(extra_onsets[i])
        for t0 in onsets:
            mask = (times >= t0 + ARTIFACT_WINDOW[0]) & (times <= t0 + ARTIFACT_WINDOW[1])
            off_mask = ((times >= t0 + ARTIFACT_OFFSET_WINDOW[0])
                        & (times < t0 + ARTIFACT_OFFSET_WINDOW[1]))
            if mask.sum() < 3 or off_mask.sum() == 0:
                continue
            win = data[i][:, mask]
            offset = data[i][:, off_mask].mean(axis=1, keepdims=True)
            proj = comp @ (win - offset)
            data[i][:, mask] = win - np.outer(comp, proj * tukey(mask.sum(), alpha=TUKEY_ALPHA))
    return mne.EpochsArray(data, epochs.info, events=epochs.events, tmin=epochs.tmin,
                           event_id=epochs.event_id, verbose=False)


def reject_ptp(epochs):
    m = (epochs.times >= REJECT_WINDOW[0]) & (epochs.times <= REJECT_WINDOW[1])
    data = epochs.get_data()[:, :, m]
    ptp = (data.max(axis=2) - data.min(axis=2)).max(axis=1)
    bad = np.where(ptp > REJECT_UV * 1e-6)[0]
    if len(bad):
        epochs.drop(bad, reason="PTP")
    return epochs


os.makedirs(OUTPUT_DIR, exist_ok=True)
files = sorted(glob.glob(os.path.join(RAW_ICA_DIR, RAW_PATTERN), recursive=True))
print(len(files), "fichier(s) -ICA.fif")

for f in files:
    subject = subject_id(f)
    print(f"\n=== {subject} ===")
    raw = mne.io.read_raw_fif(f, preload=True, verbose=False)
    raw.pick("eeg")
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    all_vis = [v for vis in CONDITIONS.values() for v in vis]
    vis_ids = {k: v for k, v in event_id.items() if k in all_vis}
    hid_ids = {k: v for k, v in event_id.items() if k.startswith(HID_PREFIX)}
    if not vis_ids:
        print("   aucun VIS -> ignore")
        continue

    epochs = mne.Epochs(raw, events, event_id=vis_ids, tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
                        baseline=None, detrend=0, preload=True, reject=None,
                        reject_by_annotation=True, verbose=False)
    if hid_ids:
        comp, var = fit_artifact_topography(raw, events, hid_ids)
        if comp is not None:
            lat = hid_latencies(epochs, events, set(hid_ids.values()), raw.info["sfreq"])
            epochs = remove_artifact(epochs, comp, lat)
    epochs = reject_ptp(epochs)
    if epochs.info["sfreq"] != RESAMPLE_SFREQ:
        epochs.resample(RESAMPLE_SFREQ, verbose=False)

    counts = {c: sum(len(epochs[v]) for v in vis if v in epochs.event_id)
              for c, vis in CONDITIONS.items()}
    print("   essais :", counts)
    if min(counts.values()) < MIN_TRIALS_PER_CONDITION:
        print("   sujet incomplet -> non sauvegarde (comme dans le script 11)")
        continue
    epochs.save(os.path.join(OUTPUT_DIR, f"{subject}-tfr-epo.fif"), overwrite=True, verbose=False)
    print("   epoques sauvegardees")

print("\nTermine.")
