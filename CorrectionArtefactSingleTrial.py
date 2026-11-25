# -*- coding: utf-8 -*-
"""
Correction single-trial VIS par PCA des epochs HID
Fenêtre -30 à +50 ms
+ test statistique disparition artefact
"""

import mne
import numpy as np
import os
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA

# =====================================
# PARAMÈTRES
# =====================================
EEG_FIF_PATH = r""

EPOCH_TMIN = -0.1
EPOCH_TMAX = 0.6
BASELINE = (-0.1, 0)
ARTIFACT_WINDOW = (-0.03, 0.05)

# =====================================
# CHARGEMENT
# =====================================
raw = mne.io.read_raw_fif(EEG_FIF_PATH, preload=True)
raw.pick_types(eeg=True)

events, event_id = mne.events_from_annotations(raw)

vis_event_ids = {k: v for k, v in event_id.items() if k.startswith('VIS')}
hid_event_ids = {k: v for k, v in event_id.items() if k.startswith('HID')}

epochs_VIS = mne.Epochs(
    raw, events, event_id=vis_event_ids,
    tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
    baseline=BASELINE,
    preload=True,
    picks='eeg'
)

epochs_HID = mne.Epochs(
    raw, events, event_id=hid_event_ids,
    tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
    baseline=BASELINE,
    preload=True,
    picks='eeg'
)

print(f"N VIS: {len(epochs_VIS)} | N HID: {len(epochs_HID)}")

if len(epochs_VIS) == 0 or len(epochs_HID) == 0:
    raise RuntimeError("VIS ou HID vide.")

# =====================================
# FENÊTRE ARTEFACT
# =====================================
times = epochs_VIS.times
mask = (times >= ARTIFACT_WINDOW[0]) & (times <= ARTIFACT_WINDOW[1])

# =====================================
# CONSTRUCTION MATRICE HID POUR PCA
# =====================================
hid_data = epochs_HID.get_data()[:, :, mask]  # (n_trials, n_channels, n_times)

n_trials, n_channels, n_times = hid_data.shape

# On concatène essais et temps
hid_reshaped = hid_data.transpose(0, 2, 1).reshape(-1, n_channels)

# PCA
pca = PCA(n_components=1)
pca.fit(hid_reshaped)

artefact_component = pca.components_[0]  # topographie spatiale dominante
artefact_component /= np.linalg.norm(artefact_component)

print("Variance expliquée par composante 1 :", pca.explained_variance_ratio_[0])

# =====================================
# CORRECTION VIS
# =====================================
epochs_corr = epochs_VIS.copy()
vis_data = epochs_corr.get_data()

for i in range(len(epochs_corr)):

    trial = vis_data[i]

    # projection dans fenêtre artefact
    trial_window = trial[:, mask]

    # projection spatiale
    proj = artefact_component @ trial_window

    # reconstruction artefact
    reconstruction = np.outer(artefact_component, proj)

    # soustraction seulement dans fenêtre
    trial[:, mask] -= reconstruction

epochs_corr._data = vis_data

print("Correction PCA terminée.")

# =====================================
# TEST STATISTIQUE
# =====================================
def compute_rms(epochs):
    data = epochs.get_data()[:, :, mask]
    rms = np.sqrt(np.mean(data**2, axis=2))
    return rms.mean(axis=1)

rms_before = compute_rms(epochs_VIS)
rms_after  = compute_rms(epochs_corr)

tval, pval = ttest_rel(rms_before, rms_after)

print("\n===== TEST DISPARITION ARTEFACT =====")
print(f"T = {tval:.3f}")
print(f"p = {pval:.10f}")
print(f"RMS avant : {np.mean(rms_before):.6e}")
print(f"RMS après : {np.mean(rms_after):.6e}")

# =====================================
# COMPARAISON ERP
# =====================================
ev_before = epochs_VIS.average()
ev_after  = epochs_corr.average()

mne.viz.plot_compare_evokeds(
    dict(Avant=ev_before, Apres=ev_after),
    combine="mean",
    picks="eeg"
)

# =====================================
# SAUVEGARDE
# =====================================
folder, filename = os.path.split(EEG_FIF_PATH)
basename, _ = os.path.splitext(filename)

save_path = os.path.join(folder, basename + "-VIS_PCA_corr-epo.fif")
epochs_corr.save(save_path, overwrite=True)

print(f"\nEpochs corrigées sauvegardées : {save_path}")