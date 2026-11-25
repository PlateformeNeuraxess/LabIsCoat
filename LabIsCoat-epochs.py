# -*- coding: utf-8 -*-
"""
Correction des artefacts VIS1-VIS8
à partir d’un fichier EPOCHS (et non raw)
"""

import mne
import os
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# PARAMÈTRES
# =====================================
EPOCHS_FIF_PATH = r""

ARTIFACT_WINDOW = (-0.075, 0.075)
REJECT_CRITERIA = dict(eeg=100e-6)

# =====================================
# CHARGEMENT DES EPOCHS
# =====================================
epochs = mne.read_epochs(EPOCHS_FIF_PATH, preload=True)
epochs.pick_types(eeg=True)

print(epochs)

folder, filename = os.path.split(EPOCHS_FIF_PATH)
basename, _ = os.path.splitext(filename)

# =====================================
# SELECTION UNIQUEMENT VIS1 à VIS8
# =====================================
vis_keys = [k for k in epochs.event_id.keys() if k.startswith("VIS")]

if len(vis_keys) == 0:
    raise ValueError("Aucun événement VIS trouvé dans les epochs.")

epochs_vis = epochs[vis_keys]
epochs_vis_avant = epochs_vis.copy()  
print("Événements VIS détectés :", vis_keys)
print(f"Nombre d'epochs avant rejet : {len(epochs_vis)}")

# =====================================
# INSPECTION MANUELLE
# =====================================
if len(epochs_vis) > 0:
    epochs_vis.plot(
        n_epochs=20,
        n_channels=40,
        scalings='auto',
        title="Inspection manuelle – epochs VIS1→VIS8",
        block=True
    )

# =====================================
# REJET AUTOMATIQUE
# =====================================

epochs_vis.drop_bad(reject=REJECT_CRITERIA)
print(f"Nombre d'epochs après rejet automatique : {len(epochs_vis)}")

epochs_corr = epochs_vis.copy()

# =====================================
# VARIANCE TEMPORELLE
# =====================================
data_corr = epochs_corr.get_data()
var_time_corr = data_corr.var(axis=1).mean(axis=0)

plt.figure(figsize=(8,4))
plt.plot(epochs_corr.times, var_time_corr)
plt.axvline(0, color='r', linestyle='--')
plt.xlabel('Temps (s)')
plt.ylabel('Variance moyenne sur canaux')
plt.title('Variance EEG (VIS1→VIS8)')
plt.show()

# =====================================
# ERP
# =====================================
if len(epochs_corr) > 0:
    evoked = epochs_corr.average()
    evoked.plot(gfp=True, time_unit='ms')
else:
    print("⚠️ Pas d'epochs restantes.")

# =====================================
# NOMBRE D'ESSAIS CONSERVÉS PAR MARQUEUR
# =====================================
txt_path = os.path.join(folder, basename + "-essais_conserves.txt")

with open(txt_path, "w", encoding="utf-8") as f:
    ligne = "\n📊 Essais conservés par marqueur :\n"
    print(ligne)
    f.write(ligne)

    for event_name in epochs_corr.event_id.keys():
        n_avant = len(epochs_vis_avant[event_name])
        n_apres = len(epochs_corr[event_name])
        pct = (n_apres / n_avant * 100) if n_avant > 0 else 0
        ligne = f"  {event_name} : {n_apres} / {n_avant}  ({pct:.1f}% conservés)\n"
        print(ligne, end="")
        f.write(ligne)

    ligne = f"\n  TOTAL : {len(epochs_corr)} / {len(epochs_vis_avant)} essais\n"
    print(ligne)
    f.write(ligne)

print(f"✅ Résultats sauvegardés : {txt_path}")

# =====================================
# SAUVEGARDE
# =====================================


save_path = os.path.join(folder, basename + "-VIS_corrige-epo.fif")
epochs_corr.save(save_path, overwrite=True)

print(f"✅ Epochs sauvegardées : {save_path}")