# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 23:16:07 2026

@author: Damien
Premiere étape. On ouvre un fichier .fif, on filtre et on realise une ICA. A la fin on enregistre.
"""

# =====================================
# LECTURE EEG EGI (.mff) + connectivité fonctionnelle
# =====================================

import matplotlib
matplotlib.use('Qt5Agg')  # backend interactif
import matplotlib.pyplot as plt
import mne
import numpy as np
import os


# =====================================
# LECTURE EEG (FICHIER .fif)
# =====================================
# On change l'extension ici dans le paramètre si besoin, 
# ou on s'assure que EEG_PATH pointe vers le .fif
EEG_PATH = r""

raw = mne.io.read_raw_fif(
    EEG_PATH,
    preload=True,
    verbose=True
)

# On garde le pick_types pour être sûr de n'avoir que l'EEG
raw.pick_types(eeg=True, stim=True) # stim=True pour garder les marqueurs si présents
print("\nFichier FIF chargé")
print(raw)

# Vérifier les annotations (le texte)
print(f"Nombre d'annotations : {len(raw.annotations)}")

# Vérifier les événements (le canal STI)
try:
    evs = mne.find_events(raw, verbose=False)
    print(f"Nombre d'événements trouvés dans les canaux STI : {len(evs)}")
except:
    print("Aucun canal de stimulation (STI) trouvé.")



# =====================================
# FILTRAGE PASSE-BANDE (0.5 – 40 Hz)
# =====================================
raw.filter(
    l_freq=0.5,
    h_freq=40.0,
    picks='eeg',
    method='fir',
    fir_design='firwin',
    phase='zero',
    verbose=True
)
print("Filtrage passe-bande 0.5–40 Hz terminé")

# =====================================
# FILTRE NOTCH (50 Hz)
# =====================================
raw.notch_filter(
    freqs=50,
    picks='eeg',
    method='fir',
    phase='zero',
    verbose=True
)
print("Filtre notch 50 Hz appliqué")

# =====================================
# INSPECTION VISUELLE DES CANAUX
# =====================================
plt.close('all')
raw.plot(
    n_channels=40,
    duration=10,
    scalings='auto',
    title='Inspection visuelle des canaux EEG',
    block=True
)
raw.plot_sensors(kind='topomap', show_names=False)

# =====================================
# PSD POUR REPÉRER LE BRUIT
# =====================================
plt.close('all')
psd = raw.compute_psd(fmin=0.5, fmax=40)
psd.plot()
print("Fermez la figure PSD pour continuer...")
plt.show(block=True)

# =====================================
# SAUVEGARDE DES CANAUX MAUVAIS AVANT INTERPOLATION
# =====================================

interpolated_channels = raw.info['bads'].copy()

# =====================================
# INTERPOLATION DES CANAUX MAUVAIS
# =====================================
raw.interpolate_bads(reset_bads=True, mode='accurate')

# =====================================
# RÉFÉRENCE MOYENNE
# =====================================
raw.set_eeg_reference(ref_channels='average', projection=False)
print("Référence moyenne appliquée")

plt.close('all')
raw.plot(
    n_channels=40,
    duration=10,
    scalings='auto',
    title='Vérification EEG après average reference'
)
psd = raw.compute_psd(fmin=0.5, fmax=40)
psd.plot()
print("Fermez la figure PSD post-average reference pour continuer...")
plt.show(block=True)

# =====================================
# VALIDATION AVANT ICA
# =====================================
while True:
    user_input = input(
        "\nQualité EEG validée après average reference ? "
        "(oui = continuer vers ICA / non = arrêter) : "
    ).strip().lower()
    if user_input in ['oui', 'o', 'yes', 'y']:
        print("→ Validation confirmée. Passage à l'ICA.\n")
        break
    elif user_input in ['non', 'n', 'no']:
        raise SystemExit("→ Pipeline arrêté volontairement avant ICA")
    else:
        print("Réponse invalide. Merci de répondre par oui ou non.")

# =====================================
# PRÉPARATION DES DONNÉES POUR ICA
# =====================================
raw_ica = raw.copy()
raw_ica.filter(
    l_freq=1.0,
    h_freq=None,
    picks='eeg',
    verbose=True
)
print("Données préparées pour l’ICA (filtre 1 Hz)")

# =====================================
# CALCUL DE L’ICA
# =====================================
n_channels = len(raw_ica.info['ch_names'])
n_components = min(64, n_channels - 1)

ica = mne.preprocessing.ICA(
    n_components=n_components,
    method='fastica',
    max_iter='auto',
    random_state=42
)

# décimation pour accélérer ICA
ica.fit(raw_ica, decim=8)
print("ICA calculée")

# =====================================
# DÉTECTION AUTOMATIQUE DES ARTÉFACTS (Version Numérique)
# =====================================

# 1. On définit les électrodes frontales par leurs numéros EGI
# Ces électrodes entourent l'œil et captent bien les clignements
frontal_proxy_eog = ['E37', 'E18', 'E26', 'E34', 'E21', 'E22', 'E25', 'E1'] 

# 2. On filtre pour ne garder que celles présentes dans vos 204 canaux
eog_chs = [ch for ch in frontal_proxy_eog if ch in raw.ch_names]

print(f"Canaux utilisés pour détecter les clignements : {eog_chs}")

if len(eog_chs) > 0:
    # On utilise ces canaux numériques pour trouver les composantes ICA "clignements"
    eog_inds, _ = ica.find_bads_eog(raw, ch_name=eog_chs, threshold=3.0)
    print(f"Composantes EOG détectées : {eog_inds}")
else:
    print("Attention : Aucun canal frontal trouvé pour l'EOG.")
    eog_inds = []

# 3. Détection muscle
muscle_inds, _ = ica.find_bads_muscle(raw, threshold=2.0)

# 4. Exclusion
ica.exclude = list(set(eog_inds + muscle_inds))

# =====================================
# APPLICATION DIRECTE DE L’ICA
# =====================================
raw = ica.apply(raw.copy())
print("ICA appliquée au signal EEG")

# Vérification rapide
raw.plot(n_channels=40, duration=10, scalings='auto', title='EEG après ICA')
psd = raw.compute_psd(fmin=0.5, fmax=40)
psd.plot()
import matplotlib.pyplot as plt
plt.show(block=True)

# ---------------------------
# Sauvegarde automatique du fichier post-ICA
# ---------------------------
# Récupération dossier et nom fichier d'origine
folder, filename = os.path.split(EEG_PATH)
basename, ext = os.path.splitext(filename)

# Nouveau nom avec suffixe -ICA
new_filename = f"{basename}-ICA.fif"
EEG_SAVE_PATH = os.path.join(folder, new_filename)

# Sauvegarde
raw.save(EEG_SAVE_PATH, overwrite=True)
print(f"Fichier EEG post-ICA sauvegardé : {EEG_SAVE_PATH}")

# =====================================
# SAUVEGARDE DU FICHIER VERBOSE (.txt)
# =====================================

verbose_filename = f"{basename}-verbose.txt"
VERBOSE_PATH = os.path.join(folder, verbose_filename)

with open(VERBOSE_PATH, "w", encoding="utf-8") as f:
    f.write("Prétraitement EEG LabIsCoat – Rapport automatique\n")
    f.write("=====================================\n\n")
    f.write(f"Fichier source : {filename}\n")
    f.write(f"Fichier traité : {new_filename}\n\n")
    n_annotations = len(raw.annotations)
    f.write(f"\nNombre d'annotations (marqueurs) conservées : {n_annotations}\n")
    if n_annotations > 0:
        descriptions = set(raw.annotations.description)
        f.write(f"Types de marqueurs trouvés : {', '.join(descriptions)}\n")

    if interpolated_channels:
        f.write("Canaux interpolés :\n")
        for ch in interpolated_channels:
            f.write(f" - {ch}\n")
    else:
        f.write("Aucun canal interpolé.\n")

print(f"Fichier verbose sauvegardé : {VERBOSE_PATH}")