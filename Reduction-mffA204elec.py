# -*- coding: utf-8 -*-
"""
Sélection et renommage des 204 canaux EEG EGI, affichage 2D et sauvegarde
"""

import mne
import os
import numpy as np

# =====================================
# CHEMIN DU FICHIER
# =====================================
EEG_PATH = r""

# =====================================
# Dictionnaire de correspondance EGI
# =====================================
egi_lookup = {
    'Fp1': 'E37', 'FP1': 'E37', 'Fp2': 'E18', 'FP2': 'E18', 'Fpz': 'E26', 'FPz': 'E26',
    'Fz': 'E21', 'Cz': 'E80', 'Pz': 'E101', 'Oz': 'E126', 'FCz': 'E15', 'CPz': 'E90', 'POz': 'E119',
    'F3': 'E36', 'F4': 'E224', 'C3': 'E59', 'C4': 'E183', 'P3': 'E87', 'P4': 'E153',
    'F7': 'E47', 'F8': 'E2', 'T7': 'E69', 'T8': 'E202', 'P7': 'E96', 'P8': 'E170',
    'AF3': 'E34', 'AF4': 'E12', 'AF7': 'E46', 'AF8': 'E10', 'FC1': 'E24', 'FC2': 'E207',
    'FC3': 'E42', 'FC4': 'E206', 'FC5': 'E49', 'FC6': 'E213', 'CP1': 'E79', 'CP2': 'E143',
    'CP3': 'E66', 'CP4': 'E164', 'CP5': 'E76', 'CP6': 'E172', 'PO3': 'E109', 'PO4': 'E140',
    'PO7': 'E97', 'PO8': 'E161', 'O1': 'E116', 'O2': 'E150', 'F1': 'E29', 'F2': 'E5',
    'F5': 'E48', 'F6': 'E222', 'C1': 'E44', 'C2': 'E185', 'C5': 'E64', 'C6': 'E194',
    'P1': 'E88', 'P2': 'E142', 'P5': 'E86', 'P6': 'E162', 'P9': 'E106', 'P10': 'E169',
    'T9': 'E67', 'T10': 'E210', 'FT7': 'E62', 'FT8': 'E211', 'TP7': 'E84', 'TP8': 'E179',
    'TP9': 'E94', 'TP10': 'E190'
}

# Liste brute de canaux à garder (ton exemple)
raw_list_string = "1,F8,3,4,F2,6,7,8,9,AF8,11,AF4,13,14,FCz,16,17,FP2,19,20,Fz,22,23,FC1,25,FPz,27,28,F1,30,32,33,AF3,35,F3,FP1,38,39,40,41,FC3,43,C1,45,AF7,F7,F5,FC5,50,51,52,53,54,55,56,57,58,C3,60,61,FT7,63,C5,65,CP3,T9,T7,70,71,72,74,75,CP5,77,78,CP1,80,81,83,TP7,85,P5,P3,P1,89,CPz,93,TP9,95,P7,PO7,98,99,100,Pz,103,104,105,P9,107,108,PO3,110,112,113,114,115,O1,117,118,POz,121,122,123,124,125,Oz,127,128,129,130,131,132,134,135,136,137,138,139,PO4,141,P2,CP2,144,146,147,148,149,O2,151,152,P4,154,155,156,157,158,159,160,PO8,P6,163,CP4,166,167,168,P10,P8,171,CP6,173,175,176,177,178,TP8,180,181,182,C4,184,C2,186,188,189,TP10,191,192,193,C6,195,196,197,198,200,201,T8,203,204,205,FC4,FC2,T10,FT8,212,FC6,214,215,220,221,F6,223,F4,Cz"

# =====================================
# CHARGEMENT DU RAW (mff)
# =====================================
raw = mne.io.read_raw_egi(EEG_PATH, preload=True)
print("Fichier chargé :", EEG_PATH)
print("Nombre total de canaux :", len(raw.ch_names))
print(raw.annotations)
unique_markers = np.unique(raw.annotations.description)
print("Marqueurs présents dans le fichier :")
for marker in unique_markers:
    print(marker)

# =====================================
# Construction de la liste de 204 canaux valides
# =====================================
canaux_204 = []
for item in raw_list_string.split(','):
    item = item.strip()
    egi_name = egi_lookup.get(item, f"E{item}")
    if egi_name in raw.ch_names:
        canaux_204.append(egi_name)

# Dédupliquer pour éviter doublons
canaux_204 = list(dict.fromkeys(canaux_204))

# Vérification
print("Nombre de canaux conservés :", len(canaux_204))
print("Canaux manquants :", [egi_lookup.get(item, f"E{item}") 
                            for item in raw_list_string.split(',') 
                            if egi_lookup.get(item, f"E{item}") not in raw.ch_names])

# =====================================
# Sélection des canaux existants seulement
# =====================================
canaux_existants = [ch for ch in canaux_204 if ch in raw.ch_names]
raw.pick(canaux_existants)
print("Pick des 204 canaux terminé. Canaux restants :", len(raw.ch_names))

# =====================================
# Renommage des canaux (déjà en E###)
# =====================================
rename_dict = {ch: ch for ch in raw.ch_names}
raw.rename_channels(rename_dict)

# =====================================
# Montage standard pour EGI 256 (subset des 204 canaux)
# =====================================
montage = mne.channels.make_standard_montage('GSN-HydroCel-256')
raw.set_montage(montage, match_case=False, on_missing='ignore')
print("Montage appliqué aux 204 canaux.")

# =====================================
# Affichage 2D des 204 canaux
# =====================================
raw.plot_sensors(kind='topomap', show_names=True)

# =====================================
# Sauvegarde du fichier nettoyé
# =====================================
raw.pick_types(eeg=True, stim=True, exclude=[])
folder, filename = os.path.split(EEG_PATH)
basename, ext = os.path.splitext(filename)
new_filename = f"{basename}_204ch_final.fif"
SAVE_PATH = os.path.join(folder, new_filename)

raw.save(SAVE_PATH, overwrite=True)
print("-" * 30)
print(f"TERMINÉ ! Fichier sauvegardé : {SAVE_PATH}")
print(f"Nombre de canaux dans le fichier enregistré : {len(raw.ch_names)}")
