# -*- coding: utf-8 -*-

"""
Configuration partagee du pipeline micro-etats

Tous les scripts importent ce fichier -> un seul endroit ou corriger
un chemin ou un parametre. Dans l'ancien pipeline, EPOCHS_DIR etait
redefini (et incoherent) dans 3 fichiers differents.
"""

import os

import mne

# ==========================================================
# CHEMINS
# ==========================================================

# Dossier contenant les fichiers *-epo.fif (un par sujet)
EPOCHS_DIR = r""

# Dossier de sortie pour tous les fichiers generes par le pipeline
OUTPUT_DIR = "outputs_microstates"

# ==========================================================
# PARADIGME NOURRITURE
# ==========================================================

# REAL_APP  = VIS1 + VIS2
# VIRT_APP  = VIS3 + VIS4
# REAL_DISG = VIS5 + VIS6
# VIRT_DISG = VIS7 + VIS8

CONDITIONS = {
    "REAL_APP": ["VIS1", "VIS2"],
    "VIRT_APP": ["VIS3", "VIS4"],
    "REAL_DISG": ["VIS5", "VIS6"],
    "VIRT_DISG": ["VIS7", "VIS8"],
}

# Fenetre d'analyse post-stimulus (secondes)
# Debut a 80 ms : avant, le signal a ete modifie par le pretraitement
# (correction PCA de l'artefact d'ecran de -30 a +50 ms, puis lissage
# Savitzky-Golay du raccord de 30 a 80 ms). Utilise par TOUS les
# scripts d'analyse (01-05, 07, 08, 10).
TMIN = 0.080
TMAX = 0.600

# Nombre de pics GFP conserves par sujet et par condition
N_PEAKS_PER_CONDITION = 30

# Distance minimale (en echantillons) entre deux pics GFP retenus
MIN_PEAK_DISTANCE = 2

# Mode test rapide : limite le nombre de sujets traites (fichiers
# *-epo.fif) pour verifier que le pipeline tourne avant de lancer sur
# tout le dossier. Mettre None pour traiter tous les sujets.
DEBUG_MAX_SUBJECTS = None

# ==========================================================
# CLUSTERING (k-means, microstate_utils.py)
# ==========================================================

K_RANGE = range(2, 9)
N_INIT = 100
MAX_ITER = 300
RANDOM_STATE = 42

# Polarite : False = une carte et son inverse de signe sont deux etats
# DIFFERENTS (convention ERP, Murray et al. 2008). True = invariance de
# polarite (convention EEG spontane, comportement de pycrostates) ;
# a garder seulement pour comparaison.
IGNORE_POLARITY = False

# Critere de choix de K : "KL" (Krzanowski-Lai, reference pour les
# micro-etats d'ERP, Murray et al. 2008) ou "mean_rank" (rang moyen
# silhouette / Calinski-Harabasz / Davies-Bouldin, qui favorise presque
# toujours K = 2 sur des ERP).
K_CRITERION = "KL"

# Nombre de micro-etats impose. None = choix automatique (K_CRITERION). Mettre un entier
# (ex. 5) pour imposer K apres inspection de microstate_K_selection.png ;
# le script 02 affiche quand meme le K qu'aurait choisi le critere
# automatique (a mentionner dans les Methodes).
K_FORCED = None

# ==========================================================
# BACKFITTING / SEGMENTATION
# ==========================================================

# Duree minimale d'un segment (ms) -> segments plus courts fusionnes
MIN_SEGMENT_DURATION_MS = 20

# Lissage temporel de Pascual-Marqui (0 = pas de lissage)
SMOOTHING_FACTOR = 10
SMOOTHING_HALF_WINDOW = 3          # en echantillons
SMOOTHING_TOL = 1e-5

# ==========================================================
# FENETRES FIXES (scripts 07 / 08 / 10)
# ==========================================================

# Ecrit par le script 03 : segments de la grande moyenne, toutes
# conditions confondues.
GA_SEGMENTS_FILE = os.path.join(OUTPUT_DIR, "grand_average_segments.csv")

# Les segments qui se terminent avant ce delai (ms) sont ignores
# (pas encore d'activite evoquee, GFP au niveau du bruit).
GA_WINDOW_MIN_END_MS = 50


def load_ga_windows(path=None):
    """Fenetres fixes {"MS3_75_145": (75, 145), ...} lues dans le CSV
    du script 03. Erreur explicite si le fichier n'existe pas."""
    import pandas as pd

    path = path or GA_SEGMENTS_FILE
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} introuvable : lancer d'abord 03_backfit_segmentation_parameters.py "
            "(ou indiquer le chemin du fichier)."
        )
    seg = pd.read_csv(path)
    windows = {}
    for r in seg.itertuples(index=False):
        if r.end_ms < GA_WINDOW_MIN_END_MS:
            continue
        name = f"{r.microstate}_{int(round(r.start_ms))}_{int(round(r.end_ms))}"
        windows[name] = (float(r.start_ms), float(r.end_ms))
    return windows


# ==========================================================
# FACTEURS EXPERIMENTAUX (pour les stats)
# ==========================================================


def reality(condition):
    return "REAL" if "REAL" in condition else "VIRTUAL"


def valence(condition):
    return "APP" if "APP" in condition else "DISG"


# ==========================================================
# CHARGEMENT HOMOGENE DES EPOCHS (utilise par 01, 03, 05)
# ==========================================================
#
# Les fichiers d'un meme dossier peuvent avoir un nombre de canaux EEG
# ou une frequence d'echantillonnage differents d'un sujet a l'autre
# (canal bruite retire chez certains, session enregistree avec un
# autre parametrage...). Empiler directement des cartes/evoques de
# formes differentes plante numpy avec une erreur cryptique. Ces deux
# fonctions centralisent la verification et la mise en coherence, pour
# que tous les scripts du pipeline appliquent la meme regle.


def get_dataset_specs(files):
    """Scanne tous les fichiers epochs et renvoie :

    - common_channels : liste triee des canaux EEG communs a tous les
      sujets (intersection).
    - target_sfreq : frequence d'echantillonnage de reference (la plus
      elevee parmi les sujets ; les autres seront suréchantillonnes).

    Affiche un avertissement si canaux ou frequence different selon
    les sujets.
    """

    channel_sets = {}
    sfreqs = {}

    for file in files:
        subject = os.path.basename(file).replace("-epo.fif", "")
        ep_info = mne.read_epochs(file, preload=False, verbose=False)
        eeg_idx = mne.pick_types(ep_info.info, eeg=True)
        eeg_ch_names = [ep_info.info["ch_names"][i] for i in eeg_idx]
        channel_sets[subject] = set(eeg_ch_names)
        sfreqs[subject] = ep_info.info["sfreq"]
        print(f"  {subject} : {len(eeg_ch_names)} canaux EEG, {ep_info.info['sfreq']} Hz")

    common_channels = sorted(set.intersection(*channel_sets.values()))
    n_per_subject = {s: len(c) for s, c in channel_sets.items()}

    if len(set(n_per_subject.values())) > 1:
        print(
            "\nATTENTION : le nombre de canaux EEG differe selon les sujets :",
            n_per_subject,
        )
        print(
            f"-> restriction a l'intersection commune : {len(common_channels)} canaux."
        )

    target_sfreq = max(sfreqs.values())

    if len(set(sfreqs.values())) > 1:
        print(
            "\nATTENTION : la frequence d'echantillonnage differe selon les sujets :",
            sfreqs,
        )
        print(f"-> tous les sujets seront ramenes a {target_sfreq} Hz.")

    return common_channels, target_sfreq


def prepare_epochs(file, common_channels, target_sfreq):
    """Charge un fichier epochs et le rend homogene avec les autres
    sujets : memes canaux EEG (intersection commune), meme frequence
    d'echantillonnage. A appeler pour chaque sujet apres avoir obtenu
    common_channels/target_sfreq via get_dataset_specs()."""

    epochs = mne.read_epochs(file, preload=True, verbose=False)
    epochs.pick_types(eeg=True)
    epochs.pick(common_channels)

    if epochs.info["sfreq"] != target_sfreq:
        epochs.resample(target_sfreq, verbose=False)

    return epochs
