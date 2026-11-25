# -*- coding: utf-8 -*-
"""
18 - Localisation de source des deux effets principaux - Lab Is Coat
====================================================================

Modele de tete commun (pas d'IRM individuelle) :
    - cerveau et BEM a 3 couches de fsaverage (MNE), source space ico-5
      (20 484 sources corticales), transformation "fsaverage" ;
    - positions des electrodes : celles des fichiers (montage GSN-HydroCel-256),
      ou le montage standard si USE_STANDARD_MONTAGE = True ;
    - reference moyenne (projecteur), matrice de bruit diagonale ad hoc
      (identique pour tous les sujets et conditions, ce qui evite d'estimer une
      covariance sur une ligne de base courte et contaminee par l'artefact).

PARTIE A - Effet de Valence sur la reponse evoquee tardive (eLORETA)
    Epoques evoquees du script 01 (config.EPOCHS_DIR). Pour chaque sujet et
    condition : ERP moyen -> eLORETA (lambda2 = 1/9, orientation libre,
    amplitude), moyenne dans les fenetres de la grande moyenne 238-493 et
    494-600 ms. Contraste DISG - APP (moyenne sur la realite) uniquement :
    la localisation sert a confirmer l'effet observe en surface.

PARTIE B - Effet du reel sur la puissance mu (beamformer DICS)
    Epoques longues du script 11b (outputs_tfr/epochs, avant CSD). Pour
    chaque sujet : un filtre DICS commun a toutes les conditions
    (8-13 Hz, -2.0 a 1.75 s), puissance mu de chaque condition dans la
    fenetre principale (0.3-1.75 s) et dans le pre-stimulus (-0.9 a -0.3 s),
    exprimee en dB par rapport a une ligne de base commune (-2.0 a -1.5 s,
    toutes conditions), comme dans le script 11. Contraste REAL - VIRT
    uniquement (fenetre principale), pour localiser l'effet observe en surface.
    DICS plutot qu'eLORETA ici : c'est la methode standard pour localiser
    une puissance oscillatoire, et elle evite une TFR sur 20 000 sources.

STATISTIQUES (les deux parties)
    Test de permutation par clusters spatiaux (5000 permutations, seuil de
    formation t bilateral p < .05, voisinage sur la surface corticale), sur
    les contrastes intra-sujet. Pour chaque cluster : p, nombre de sources,
    pic (t, coordonnees MNI), regions de l'atlas Desikan-Killiany (aparc).
    Partie B, en plus : ROI a priori (gyrus precentral + postcentral, gauche et
    droit) et test t apparie, pour relier le resultat a l'effet central droit
    du scalp.

Sorties : outputs_source/
    evoked_source_data.npz, mu_source_data.npz   (donnees par sujet, cache)
    clusters_<partie>_<contraste>.csv
    stc_<...>-lh.stc / -rh.stc                   (cartes moyennes et t)
    mu_roi_tests.csv
    figures/*.png                                (si pyvista est installe)

Usage : PART = "evoked", "mu" ou "both". RECOMPUTE = False relance seulement
les statistiques a partir du cache.
"""

import glob
import os
import inspect

import numpy as np
import pandas as pd
import mne
from scipy import stats
from mne.minimum_norm import make_inverse_operator, apply_inverse, prepare_inverse_operator
from mne.beamformer import make_dics, apply_dics_csd
from mne.time_frequency import csd_morlet

import config

# ==========================================================
# PARAMETRES
# ==========================================================

PART = "both"            # "evoked", "mu" ou "both"
RECOMPUTE = True         # False = statistiques seules, a partir du cache
OUTDIR = "outputs_source"
FIGDIR = os.path.join(OUTDIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)

EVOKED_DIR = config.EPOCHS_DIR
TFR_EPOCHS_DIR = os.path.join("outputs_tfr", "epochs")
EXCLUDED = ["DN12"]      # sujet exclu de l'EEG (moins de 2 essais par condition)

SRC_SPACING = "ico4"     # "ico4" (5124 sources, rapide) ou "ico5" (20 484 sources)
USE_STANDARD_MONTAGE = False   # True : remplace les positions par GSN-HydroCel-256 standard

CONDITIONS = config.CONDITIONS  # {"REAL_APP": [...], ...}

# Partie A
EVOKED_WINDOWS = {"238-493": (0.238, 0.493), "494-600": (0.494, 0.600)}
LAMBDA2 = 1.0 / 9.0

# Partie B
MU_METHOD = "eloreta"    # "eloreta" (meme methode que la partie A) ou "dics"
MU_BAND = (8.0, 13.0)
MU_FREQS = np.arange(8.0, 14.0, 1.0)     # 8-13 Hz
MU_N_CYCLES = 5
MAIN_WIN = (0.3, 1.75)
PRESTIM_WIN = (-0.9, -0.3)
BASE_WIN = (-2.0, -1.5)
FILTER_WIN = (-2.0, 1.75)
DICS_REG = 0.05
CSD_DECIM = 5            # 250 Hz -> 50 Hz pour le calcul des CSD (suffisant pour 8-13 Hz)
N_JOBS = 4               # coeurs utilises pour les CSD (1 si probleme)

# Statistiques
N_PERM = 5000            # 1000 pour un premier essai rapide
P_FORM = 0.05
SEED = 42


# ==========================================================
# MODELE DE TETE
# ==========================================================

fs_dir = mne.datasets.fetch_fsaverage(verbose=True)     # telecharge une fois (~200 Mo)
subjects_dir = os.path.dirname(fs_dir)
if SRC_SPACING == "ico5":
    SRC = mne.read_source_spaces(os.path.join(fs_dir, "bem", "fsaverage-ico-5-src.fif"),
                                 verbose=False)
else:   # ico4 : 5124 sources, environ 4 fois plus rapide
    SRC = mne.setup_source_space("fsaverage", spacing=SRC_SPACING, subjects_dir=subjects_dir,
                                 add_dist=False, verbose=False)
BEM = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")
VERTICES = [SRC[0]["vertno"], SRC[1]["vertno"]]
N_LH = len(VERTICES[0])
ADJ = mne.spatial_src_adjacency(SRC, verbose=False)
LABELS = mne.read_labels_from_annot("fsaverage", "aparc", subjects_dir=subjects_dir, verbose=False)
LABELS = [lb for lb in LABELS if "unknown" not in lb.name]

# region aparc de chaque source (index dans lh + rh concatenes)
vertex_label = np.array(["?"] * (N_LH + len(VERTICES[1])), dtype=object)
for lb in LABELS:
    h = 0 if lb.hemi == "lh" else 1
    idx = np.where(np.isin(VERTICES[h], lb.vertices))[0] + (0 if h == 0 else N_LH)
    vertex_label[idx] = lb.name

_fwd_cache = {}
_inv_cache = {}


def prepare_info(inst):
    """Montage, canaux sans position retires, reference moyenne (projecteur)."""
    inst = inst.copy().pick("eeg", exclude="bads")
    if USE_STANDARD_MONTAGE or inst.get_montage() is None:
        inst.set_montage(mne.channels.make_standard_montage("GSN-HydroCel-256"),
                         on_missing="ignore")
    bad_pos = [ch["ch_name"] for ch in inst.info["chs"]
               if not np.all(np.isfinite(ch["loc"][:3])) or np.allclose(ch["loc"][:3], 0)]
    if bad_pos:
        print(f"   canaux sans position retires : {bad_pos}")
        inst.drop_channels(bad_pos)
    inst.set_eeg_reference("average", projection=True, verbose=False)
    inst.apply_proj(verbose=False)
    return inst


def forward_for(info):
    """Solution directe fsaverage, mise en cache par jeu de canaux."""
    key = tuple(info["ch_names"])
    if key not in _fwd_cache:
        fname = os.path.join(OUTDIR, f"fsaverage-{SRC_SPACING}-{len(key)}ch-fwd.fif")
        fwd = None
        if os.path.exists(fname):
            fwd = mne.read_forward_solution(fname, verbose=False)
            if fwd["info"]["ch_names"] != list(key):
                fwd = None
            else:
                print(f"   solution directe relue : {fname}")
        if fwd is None:
            print(f"   calcul de la solution directe ({len(key)} electrodes, {SRC_SPACING}) "
                  "- plusieurs minutes")
            fwd = mne.make_forward_solution(
                info, trans="fsaverage", src=SRC, bem=BEM, eeg=True, meg=False,
                mindist=5.0, n_jobs=N_JOBS, verbose=True)
            mne.write_forward_solution(fname, fwd, overwrite=True, verbose=False)
            print(f"   sauvegardee : {fname} (reutilisee aux prochains lancements)")
        _fwd_cache[key] = fwd
    return _fwd_cache[key]


def inverse_for(info):
    """Operateur eLORETA prepare, calcule une seule fois par jeu de canaux."""
    key = tuple(info["ch_names"])
    if key not in _inv_cache:
        print("   calcul de l'operateur inverse eLORETA (une seule fois)")
        fwd = forward_for(info)
        cov = mne.make_ad_hoc_cov(info, verbose=False)
        inv = make_inverse_operator(info, fwd, cov, loose=0.2, depth=0.8, verbose=False)
        print("   ajustement eLORETA (iterations affichees ci-dessous)")
        _inv_cache[key] = prepare_inverse_operator(inv, nave=1, lambda2=LAMBDA2,
                                                   method="eLORETA", verbose=True)
    return _inv_cache[key]


def eloreta_kernel(info):
    """Matrice (sources, 3 orientations, canaux) de l'operateur eLORETA : on
    l'obtient en l'appliquant a une matrice identite (une 'mesure' par canal)."""
    inv = inverse_for(info)
    n = len(info["ch_names"])
    ident = mne.EvokedArray(np.eye(n), info, tmin=0, nave=1, verbose=False)
    stc = apply_inverse(ident, inv, LAMBDA2, method="eLORETA", pick_ori="vector",
                        prepared=True, verbose=False)
    return stc.data                                   # (sources, 3, canaux)


def subject_code(path):
    import re
    m = re.search(r"([A-Z]{2}\d{2})", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def cond_epochs(ep, cond):
    names = [v for v in CONDITIONS[cond] if v in ep.event_id]
    return ep[names] if names else None


# ==========================================================
# PARTIE A : eLORETA sur les ERP
# ==========================================================

def compute_evoked_sources():
    files = sorted(glob.glob(os.path.join(EVOKED_DIR, "*-epo.fif")))
    subjects, data = [], []                 # data : (sujets, conditions, fenetres, sources)
    for f in files:
        s = subject_code(f)
        if s in EXCLUDED:
            continue
        print(f"\n[A] {s}")
        ep = prepare_info(mne.read_epochs(f, preload=True, verbose=False))
        # Modele de tete, electrodes et matrice de bruit identiques pour tous les
        # sujets -> l'operateur inverse est le meme : on le calcule une seule fois
        # (l'ajustement iteratif d'eLORETA est l'etape la plus longue).
        inv = inverse_for(ep.info)
        sub = []
        for cond in CONDITIONS:
            e = cond_epochs(ep, cond)
            evk = e.average()
            evk.nave = 1                   # meme nave pour toutes les conditions (inverse prepare)
            stc = apply_inverse(evk, inv, LAMBDA2, method="eLORETA", pick_ori=None,
                                prepared=True, verbose=False)
            sub.append([stc.copy().crop(a, b).data.mean(axis=1)
                        for a, b in EVOKED_WINDOWS.values()])
        subjects.append(s)
        data.append(sub)
    data = np.array(data)
    np.savez_compressed(os.path.join(OUTDIR, "evoked_source_data.npz"), data=data,
                        subjects=np.array(subjects), conditions=np.array(list(CONDITIONS)),
                        windows=np.array(list(EVOKED_WINDOWS)))
    return data, subjects


# ==========================================================
# PARTIE B : DICS sur la puissance mu
# ==========================================================

def dics_power(csd, filters):
    stc, _ = apply_dics_csd(csd.mean(), filters, verbose=False)
    return stc.data[:, 0]


def compute_mu_sources():
    files = sorted(glob.glob(os.path.join(TFR_EPOCHS_DIR, "*-tfr-epo.fif")))
    if not files:
        raise SystemExit(f"Aucune epoque dans {TFR_EPOCHS_DIR} : lancer d'abord 11b_save_tfr_epochs.py")
    subjects, data = [], []                 # data : (sujets, conditions, [main, prestim], sources)
    kw = dict(frequencies=MU_FREQS, n_cycles=MU_N_CYCLES, decim=CSD_DECIM, n_jobs=N_JOBS,
              verbose=False)
    for f in files:
        s = subject_code(f)
        if s in EXCLUDED:
            continue
        print(f"\n[B] {s}")
        ep = prepare_info(mne.read_epochs(f, preload=True, verbose=False))
        fwd = forward_for(ep.info)
        csd_filter = csd_morlet(ep, tmin=FILTER_WIN[0], tmax=FILTER_WIN[1], **kw)
        filters = make_dics(ep.info, fwd, csd_filter.mean(), reg=DICS_REG, pick_ori="max-power",
                            weight_norm="unit-noise-gain", reduce_rank=False, real_filter=True,
                            rank=None, verbose=False)
        base = dics_power(csd_morlet(ep, tmin=BASE_WIN[0], tmax=BASE_WIN[1], **kw), filters)
        sub = []
        for cond in CONDITIONS:
            e = cond_epochs(ep, cond)
            p_main = dics_power(csd_morlet(e, tmin=MAIN_WIN[0], tmax=MAIN_WIN[1], **kw), filters)
            p_pre = dics_power(csd_morlet(e, tmin=PRESTIM_WIN[0], tmax=PRESTIM_WIN[1], **kw), filters)
            sub.append([10 * np.log10(p_main / base), 10 * np.log10(p_pre / base)])
        subjects.append(s)
        data.append(sub)
    data = np.array(data)
    np.savez_compressed(os.path.join(OUTDIR, "mu_source_data.npz"), data=data,
                        subjects=np.array(subjects), conditions=np.array(list(CONDITIONS)),
                        windows=np.array(["main", "prestim"]))
    return data, subjects


def compute_mu_sources_eloreta():
    """Puissance mu en sources par eLORETA. La puissance d'un signal filtre
    8-13 Hz dans une fenetre est quadratique : en source, elle vaut
    diag(K C K'), ou C est la covariance des electrodes dans la fenetre et K
    l'operateur eLORETA. On calcule donc C par condition et par fenetre au
    niveau des electrodes, puis on la projette : exact et tres rapide."""
    files = sorted(glob.glob(os.path.join(TFR_EPOCHS_DIR, "*-tfr-epo.fif")))
    if not files:
        raise SystemExit(f"Aucune epoque dans {TFR_EPOCHS_DIR} : lancer d'abord 11b_save_tfr_epochs.py")
    subjects, data = [], []                 # (sujets, conditions, [main, prestim], sources)
    K = None
    for f in files:
        s = subject_code(f)
        if s in EXCLUDED:
            continue
        print(f"\n[B] {s}")
        ep = prepare_info(mne.read_epochs(f, preload=True, verbose=False))
        if K is None:
            K = eloreta_kernel(ep.info)
        ep = ep.filter(*MU_BAND, verbose=False)
        times = ep.times

        def src_power(e, win):
            m = (times >= win[0]) & (times <= win[1])
            x = e.get_data()[:, :, m]                          # (essais, canaux, temps)
            C = np.einsum("ect,edt->cd", x, x) / (x.shape[0] * x.shape[2])
            return np.einsum("soj,jk,sok->s", K, C, K)         # somme des 3 orientations

        base = src_power(ep, BASE_WIN)                         # ligne de base commune
        sub = []
        for cond in CONDITIONS:
            e = cond_epochs(ep, cond)
            sub.append([10 * np.log10(src_power(e, MAIN_WIN) / base),
                        10 * np.log10(src_power(e, PRESTIM_WIN) / base)])
        subjects.append(s)
        data.append(sub)
    data = np.array(data)
    np.savez_compressed(os.path.join(OUTDIR, "mu_source_data_eloreta.npz"), data=data,
                        subjects=np.array(subjects), conditions=np.array(list(CONDITIONS)),
                        windows=np.array(["main", "prestim"]))
    return data, subjects


# ==========================================================
# STATISTIQUES
# ==========================================================

def contrast(data, kind):
    """data : (sujets, conditions, ...) dans l'ordre de CONDITIONS."""
    c = {k: i for i, k in enumerate(CONDITIONS)}
    ra, va, rd, vd = (data[:, c[k]] for k in ["REAL_APP", "VIRT_APP", "REAL_DISG", "VIRT_DISG"])
    if kind == "valence":
        return (rd + vd) / 2 - (ra + va) / 2
    if kind == "reality":
        return (ra + rd) / 2 - (va + vd) / 2
    if kind == "interaction":
        return (ra - va) - (rd - vd)
    raise ValueError(kind)


def to_stc(values, tag):
    stc = mne.SourceEstimate(values[:, None], vertices=VERTICES, tmin=0, tstep=1,
                             subject="fsaverage")
    stc.save(os.path.join(OUTDIR, f"stc_{tag}"), overwrite=True, verbose=False)
    return stc


def cluster_test(X, tag):
    """X : (sujets, sources). Clusters spatiaux, resume et cartes."""
    n = X.shape[0]
    t_thr = stats.t.ppf(1 - P_FORM / 2, n - 1)
    T, clusters, pv, _ = mne.stats.permutation_cluster_1samp_test(
        X, threshold=t_thr, adjacency=ADJ, n_permutations=N_PERM, tail=0,
        out_type="indices", verbose=False, **{RNG_KW: SEED})
    rows = []
    for k, (cl, p) in enumerate(zip(clusters, pv)):
        idx = cl[0] if isinstance(cl, tuple) else cl
        peak = idx[np.argmax(np.abs(T[idx]))]
        hemi = 0 if peak < N_LH else 1
        vert = VERTICES[hemi][peak - (0 if hemi == 0 else N_LH)]
        mni = mne.vertex_to_mni(vert, hemi, "fsaverage", subjects_dir=subjects_dir, verbose=False)
        regions = pd.Series(vertex_label[idx]).value_counts()
        rows.append({
            "cluster": k, "p": p, "sign": "positive" if T[peak] > 0 else "negative",
            "n_sources": len(idx), "peak_t": T[peak], "peak_region": vertex_label[peak],
            "peak_hemi": "lh" if hemi == 0 else "rh",
            "peak_mni_x": mni[0], "peak_mni_y": mni[1], "peak_mni_z": mni[2],
            "regions": "; ".join(f"{r} ({v})" for r, v in regions.head(8).items()),
        })
    out = pd.DataFrame(rows).sort_values("p") if rows else pd.DataFrame()
    out.to_csv(os.path.join(OUTDIR, f"clusters_{tag}.csv"), index=False)
    to_stc(X.mean(axis=0), f"{tag}_mean")
    to_stc(T, f"{tag}_t")
    sig = out[out.p < 0.05] if len(out) else out
    print(f"\n--- {tag} (n = {n}) : {len(sig)} cluster(s) p < .05 ---")
    if len(sig):
        cols = ["p", "sign", "n_sources", "peak_t", "peak_region", "peak_hemi",
                "peak_mni_x", "peak_mni_y", "peak_mni_z"]
        print(sig[cols].round(3).to_string(index=False))
        for _, r in sig.iterrows():
            print(f"   regions : {r.regions}")
    else:
        min_p = out.p.min() if len(out) else np.nan
        print(f"   aucun (plus petit p de cluster = {min_p:.3f})")
    plot_map(X.mean(axis=0), f"{tag}_mean")
    plot_map(T, f"{tag}_t")
    return out


RNG_KW = ("rng" if "rng" in inspect.signature(mne.stats.permutation_cluster_1samp_test).parameters
          else "seed")


def plot_map(values, tag):
    """Cartes corticales (faces laterale et mediale, deux hemispheres).
    pyvista si disponible, sinon rendu matplotlib (plus lent, sans 3D)."""
    import matplotlib.pyplot as plt
    stc = mne.SourceEstimate(values[:, None], vertices=VERTICES, tmin=0, tstep=1,
                             subject="fsaverage")
    lim = float(np.percentile(np.abs(values), 99)) or 1.0
    clim = dict(kind="value", pos_lims=[0.0, lim / 2, lim])
    try:
        for hemi in ["lh", "rh"]:
            brain = stc.plot(subject="fsaverage", subjects_dir=subjects_dir, hemi=hemi,
                             views=["lateral", "medial"], colormap="RdBu_r", transparent=False,
                             clim=clim, time_viewer=False, show_traces=False,
                             background="white", size=(900, 450), backend="pyvistaqt",
                             verbose=False)
            brain.save_image(os.path.join(FIGDIR, f"{tag}_{hemi}.png"))
            brain.close()
        return
    except Exception:
        pass
    try:
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), subplot_kw=dict(projection="3d"))
        k = 0
        for hemi in ["lh", "rh"]:
            for view in ["lateral", "medial"]:
                stc.plot(subject="fsaverage", subjects_dir=subjects_dir, hemi=hemi, views=view,
                         colormap="RdBu_r", transparent=False, clim=clim, backend="matplotlib",
                         figure=fig, colorbar=(k == 3), time_label=None, verbose=False)
                k += 1
        fig.suptitle(tag, fontsize=10)
        fig.savefig(os.path.join(FIGDIR, f"{tag}.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"   (figure non produite : {type(e).__name__}: {e})")


def roi_tests(X_by_cond, subjects):
    """ROI a priori sensorimotrices (precentral + postcentral), partie B."""
    rows = []
    for hemi in ["lh", "rh"]:
        mask = np.isin(vertex_label, [f"precentral-{hemi}", f"postcentral-{hemi}"])
        for w, wname in enumerate(["main"]):
            d = contrast(X_by_cond[:, :, w], "reality")[:, mask].mean(axis=1)
            t, p = stats.ttest_1samp(d, 0)
            rows.append({"roi": f"pre+postcentral {hemi}", "window": wname, "n": len(d),
                         "mean_db": d.mean(), "sd": d.std(ddof=1), "t": t, "p": p,
                         "dz": d.mean() / d.std(ddof=1)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUTDIR, "mu_roi_tests.csv"), index=False)
    print("\n--- ROI sensorimotrices, mu, REAL - VIRT (dB) ---")
    print(out.round(3).to_string(index=False))


# ==========================================================
# EXECUTION
# ==========================================================

if PART in ("evoked", "both"):
    if RECOMPUTE:
        A, subj_A = compute_evoked_sources()
    else:
        z = np.load(os.path.join(OUTDIR, "evoked_source_data.npz"))
        A, subj_A = z["data"], list(z["subjects"])
    print(f"\n===== PARTIE A : eLORETA, {len(subj_A)} sujets =====")
    # Confirmatoire uniquement : localiser l'effet de Valence observe au niveau
    # des electrodes (GFP, 238-493 et 494-600 ms). Pas de test de la realite ici :
    # aucun effet du reel n'a ete trouve sur l'activite evoquee en surface.
    for w, wname in enumerate(EVOKED_WINDOWS):
        cluster_test(contrast(A[:, :, w], "valence"), f"evoked_valence_{wname}")

if PART in ("mu", "both"):
    cache = "mu_source_data_eloreta.npz" if MU_METHOD == "eloreta" else "mu_source_data.npz"
    if RECOMPUTE:
        B, subj_B = (compute_mu_sources_eloreta() if MU_METHOD == "eloreta"
                     else compute_mu_sources())
    else:
        z = np.load(os.path.join(OUTDIR, cache))
        B, subj_B = z["data"], list(z["subjects"])
    print(f"\n===== PARTIE B : {MU_METHOD} mu, {len(subj_B)} sujets =====")
    # Carte illustrative : variation de mu (fenetre principale vs ligne de base),
    # toutes conditions confondues -> ou se situe l'activite mu elle-meme.
    plot_map(B[:, :, 0].mean(axis=(0, 1)), f"mu_{MU_METHOD}_all_conditions_vs_baseline")
    to_stc(B[:, :, 0].mean(axis=(0, 1)), f"mu_{MU_METHOD}_all_conditions_vs_baseline")
    # Confirmatoire uniquement : localiser l'effet du reel sur la puissance mu
    # (fenetre principale), observe au niveau des electrodes centrales droites.
    cluster_test(contrast(B[:, :, 0], "reality"), f"mu_{MU_METHOD}_reality_main")
    roi_tests(B, subj_B)

print(f"\nSorties dans {OUTDIR}")
