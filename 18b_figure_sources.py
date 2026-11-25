# -*- coding: utf-8 -*-
"""
18b - Figure de l'article : localisation de source (illustrative) - Lab Is Coat
==============================================================================

A lancer APRES le script 18 (dans le meme dossier). Ne refait pas les
calculs de sources : relit les donnees sauvegardees par le script 18
    outputs_source/evoked_source_data.npz
    outputs_source/mu_source_data_eloreta.npz
refait les tests confirmatoires (memes parametres, meme graine, donc memes
clusters), et compose une figure unique en trois lignes :

  A - Valence, 238-493 ms (DISG - APP, amplitude eLORETA) :
      t(27) des sources du (des) cluster(s) significatif(s) uniquement.
  B - Valence, 494-600 ms : idem.
  C - Realite, puissance mu 8-13 Hz, 0.3-1.75 s (REAL - PICTURE) :
      carte t(27) seuillee a |t| > 2.05 (p < .05 non corrige, illustratif),
      contour de la ROI a priori (gyrus precentral + postcentral).

Chaque ligne a sa propre echelle de couleur (de 2.05 au maximum de |t|
affiche dans la ligne) et sa propre barre de couleur ; les valeurs
|t| < 2.05 ne sont pas affichees (bande grise sur la barre).

Chaque ligne : hemisphere gauche lateral, gauche medial, droit lateral,
droit medial ; surface gonflee fsaverage.

Sorties : outputs_source/figures/figure_sources.png (300 dpi) et .pdf
          outputs_source/clusters_figure_<contraste>.csv (rappel des clusters)
"""

import os
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy import stats
import mne

OUTDIR = "outputs_source"
FIGDIR = os.path.join(OUTDIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)
SRC_SPACING = "ico4"          # comme le script 18
N_PERM, P_FORM, SEED = 5000, 0.05, 42
TILE = (700, 500)             # taille de chaque vue (pixels)

# ---------------- reglages de la figure (a modifier) ----------------
# Palette : les effets montres sont tous positifs (DISG > APP, REAL > PICTURE),
# une palette sequentielle suffit et se lit mieux qu'un bleu-rouge.
#   "hot" (noir-rouge-jaune, classique en neuroimagerie), "YlOrRd", "inferno",
#   "magma", "viridis" ; "RdBu_r" pour revenir a l'ancien rendu.
CMAP = "YlOrRd"
# Haut de l'echelle : percentile des |t| affiches dans chaque ligne (et non
# le maximum) -> la plupart des sources atteignent les couleurs vives et un
# pic isole (ex. t = 7.1 dans l'insula) ne "blanchit" plus le reste.
LIM_PERCENTILE = 99
# Vues : ajouter "ventral" pour voir le cortex orbitofrontal ; mettre
# HEMIS = ["rh"] pour ne montrer que l'hemisphere droit (vues plus grandes).
HEMIS = ["lh", "rh"]
VIEWS = ["lateral", "medial"]
CORTEX = "classic"            # fond gris clair/fonce des sillons ("low_contrast", "bone")
# --------------------------------------------------------------------


# ---------------- source space (identique au script 18) ----------------
fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
subjects_dir = os.path.dirname(fs_dir)
SRC = mne.setup_source_space("fsaverage", spacing=SRC_SPACING, subjects_dir=subjects_dir,
                             add_dist=False, verbose=False)
VERTICES = [SRC[0]["vertno"], SRC[1]["vertno"]]
ADJ = mne.spatial_src_adjacency(SRC, verbose=False)
LABELS = {lb.name: lb for lb in mne.read_labels_from_annot("fsaverage", "aparc",
                                                           subjects_dir=subjects_dir,
                                                           verbose=False)}
RNG_KW = ("rng" if "rng" in inspect.signature(mne.stats.permutation_cluster_1samp_test).parameters
          else "seed")


def contrast(data, conds, kind):
    c = {k: i for i, k in enumerate(conds)}
    ra, va, rd, vd = (data[:, c[k]] for k in ["REAL_APP", "VIRT_APP", "REAL_DISG", "VIRT_DISG"])
    if kind == "valence":
        return (rd + vd) / 2 - (ra + va) / 2
    return (ra + rd) / 2 - (va + vd) / 2


def clusters(X, tag):
    """Test de clusters spatial ; renvoie t, masque des clusters p < .05,
    seuil et plus petit p."""
    n = X.shape[0]
    thr = stats.t.ppf(1 - P_FORM / 2, n - 1)
    T, cl, pv, _ = mne.stats.permutation_cluster_1samp_test(
        X, threshold=thr, adjacency=ADJ, n_permutations=N_PERM, tail=0,
        out_type="indices", verbose=False, **{RNG_KW: SEED})
    mask = np.zeros(X.shape[1], bool)
    rows = []
    for c, p in zip(cl, pv):
        idx = c[0] if isinstance(c, tuple) else c
        rows.append({"p": p, "n_sources": len(idx), "max_abs_t": np.abs(T[idx]).max()})
        if p < .05:
            mask[idx] = True
    pd.DataFrame(rows).sort_values("p").to_csv(
        os.path.join(OUTDIR, f"clusters_figure_{tag}.csv"), index=False)
    p_min = float(np.min(pv)) if len(pv) else np.nan
    print(f"{tag} : {int((np.array(pv) < .05).sum())} cluster(s) p < .05, plus petit p = {p_min:.3f}")
    return T, mask, thr, p_min


def nice_lim(values, thr):
    """Haut de l'echelle : percentile LIM_PERCENTILE des |t| affiches,
    arrondi au 0.5 superieur."""
    v = np.abs(values[values != 0])
    if v.size == 0:
        return thr + 0.5
    m = np.percentile(v, LIM_PERCENTILE)
    return max(np.ceil(m * 2) / 2, thr + 0.5)


def render_row(values, thr, lim, roi=None):
    """Une capture par (hemisphere, vue) d'une carte t."""
    signed = (values < 0).any()                      # valeurs negatives a montrer ?
    if signed:
        cmap, clim = "RdBu_r", dict(kind="value", pos_lims=[thr, thr + 0.01, lim])
    else:
        cmap, clim = CMAP, dict(kind="value", lims=[thr, thr + 0.01, lim])
    stc = mne.SourceEstimate(values[:, None], vertices=VERTICES, tmin=0, tstep=1,
                             subject="fsaverage")
    imgs = []
    for hemi in HEMIS:
        for view in VIEWS:
            # fmid juste au-dessus du seuil : pas de fondu transparent, toutes les
            # sources affichees sont opaques
            brain = stc.plot(subject="fsaverage", subjects_dir=subjects_dir, hemi=hemi,
                             views=view, surface="inflated", colormap=cmap, clim=clim,
                             transparent=True, colorbar=False, time_viewer=False,
                             show_traces=False, background="white", size=TILE,
                             cortex=CORTEX, backend="pyvistaqt", verbose=False)
            if roi is not None and hemi in roi:
                brain.add_label(roi[hemi], borders=2, color="black")
            imgs.append(brain.screenshot())
            brain.close()
    return imgs, signed


rows = []

# ---------------- A et B : Valence (ERP), deux fenetres ----------------
z = np.load(os.path.join(OUTDIR, "evoked_source_data.npz"))
windows = list(z["windows"])
for letter, win in zip("AB", ["238-493", "494-600"]):
    X = contrast(z["data"][:, :, windows.index(win)], list(z["conditions"]), "valence")
    T, mask, thr, p_min = clusters(X, f"valence_{win}")
    vals = np.where(mask, T, 0.0)
    lim = nice_lim(vals, thr)
    imgs, signed = render_row(vals, thr, lim)
    rows.append(dict(imgs=imgs, signed=signed, thr=thr, lim=lim, vmax=np.abs(vals).max(),
                     title=f"{letter}  Valence (disgusting − appetizing), "
                           f"{win.replace('-', '–')} ms, evoked amplitude "
                           f"(cluster p = {p_min:.3f})"))

# ---------------- C : Realite (mu) ----------------
z = np.load(os.path.join(OUTDIR, "mu_source_data_eloreta.npz"))
X = contrast(z["data"][:, :, 0], list(z["conditions"]), "reality")
T, mask, thr, p_min = clusters(X, "mu_reality_main")
vals = np.where(np.abs(T) > thr, T, 0.0)
lim = nice_lim(vals, thr)
roi = {h: LABELS[f"precentral-{h}"] + LABELS[f"postcentral-{h}"] for h in ["lh", "rh"]}
imgs, signed = render_row(vals, thr, lim, roi)
rows.append(dict(imgs=imgs, signed=signed, thr=thr, lim=lim, vmax=np.abs(vals).max(),
                 title="C  Reality (real − picture), μ power, 0.3–1.75 s "
                       "(|t| > 2.05, uncorrected)"))
for r in rows:
    print(f"{r['title'][:1]} : echelle {r['thr']:.2f} -> {r['lim']:.1f} "
          f"(max affiche {r['vmax']:.2f}), palette {'RdBu_r' if r['signed'] else CMAP}")

# ---------------- composition ----------------
hemi_name = {"lh": "Left", "rh": "Right"}
tiles = [f"{hemi_name[h]}, {v}" for h in HEMIS for v in VIEWS]
ncol = len(tiles)
fig, axes = plt.subplots(3, ncol, figsize=(3 * ncol, 7.6), squeeze=False)
fig.subplots_adjust(left=0.01, right=0.89, top=0.93, bottom=0.02, wspace=0.02, hspace=0.18)
for r, row in enumerate(rows):
    for c, img in enumerate(row["imgs"]):
        ax = axes[r, c]
        ax.imshow(img)
        ax.axis("off")
        if r == 0:
            ax.set_title(tiles[c], fontsize=9, pad=14)
    axes[r, 0].text(0, 1.0, row["title"], transform=axes[r, 0].transAxes,
                    fontsize=9.5, fontweight="bold", ha="left", va="bottom")
    # barre de couleur propre a la ligne
    pos = axes[r, ncol - 1].get_position()
    cax = fig.add_axes([0.905, pos.y0 + pos.height * 0.12, 0.010, pos.height * 0.76])
    if row["signed"]:
        cb = fig.colorbar(ScalarMappable(Normalize(-row["lim"], row["lim"]), "RdBu_r"),
                          cax=cax)
        cb.ax.axhspan(-row["thr"], row["thr"], color="0.75")
        ticks = [-row["lim"], -row["thr"], row["thr"], row["lim"]]
    else:
        cb = fig.colorbar(ScalarMappable(Normalize(row["thr"], row["lim"]), CMAP), cax=cax,
                          extend="max" if row["vmax"] > row["lim"] else "neither")
        ticks = [row["thr"], (row["thr"] + row["lim"]) / 2, row["lim"]]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.2f}" if abs(abs(t) - row["thr"]) < 1e-6 else f"{t:.1f}"
                       for t in ticks])
    cb.ax.tick_params(labelsize=7)
    cb.set_label("t(27)", fontsize=8)
import time
for ext in ["png", "pdf"]:
    fname = os.path.abspath(os.path.join(FIGDIR, f"figure_sources.{ext}"))
    try:
        fig.savefig(fname, dpi=300)
    except OSError:
        # fichier ouvert dans une visionneuse (verrouille sous Windows) :
        # on sauvegarde sous un autre nom
        fname = fname.replace(f".{ext}", f"_{time.strftime('%H%M%S')}.{ext}")
        fig.savefig(fname, dpi=300)
    print(f"Figure : {fname}")
plt.close(fig)
