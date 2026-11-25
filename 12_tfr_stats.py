# -*- coding: utf-8 -*-
"""
12 - Statistiques et figures temps-fréquence - Lab Is Coat

Entrée : sorties de 11_tfr_extract.py (outputs_tfr/tfr/*_tfr.npz + *-info.fif)

Analyses :
  A) Confirmatoire (hypothèse a priori, Marini et al. 2019) :
     puissance moyenne (dB / baseline) dans la bande mu 8-13 Hz (et bêta),
     ROI centrales gauche / droite (C3-CP3 / C4-CP4, mêmes électrodes que
     09_erp_9roi.py), fenêtre ANALYSIS_WIN.
     ANOVA à mesures répétées Réalité x Valence x Hémisphère.
     Contrôle : mêmes mesures sur ROI occipitales (O1 / O2) -> un effet du réel
     purement visuel apparaîtrait surtout en occipital.
  B) Exploratoire : tests par permutation de clusters temps x fréquence, par ROI,
     pour l'effet Réalité (REAL - VIRT, valence moyennée) et l'interaction
     (REAL_APP - VIRT_APP) - (REAL_DISG - VIRT_DISG).
  C) Exploratoire : clusters spatio-temporels (canaux x temps) dans la bande mu,
     effet Réalité.

Sorties (OUTPUT_DIR) :
  tfr_roi_band_windows.csv     puissance par sujet / condition / ROI / bande / fenêtre
  tfr_anova.txt                ANOVA RM (fenêtre principale) + t appariés REAL vs VIRT
  tfr_clusters_tf.csv          clusters temps x fréquence (B)
  tfr_clusters_spatial.csv     clusters canaux x temps (C)
  figures/tfr_roi_maps.png     cartes TF par ROI : REAL, VIRT, REAL-VIRT (+ contours clusters)
  figures/tfr_band_timecourses.png  décours mu / bêta, 4 conditions ± SEM
  figures/tfr_mu_topomaps.png  topographies mu REAL-VIRT par fenêtre de 250 ms
"""

import os
import glob

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy import sparse, stats
from mne.stats import permutation_cluster_1samp_test, spatio_temporal_cluster_1samp_test
import inspect

# 'rng' (MNE >= 1.12) ou 'seed' (versions antérieures)
_RNG_KW = "rng" if "rng" in inspect.signature(permutation_cluster_1samp_test).parameters else "seed"


# ==========================================================
# PARAMÈTRES
# ==========================================================

OUTPUT_DIR = "outputs_tfr"
TFR_DIR = os.path.join(OUTPUT_DIR, "tfr")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

BANDS = {
    "theta": (4, 7),
    "mu": (8, 13),
    "beta_low": (14, 20),
    "beta_high": (21, 30),
}

# Fenêtre principale (s) : après la zone contaminée par l'artefact d'ouverture (0 s)
# et avant celle de la fermeture de l'écran (+2 s)
ANALYSIS_WIN = (0.3, 1.75)
# Contrôle pré-stimulus (entre l'appui bouton ~ -1 s et le VIS) : avec une baseline
# commune, une différence REAL - VIRT ici signalerait une anticipation de la condition.
PRESTIM_WIN = (-0.9, -0.3)
# Délai typique de l'appui bouton (trait vertical sur les figures)
PRESS_TIME = -0.9
# Fenêtres descriptives
BIN_EDGES = np.arange(0.0, 1.7501, 0.25)

# ROI centrales : mêmes électrodes que 09_erp_9roi.py
ROI_CHANNELS = {
    "Central gauche": ["E59", "E52", "E51", "E65", "E66", "E58"],        # C3, CP3
    "Central droit": ["E183", "E184", "E196", "E182", "E164", "E195"],   # C4, CP4
}
# ROI occipitales de contrôle : électrode centrale (Luu & Ferree 2005)
# + ses voisines les plus proches (calculées sur les positions réelles)
OCC_CENTERS = {"Occipital gauche": "E116", "Occipital droit": "E150"}   # O1, O2
N_ROI_CHANNELS = 6
MIN_CHANNELS_PER_ROI = 4

# Clusters
N_PERMUTATIONS = 5000
CLUSTER_P = 0.05            # seuil t "formant" les clusters (bilatéral)
CLUSTER_WIN = (0.0, 1.75)   # fenêtre testée : avant que la fermeture de l'écran (+2 s)
                            # ne contamine la TFR (demi-ondelette 0.25 s à 4 Hz)
SEED = 42
NEIGHBOR_FACTOR = 1.6       # adjacence : distance < facteur x distance médiane au plus proche voisin

CONTRASTS = {
    "Reality (REAL-VIRT)": {"REAL_APP": .5, "REAL_DISG": .5, "VIRT_APP": -.5, "VIRT_DISG": -.5},
    "Interaction R x V": {"REAL_APP": 1, "VIRT_APP": -1, "REAL_DISG": -1, "VIRT_DISG": 1},
}

STYLE = {
    "REAL_APP": dict(color="crimson", ls="-"),
    "VIRT_APP": dict(color="steelblue", ls="-"),
    "REAL_DISG": dict(color="crimson", ls="--"),
    "VIRT_DISG": dict(color="steelblue", ls="--"),
}


# ==========================================================
# CHARGEMENT
# ==========================================================

files = sorted(glob.glob(os.path.join(TFR_DIR, "*_tfr.npz")))
if not files:
    raise SystemExit(f"Aucun *_tfr.npz dans {TFR_DIR} : lancer d'abord 11_tfr_extract.py")

subjects, data, chs = [], [], []
freqs = times = conditions = None
info0 = None
for f in files:
    z = np.load(f, allow_pickle=False)
    s = os.path.basename(f).replace("_tfr.npz", "")
    if times is None:
        freqs, times, conditions = z["freqs"], z["times"], list(z["conditions"])
        info0 = mne.io.read_info(os.path.join(TFR_DIR, f"{s}-info.fif"), verbose=False)
    elif len(z["times"]) != len(times) or not np.allclose(z["times"], times):
        raise ValueError(f"{s} : axe temporel différent (relancer 11 avec les mêmes paramètres)")
    subjects.append(s)
    data.append(z["power_db"])
    chs.append(list(z["ch_names"]))

common = [c for c in chs[0] if all(c in cs for cs in chs[1:])]
if len(common) < len(chs[0]):
    print(f"Canaux communs à tous les sujets : {len(common)}")
P = np.stack([d[:, [cs.index(c) for c in common]] for d, cs in zip(data, chs)])
# P : (sujets, conditions, canaux, fréquences, temps)
info = mne.pick_info(info0, [info0.ch_names.index(c) for c in common])
n_subj = len(subjects)
print(f"{n_subj} sujets, {len(common)} canaux, {len(freqs)} fréquences, {len(times)} instants")

ci = {c: i for i, c in enumerate(conditions)}
pos = np.array([info["chs"][i]["loc"][:3] for i in range(len(common))])


# ==========================================================
# ROI
# ==========================================================

def nearest(center, n):
    i = common.index(center)
    d = np.linalg.norm(pos - pos[i], axis=1)
    return [common[j] for j in np.argsort(d)[:n]]


rois = {}
for name, lst in ROI_CHANNELS.items():
    rois[name] = [c for c in lst if c in common]
for name, center in OCC_CENTERS.items():
    rois[name] = nearest(center, N_ROI_CHANNELS) if center in common else []

for name, lst in list(rois.items()):
    print(f"ROI {name:17s}: {lst}")
    if len(lst) < MIN_CHANNELS_PER_ROI:
        print(f"   ⚠️ moins de {MIN_CHANNELS_PER_ROI} électrodes -> ROI ignorée")
        rois.pop(name)

roi_idx = {k: [common.index(c) for c in v] for k, v in rois.items()}


def roi_tf(roi):
    """(sujets, conditions, freq, temps) moyenné sur la ROI."""
    return P[:, :, roi_idx[roi]].mean(axis=2)


def contrast(arr, weights):
    """arr : (sujets, conditions, ...) -> (sujets, ...)."""
    return sum(w * arr[:, ci[c]] for c, w in weights.items())


def fmask(band):
    lo, hi = BANDS[band]
    return (freqs >= lo) & (freqs <= hi)


def tmask(win):
    return (times >= win[0]) & (times <= win[1])


# ==========================================================
# A) ROI x bande x fenêtre + ANOVA
# ==========================================================

rows = []
windows = {"main": ANALYSIS_WIN, "prestim": PRESTIM_WIN}
for a, b in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
    windows[f"{int(a*1000)}-{int(b*1000)}"] = (a, b)

for roi in rois:
    X = roi_tf(roi)
    for band in BANDS:
        Xb = X[:, :, fmask(band)].mean(axis=2)            # (sujets, cond, temps)
        for wname, win in windows.items():
            v = Xb[:, :, tmask(win)].mean(axis=2)
            for si, s in enumerate(subjects):
                for c in conditions:
                    rows.append({
                        "subject": s, "condition": c,
                        "Reality": "REAL" if "REAL" in c else "VIRTUAL",
                        "Valence": "APP" if "APP" in c else "DISG",
                        "roi": roi,
                        "Region": "Central" if roi.startswith("Central") else "Occipital",
                        "Hemisphere": "Left" if "gauche" in roi else "Right",
                        "band": band, "window": wname,
                        "t_start": win[0], "t_end": win[1],
                        "power_db": float(v[si, ci[c]]),
                    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUTPUT_DIR, "tfr_roi_band_windows.csv"), index=False)
print("\nTableau ROI x bande x fenêtre sauvegardé")

lines = [f"Fenêtre principale : {ANALYSIS_WIN[0]}-{ANALYSIS_WIN[1]} s, n = {n_subj} sujets\n"]
try:
    from statsmodels.stats.anova import AnovaRM
    for region in ["Central", "Occipital"]:
        for band in BANDS:
            d = df[(df.window == "main") & (df.band == band) & (df.Region == region)]
            if d.empty:
                continue
            res = AnovaRM(d, depvar="power_db", subject="subject",
                          within=["Reality", "Valence", "Hemisphere"]).fit()
            lines.append(f"\n===== {region} - {band} {BANDS[band]} Hz =====\n")
            lines.append(res.anova_table.round(4).to_string())
            lines.append("\n")
except ImportError:
    lines.append("statsmodels absent : ANOVA non calculée (pip install statsmodels)\n")

for wname, wlabel in [("main", f"fenêtre principale {ANALYSIS_WIN}"),
                      ("prestim", f"CONTRÔLE pré-stimulus {PRESTIM_WIN}")]:
    lines.append(f"\n===== t appariés REAL vs VIRTUAL (valence moyennée), {wlabel} =====\n")
    for roi in rois:
        for band in BANDS:
            d = df[(df.window == wname) & (df.band == band) & (df.roi == roi)]
            m = d.groupby(["subject", "Reality"]).power_db.mean().unstack()
            t, p = stats.ttest_rel(m["REAL"], m["VIRTUAL"])
            dz = (m["REAL"] - m["VIRTUAL"]).mean() / (m["REAL"] - m["VIRTUAL"]).std(ddof=1)
            lines.append(f"{roi:17s} {band:9s} REAL {m['REAL'].mean():6.2f} dB | "
                         f"VIRT {m['VIRTUAL'].mean():6.2f} dB | t({n_subj-1}) = {t:6.2f}, "
                         f"p = {p:.4f}, dz = {dz:5.2f}\n")

with open(os.path.join(OUTPUT_DIR, "tfr_anova.txt"), "w", encoding="utf-8") as fh:
    fh.writelines(lines)
print("".join(lines[-2 * (len(rois) * len(BANDS) + 1):]))


# ==========================================================
# B) Clusters temps x fréquence par ROI
# ==========================================================

cw = tmask(CLUSTER_WIN)
t_thresh = stats.t.ppf(1 - CLUSTER_P / 2, n_subj - 1)
tf_rows, tf_masks = [], {}

for roi in rois:
    X = roi_tf(roi)[..., cw]
    for cname, w in CONTRASTS.items():
        D = contrast(X, w)                                  # (sujets, freq, temps)
        T, clusters, pv, _ = permutation_cluster_1samp_test(
            D, threshold=t_thresh, n_permutations=N_PERMUTATIONS, tail=0,
            out_type="mask", verbose=False, **{_RNG_KW: SEED})
        sig = np.zeros_like(T, dtype=bool)
        for k, (cl, p) in enumerate(zip(clusters, pv)):
            ff, tt = np.where(cl)
            tf_rows.append({
                "roi": roi, "contrast": cname, "cluster": k, "p": p,
                "sign": "positive" if T[cl].mean() > 0 else "negative",
                "f_min": freqs[ff.min()], "f_max": freqs[ff.max()],
                "t_min": times[cw][tt.min()], "t_max": times[cw][tt.max()],
                "sum_t": T[cl].sum(), "mean_diff_db": D[:, cl].mean(),
            })
            if p < 0.05:
                sig |= cl
        tf_masks[(roi, cname)] = sig

tf_df = pd.DataFrame(tf_rows).sort_values(["roi", "contrast", "p"]) if tf_rows else pd.DataFrame()
tf_df.to_csv(os.path.join(OUTPUT_DIR, "tfr_clusters_tf.csv"), index=False)
print("\nClusters temps x fréquence (p < 0.05) :")
print(tf_df[tf_df.p < 0.05].to_string(index=False) if len(tf_df) else "   aucun")


# ==========================================================
# C) Clusters canaux x temps, bande mu, effet Réalité
# ==========================================================

d_nn = np.linalg.norm(pos[:, None] - pos[None], axis=2)
np.fill_diagonal(d_nn, np.inf)
radius = NEIGHBOR_FACTOR * np.median(d_nn.min(axis=1))
adjacency = sparse.csr_matrix(d_nn < radius)
print(f"\nAdjacence : {adjacency.sum(axis=1).mean():.1f} voisins en moyenne par électrode")

mu = P[:, :, :, fmask("mu")].mean(axis=3)[..., cw]          # (sujets, cond, canaux, temps)
D_mu = contrast(mu, CONTRASTS["Reality (REAL-VIRT)"]).transpose(0, 2, 1)  # (sujets, temps, canaux)
T_sp, cl_sp, pv_sp, _ = spatio_temporal_cluster_1samp_test(
    D_mu, threshold=t_thresh, n_permutations=N_PERMUTATIONS, tail=0,
    adjacency=adjacency, out_type="mask", verbose=False, **{_RNG_KW: SEED})

sp_rows, sig_sp = [], np.zeros_like(T_sp, dtype=bool)
for k, (cl, p) in enumerate(zip(cl_sp, pv_sp)):
    tt, cc = np.where(cl)
    sp_rows.append({
        "cluster": k, "p": p, "sign": "positive" if T_sp[cl].mean() > 0 else "negative",
        "t_min": times[cw][tt.min()], "t_max": times[cw][tt.max()],
        "n_channels": len(np.unique(cc)),
        "channels": " ".join(sorted({common[c] for c in cc}, key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)),
        "sum_t": T_sp[cl].sum(),
    })
    if p < 0.05:
        sig_sp |= cl
sp_df = pd.DataFrame(sp_rows).sort_values("p") if sp_rows else pd.DataFrame()
sp_df.to_csv(os.path.join(OUTPUT_DIR, "tfr_clusters_spatial.csv"), index=False)
print("Clusters spatio-temporels mu (p < 0.05) :")
print(sp_df[sp_df.p < 0.05].drop(columns="channels").to_string(index=False) if len(sp_df) else "   aucun")


# ==========================================================
# FIGURES
# ==========================================================

# 1) Cartes TF par ROI
roi_list = list(rois)
fig, axes = plt.subplots(len(roi_list), 3, figsize=(13, 3.0 * len(roi_list)),
                         sharex=True, sharey=True, squeeze=False)
real = {k: .5 for k in ["REAL_APP", "REAL_DISG"]}
virt = {k: .5 for k in ["VIRT_APP", "VIRT_DISG"]}
ext = [times[0], times[-1], freqs[0] - .5, freqs[-1] + .5]
vmax = 3.0
for r, roi in enumerate(roi_list):
    X = roi_tf(roi)
    maps = [contrast(X, real).mean(0), contrast(X, virt).mean(0),
            contrast(X, CONTRASTS["Reality (REAL-VIRT)"]).mean(0)]
    titles = ["REAL", "VIRTUAL", "REAL - VIRTUAL"]
    for c in range(3):
        ax = axes[r, c]
        lim = vmax if c < 2 else vmax / 2
        im = ax.imshow(maps[c], aspect="auto", origin="lower", extent=ext,
                       cmap="RdBu_r", vmin=-lim, vmax=lim)
        if c == 2:
            sig = np.zeros_like(maps[c], dtype=bool)
            sig[:, cw] = tf_masks[(roi, "Reality (REAL-VIRT)")]
            if sig.any():
                ax.contour(times, freqs, sig, levels=[.5], colors="k", linewidths=1.2)
        ax.axvline(0, color="k", lw=.6, ls=":")
        ax.axvline(PRESS_TIME, color="dimgrey", lw=.8, ls="-.")
        ax.axvline(2.0, color="k", lw=.6, ls=":")
        for lo, hi in [BANDS["mu"]]:
            ax.axhline(lo, color="grey", lw=.5, ls="--")
            ax.axhline(hi, color="grey", lw=.5, ls="--")
        ax.set_title(f"{roi} - {titles[c]}", fontsize=10)
        if c == 0:
            ax.set_ylabel("Fréquence (Hz)")
        if r == len(roi_list) - 1:
            ax.set_xlabel("Temps (s)")
        plt.colorbar(im, ax=ax, fraction=.046, pad=.02, label="dB")
fig.suptitle(f"Puissance relative à la baseline ({n_subj} sujets) - contours : clusters p<0.05"
             " - tirets : appui bouton",
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "tfr_roi_maps.png"), dpi=300)
plt.close(fig)

# 2) Décours temporels mu / bêta
plot_bands = ["mu", "beta_low"]
fig, axes = plt.subplots(len(plot_bands), len(roi_list), figsize=(4 * len(roi_list), 3.4 * len(plot_bands)),
                         sharex=True, sharey="row", squeeze=False)
for r, band in enumerate(plot_bands):
    for c, roi in enumerate(roi_list):
        ax = axes[r, c]
        Xb = roi_tf(roi)[:, :, fmask(band)].mean(axis=2)
        for cond in conditions:
            m = Xb[:, ci[cond]].mean(0)
            se = Xb[:, ci[cond]].std(0, ddof=1) / np.sqrt(n_subj)
            ax.plot(times, m, label=cond, lw=1.5, **STYLE[cond])
            ax.fill_between(times, m - se, m + se, color=STYLE[cond]["color"], alpha=.12, lw=0)
        ax.axhline(0, color="k", lw=.6)
        ax.axvline(0, color="grey", lw=.8, ls=":")
        ax.axvspan(*ANALYSIS_WIN, color="gold", alpha=.08)
        ax.axvline(PRESS_TIME, color="dimgrey", lw=.8, ls="-.")
        ax.axvline(2.0, color="k", lw=.6, ls=":")
        ax.set_title(f"{roi} - {band} {BANDS[band]} Hz", fontsize=10)
        if c == 0:
            ax.set_ylabel("Puissance (dB)")
        if r == len(plot_bands) - 1:
            ax.set_xlabel("Temps (s)")
        ax.grid(alpha=.2)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
fig.tight_layout(rect=(0, .05, 1, 1))
fig.savefig(os.path.join(FIG_DIR, "tfr_band_timecourses.png"), dpi=300)
plt.close(fig)

# 3) Topographies mu REAL - VIRT
tw = times[cw]
nb = len(BIN_EDGES) - 1
fig, axes = plt.subplots(1, nb, figsize=(2.2 * nb, 2.8))
mean_diff = D_mu.mean(0)                                    # (temps, canaux)
lim = np.abs(mean_diff).max()
for k, (a, b) in enumerate(zip(BIN_EDGES[:-1], BIN_EDGES[1:])):
    m = (tw >= a) & (tw < b)
    mk = sig_sp[m].any(axis=0)
    mne.viz.plot_topomap(mean_diff[m].mean(0), info, axes=axes[k], show=False,
                         cmap="RdBu_r", vlim=(-lim, lim), contours=0,
                         mask=mk, mask_params=dict(marker="o", markerfacecolor="k",
                                                   markersize=3, markeredgewidth=0))
    axes[k].set_title(f"{int(a*1000)}-{int(b*1000)} ms", fontsize=9)
fig.suptitle("Mu 8-13 Hz : REAL - VIRTUAL (dB) - points : cluster spatio-temporel p<0.05",
             fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "tfr_mu_topomaps.png"), dpi=300)
plt.close(fig)

print(f"\nFigures sauvegardées dans {FIG_DIR}")
print("Terminé.")
