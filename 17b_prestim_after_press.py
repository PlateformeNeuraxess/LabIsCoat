# -*- coding: utf-8 -*-
"""
17b - Controle : fenetre pre-stimulus entierement apres l'appui bouton - Lab Is Coat
===================================================================================

Pourquoi : la fenetre pre-stimulus de l'article (-0.9 a -0.3 s) commence a
l'appui median (-0.9 s). Les ondelettes etalent la puissance sur environ
0.25 s, donc cette fenetre contient un peu d'activite d'avant l'appui.
L'appui le plus precoce a lieu vers -1.0 s et le plus tardif vers -0.8 s :
une fenetre -0.55 a -0.3 s (= -0.8 + 0.25) ne contient plus que de
l'activite posterieure a l'appui chez tous les essais.

Ce script NE RELANCE PAS le script 12 et n'ecrase aucune sortie : il relit
les TFR du script 11 (outputs_tfr/tfr/*_tfr.npz), avec les memes ROI, bandes
et la meme fenetre principale que les scripts 12 et 17, et refait :

  1) REAL vs PICTURE dans la fenetre -0.55/-0.3 s (4 ROI x 4 bandes) :
     les conditions ne doivent toujours pas differer avant le stimulus ;
  2) la lateralisation beta basse (droite - gauche) dans cette fenetre :
     elle doit rester positive (rebond post-mouvement a droite) ;
  3) sa correlation avec l'effet du reel sur mu central droit (0.3-1.75 s) :
     elle doit rester nulle (Spearman, IC bootstrap, BF de Jeffreys).

Sortie : outputs_tfr/prestim_after_press.csv
"""

import os
import glob

import numpy as np
import pandas as pd
import mne
from scipy import stats, integrate

OUTPUT_DIR = "outputs_tfr"
TFR_DIR = os.path.join(OUTPUT_DIR, "tfr")
PRESTIM_WIN = (-0.55, -0.3)          # entierement apres l'appui le plus tardif (+ 0.25 s)
MAIN_WIN = (0.3, 1.75)
BANDS = {"theta": (4, 7), "mu": (8, 13), "beta_low": (14, 20), "beta_high": (21, 30)}
ROI_CHANNELS = {
    "Central gauche": ["E59", "E52", "E51", "E65", "E66", "E58"],
    "Central droit": ["E183", "E184", "E196", "E182", "E164", "E195"],
}
OCC_CENTERS = {"Occipital gauche": "E116", "Occipital droit": "E150"}
N_ROI_CHANNELS = 6
N_BOOT, SEED = 5000, 42

# ---------------- chargement (comme le script 12) ----------------
files = sorted(glob.glob(os.path.join(TFR_DIR, "*_tfr.npz")))
subjects, data, chs = [], [], []
freqs = times = conditions = info0 = None
for f in files:
    z = np.load(f, allow_pickle=False)
    s = os.path.basename(f).replace("_tfr.npz", "")
    if times is None:
        freqs, times, conditions = z["freqs"], z["times"], list(z["conditions"])
        info0 = mne.io.read_info(os.path.join(TFR_DIR, f"{s}-info.fif"), verbose=False)
    subjects.append(s)
    data.append(z["power_db"])
    chs.append([str(c) for c in z["ch_names"]])
common = [c for c in chs[0] if all(c in cs for cs in chs[1:])]
P = np.stack([d[:, [cs.index(c) for c in common]] for d, cs in zip(data, chs)])
info = mne.pick_info(info0, [info0.ch_names.index(c) for c in common])
pos = np.array([info["chs"][i]["loc"][:3] for i in range(len(common))])
ci = {c: i for i, c in enumerate(conditions)}
n = len(subjects)
print(f"{n} sujets ; fenetre pre-stimulus {PRESTIM_WIN} s")

rois = {k: [c for c in v if c in common] for k, v in ROI_CHANNELS.items()}
for name, center in OCC_CENTERS.items():
    d = np.linalg.norm(pos - pos[common.index(center)], axis=1)
    rois[name] = [common[j] for j in np.argsort(d)[:N_ROI_CHANNELS]]


def power(roi, band, win, conds=None):
    """(sujets,) : puissance dB moyennee sur la ROI, la bande, la fenetre et les conditions."""
    X = P[:, :, [common.index(c) for c in rois[roi]]].mean(axis=2)
    fm = (freqs >= BANDS[band][0]) & (freqs <= BANDS[band][1])
    tm = (times >= win[0]) & (times <= win[1])
    X = X[:, :, fm][..., tm].mean(axis=(2, 3))                    # (sujets, conditions)
    idx = [ci[c] for c in (conds or conditions)]
    return X[:, idx].mean(axis=1)


REAL, VIRT = ["REAL_APP", "REAL_DISG"], ["VIRT_APP", "VIRT_DISG"]
rows = []

# 1) REAL vs PICTURE avant le stimulus
print("\n1) REAL vs PICTURE, pre-stimulus :")
for roi in rois:
    for band in BANDS:
        d = power(roi, band, PRESTIM_WIN, REAL) - power(roi, band, PRESTIM_WIN, VIRT)
        t, p = stats.ttest_1samp(d, 0)
        rows.append({"test": "REAL-VIRT prestim", "roi": roi, "band": band,
                     "mean": d.mean(), "t": t, "p": p, "dz": d.mean() / d.std(ddof=1)})
res1 = pd.DataFrame(rows)
print(f"   max |t({n-1})| = {res1.t.abs().max():.2f}, min p = {res1.p.min():.3f}")

# 2) lateralisation beta basse
lat = (power("Central droit", "beta_low", PRESTIM_WIN)
       - power("Central gauche", "beta_low", PRESTIM_WIN))
t, p = stats.ttest_1samp(lat, 0)
print(f"\n2) Lateralisation beta basse (droite - gauche) : {lat.mean():+.2f} ± "
      f"{lat.std(ddof=1):.2f} dB, t({n-1}) = {t:.2f}, p = {p:.4f}, "
      f"{(lat > 0).sum()}/{n} participants > 0")
rows.append({"test": "LAT_BETA_LOW prestim", "roi": "droit - gauche", "band": "beta_low",
             "mean": lat.mean(), "t": t, "p": p, "dz": lat.mean() / lat.std(ddof=1)})

# 3) correlation avec l'effet du reel sur mu central droit (fenetre principale)
mu_r = (power("Central droit", "mu", MAIN_WIN, REAL)
        - power("Central droit", "mu", MAIN_WIN, VIRT))


def bf10_correlation(r, n):
    lik = lambda rho: (1 - rho ** 2) ** ((n - 1) / 2) / (1 - rho * r) ** ((2 * n - 3) / 2)
    return 0.5 * integrate.quad(lik, -1, 1, limit=200)[0]


rho, p = stats.spearmanr(lat, mu_r)
rng = np.random.default_rng(SEED)
boot = []
for _ in range(N_BOOT):
    i = rng.integers(0, n, n)
    boot.append(stats.spearmanr(lat[i], mu_r[i]).statistic)
lo, hi = np.nanpercentile(boot, [2.5, 97.5])
bf = bf10_correlation(rho, n)
print(f"\n3) Correlation lateralisation x effet mu : rho = {rho:.3f} [{lo:.2f}, {hi:.2f}], "
      f"p = {p:.3f}, BF10 = {bf:.2f}, BF01 = {1/bf:.2f}")
rows.append({"test": "rho LAT_BETA_LOW x MU_R", "roi": "", "band": "", "mean": rho,
             "t": np.nan, "p": p, "dz": np.nan, "ci_low": lo, "ci_high": hi, "BF01": 1 / bf})

pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "prestim_after_press.csv"), index=False)
print(f"\nSortie : {os.path.join(OUTPUT_DIR, 'prestim_after_press.csv')}")
