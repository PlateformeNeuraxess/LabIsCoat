# -*- coding: utf-8 -*-
"""
17 - Etat sensorimoteur pre-stimulus x effet mu du reel - Lab Is Coat
=====================================================================

Question (Discussion, "Comparison with Marini et al." et S1) : l'effet du
reel sur la puissance mu centrale droite (0.3-1.75 s) depend-il de l'etat
moteur lateralise present AVANT le stimulus ? A l'ouverture de l'ecran,
l'hemisphere droit (controlateral a la main gauche qui vient d'appuyer)
montre un rebond beta, et l'hemisphere gauche (main droite prete a
repondre) une desynchronisation. Si l'effet du reel se construit sur cet
etat, les participants les plus lateralises devraient avoir l'effet le
plus grand.

Indices par participant (toutes conditions moyennees pour le pre-stimulus,
puisque l'etat pre-stimulus ne differe pas entre conditions) :

  Predicteur principal (defini a l'avance, un seul test) :
    LAT_BETA_LOW  = beta basse (14-20 Hz), pre-stimulus (-0.9 a -0.3 s),
                    central droit - central gauche (dB)
                    > 0 : rebond droit et/ou desynchronisation gauche

  Predicteurs secondaires (exploratoires, non corriges) :
    LAT_BETA_HIGH = idem en beta haute (21-30 Hz)
    LAT_MU        = idem en mu (8-13 Hz)
    R_BETA_LOW    = beta basse centrale droite seule (rebond post-mouvement)
    R_MU_PRE      = mu central droit seul, pre-stimulus (niveau de depart)

  Variable dependante :
    MU_R = mu central droit, fenetre principale (0.3-1.75 s),
           (REAL_APP + REAL_DISG)/2 - (VIRT_APP + VIRT_DISG)/2 (dB)
    (+ MU_R_APP et MU_R_DISG, exploratoires)

Statistiques : rho de Spearman, IC 95 % bootstrap (5000 tirages), BF10 de
Jeffreys (a priori uniforme sur rho ; Ly, Verhagen & Wagenmakers, 2016),
comme dans le script 14.

Entree : outputs_tfr/tfr_roi_band_windows.csv (script 12)
Sorties (outputs_tfr) :
    prestim_mu_indices.csv        indices par participant
    prestim_mu_correlations.csv   correlations
    figures/prestim_mu.png        nuage de points du test principal
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, integrate

OUTPUT_DIR = "outputs_tfr"
IN_FILE = os.path.join(OUTPUT_DIR, "tfr_roi_band_windows.csv")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

N_BOOT = 5000
SEED = 42
RIGHT, LEFT = "Central droit", "Central gauche"


# ==========================================================
# OUTILS (identiques au script 14)
# ==========================================================

def bf10_correlation(r, n):
    """BF10 pour une correlation (Jeffreys, a priori uniforme sur rho)."""
    def lik(rho):
        return (1 - rho ** 2) ** ((n - 1) / 2) / (1 - rho * r) ** ((2 * n - 3) / 2)
    val, _ = integrate.quad(lik, -1, 1, limit=200)
    return 0.5 * val


def boot_ci(x, y, rng):
    n = len(x)
    vals = np.empty(N_BOOT)
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        vals[b] = stats.spearmanr(x[i], y[i]).statistic
    vals = vals[np.isfinite(vals)]
    return np.percentile(vals, [2.5, 97.5])


# ==========================================================
# INDICES PAR PARTICIPANT
# ==========================================================

df = pd.read_csv(IN_FILE)


def power(roi, band, window, conds=None):
    """Puissance par sujet (moyenne sur les conditions donnees, ou toutes)."""
    d = df[(df.roi == roi) & (df.band == band) & (df.window == window)]
    if conds is not None:
        d = d[d.condition.isin(conds)]
    return d.groupby("subject").power_db.mean()


REAL = ["REAL_APP", "REAL_DISG"]
VIRT = ["VIRT_APP", "VIRT_DISG"]

ind = pd.DataFrame({
    "LAT_BETA_LOW": power(RIGHT, "beta_low", "prestim") - power(LEFT, "beta_low", "prestim"),
    "LAT_BETA_HIGH": power(RIGHT, "beta_high", "prestim") - power(LEFT, "beta_high", "prestim"),
    "LAT_MU": power(RIGHT, "mu", "prestim") - power(LEFT, "mu", "prestim"),
    "R_BETA_LOW": power(RIGHT, "beta_low", "prestim"),
    "R_MU_PRE": power(RIGHT, "mu", "prestim"),
    "MU_R": power(RIGHT, "mu", "main", REAL) - power(RIGHT, "mu", "main", VIRT),
    "MU_R_APP": power(RIGHT, "mu", "main", ["REAL_APP"]) - power(RIGHT, "mu", "main", ["VIRT_APP"]),
    "MU_R_DISG": power(RIGHT, "mu", "main", ["REAL_DISG"]) - power(RIGHT, "mu", "main", ["VIRT_DISG"]),
}).dropna()
n = len(ind)
ind.to_csv(os.path.join(OUTPUT_DIR, "prestim_mu_indices.csv"))

print(f"{n} participants")
print("\nEtat pre-stimulus (moyenne ± ET ; test contre 0) :")
for c in ["LAT_BETA_LOW", "LAT_BETA_HIGH", "LAT_MU", "R_BETA_LOW", "R_MU_PRE"]:
    t, p = stats.ttest_1samp(ind[c], 0)
    n_pos = (ind[c] > 0).sum()
    print(f"  {c:14s} {ind[c].mean():+.2f} ± {ind[c].std(ddof=1):.2f} dB, "
          f"t({n-1}) = {t:.2f}, p = {p:.3g} ; {n_pos}/{n} participants > 0")


# ==========================================================
# CORRELATIONS
# ==========================================================

rng = np.random.default_rng(SEED)
tests = [("LAT_BETA_LOW", "MU_R", "principal")]
for pred in ["LAT_BETA_HIGH", "LAT_MU", "R_BETA_LOW", "R_MU_PRE"]:
    tests.append((pred, "MU_R", "exploratoire"))
for dv in ["MU_R_APP", "MU_R_DISG"]:
    tests.append(("LAT_BETA_LOW", dv, "exploratoire"))

rows = []
for x, y, status in tests:
    r = stats.spearmanr(ind[x], ind[y])
    lo, hi = boot_ci(ind[x].values, ind[y].values, rng)
    bf = bf10_correlation(r.statistic, n)
    rows.append({"predictor": x, "outcome": y, "status": status, "n": n,
                 "rho": r.statistic, "ci_low": lo, "ci_high": hi, "p": r.pvalue,
                 "BF10": bf, "BF01": 1 / bf})
res = pd.DataFrame(rows)
res["evidence"] = pd.cut(res.BF10, [0, 1 / 3, 3, np.inf],
                         labels=["pour H0 (BF01>3)", "non concluant", "pour H1 (BF10>3)"])
res.to_csv(os.path.join(OUTPUT_DIR, "prestim_mu_correlations.csv"), index=False)

pd.set_option("display.width", 180)
print("\nCorrelations (Spearman) :")
print(res.round(3).to_string(index=False))


# ==========================================================
# FIGURE (test principal)
# ==========================================================

r0 = res.iloc[0]
fig, ax = plt.subplots(figsize=(4.6, 4))
ax.scatter(ind.LAT_BETA_LOW, ind.MU_R, s=30, color="#2a6db0", edgecolor="white", lw=.8, zorder=3)
k, c0 = np.polyfit(ind.LAT_BETA_LOW, ind.MU_R, 1)
xs = np.linspace(ind.LAT_BETA_LOW.min(), ind.LAT_BETA_LOW.max(), 50)
ax.plot(xs, k * xs + c0, color="#2a6db0", lw=1.5)
ax.axhline(0, color="grey", lw=.6)
ax.axvline(0, color="grey", lw=.6)
ax.set_xlabel("Pre-stimulus low-beta lateralization,\nright − left central (dB)")
ax.set_ylabel("μ power, real − picture,\nright central (dB)")
ax.set_title(f"ρ = {r0.rho:.2f} [{r0.ci_low:.2f}, {r0.ci_high:.2f}], p = {r0.p:.3f}, "
             f"BF10 = {r0.BF10:.2f}", fontsize=9)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "prestim_mu.png"), dpi=300)
plt.close(fig)
print(f"\nSorties dans {OUTPUT_DIR}")
