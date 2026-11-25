# -*- coding: utf-8 -*-

"""
15 - Facteurs de Bayes et tests d'equivalence - Lab Is Coat
===========================================================

Objectif : quantifier les resultats NULS sur lesquels repose le titre
("Dissociable neural signatures of emotional value and physical reality
in food perception") :
    - absence d'interaction Reality x Valence (mu, GFP, micro-etats) ;
    - absence d'effet Reality sur les reponses evoquees (GFP, micro-etats).
Une p > .05 ne montre pas l'absence d'effet ; le facteur de Bayes BF01 et
le test d'equivalence (TOST) le peuvent.

Methode
-------
Plan 2 x 2 intra-sujet complet -> chaque effet est un contraste par sujet,
et le test de l'effet est un t a un echantillon sur ce contraste
(equivalent au test F a 1 ddl de l'ANOVA a mesures repetees et au modele
mixte des scripts 04/07 quand les donnees sont completes) :
    Reality     = (RA + RD)/2 - (VA + VD)/2
    Valence     = (RD + VD)/2 - (RA + VA)/2
    Interaction = (RA - VA) - (RD - VD)     (> 0 : effet du reel plus fort
                                             pour l'appetissant)

- BF10 : test t bayesien JZS (Rouder et al., 2009), a priori de Cauchy
  sur la taille d'effet, echelle r = 0.707 (defaut de JASP). Robustesse :
  r = 0.5 et r = 1. BF01 = 1/BF10. Lecture (Jeffreys / Lee & Wagenmakers) :
  BF01 > 3 = soutien modere a H0, > 10 = fort ; 1/3 < BF < 3 = non concluant.
- TOST (Lakens, 2017) : bornes d'equivalence en dz = +/- EQ_BOUND_DZ
  (0.5 par defaut = effet moyen ; pour reference, l'effet mu du reel vaut
  dz = 0.68). p_TOST < .05 = effet significativement plus petit que la
  borne.
- dz = moyenne / ecart-type du contraste ; IC 90 % (coherent avec TOST).

Mesures (sorties des scripts precedents, lancer depuis le meme dossier)
    MU_R      : puissance mu, central droit, fenetre principale (script 12)
    GFP_PEAK  : amplitude du pic de GFP global (script 03)
    GFP_<fen> : GFP moyen par fenetre de la grande moyenne (script 07)
    MSk_<par> : couverture, occurrence, duree des micro-etats (script 03),
                MS2 exclu (carte de bruit)

Sorties (OUTDIR)
    bayes_equivalence.csv     tous les effets x toutes les mesures
    bayes_summary.txt         resume des mesures principales
    bayes_forest.png          BF01 (echelle log) pour Reality et Interaction
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, integrate


# ==========================================================
# PARAMETRES
# ==========================================================

TFR_FILE = os.path.join("outputs_tfr", "tfr_roi_band_windows.csv")
GFP_PEAK_FILE = os.path.join("outputs_microstates", "gfp_peak_latency.csv")
MS_PARAMS_FILE = os.path.join("outputs_microstates", "microstate_parameters.csv")
GFP_WINDOWS_FILE = "microstate_parameters_fixed_window.csv"

ROI, BAND, WINDOW = "Central droit", "mu", "main"
MS_PARAMS = ["coverage_percent", "occurrence_per_sec", "mean_duration_ms"]
MS_EXCLUDED = ["MS2"]

EXCLUDED_INCLUSIONS = [12]   # sujet 12 : exclu du temps-frequence

PRIOR_R = 0.707
PRIOR_R_ROBUST = [0.5, 1.0]
EQ_BOUND_DZ = 0.5

# Mesures "principales" (resume) : (mesure, effet)
PRIMARY = [
    ("MU_R", "Interaction"),
    ("GFP_PEAK", "Interaction"),
    ("GFP_PEAK", "Reality"),
]   # + toutes les fenetres GFP pour Reality et Interaction (ajoutees plus bas)

OUTDIR = "outputs_bayes"
os.makedirs(OUTDIR, exist_ok=True)

CONDS = ["REAL_APP", "VIRT_APP", "REAL_DISG", "VIRT_DISG"]


# ==========================================================
# OUTILS
# ==========================================================

def inclusion_number(code):
    s = str(code)
    m = re.search(r"[A-Za-z]{2}(\d{2})", s) or re.search(r"(\d+)\s*$", s)
    return int(m.group(1)) if m else np.nan


def jzs_bf10(t, n, r=0.707):
    """BF10 du test t a un echantillon, a priori JZS (Cauchy, echelle r ;
    Rouder et al., 2009, eq. 1)."""
    nu = n - 1

    def integrand(g):
        return ((1 + n * g * r ** 2) ** -0.5
                * (1 + t ** 2 / ((1 + n * g * r ** 2) * nu)) ** (-(nu + 1) / 2)
                * (2 * np.pi) ** -0.5 * g ** -1.5 * np.exp(-1 / (2 * g)))

    num, _ = integrate.quad(integrand, 0, np.inf, limit=500)
    den = (1 + t ** 2 / nu) ** (-(nu + 1) / 2)
    return num / den


def tost_p(x, bound_dz):
    """TOST sur la moyenne d'un contraste, bornes +/- bound_dz * SD."""
    n, m, sd = len(x), x.mean(), x.std(ddof=1)
    se = sd / np.sqrt(n)
    delta = bound_dz * sd
    p_low = stats.t.sf((m + delta) / se, n - 1)     # H0 : mu <= -delta
    p_high = stats.t.cdf((m - delta) / se, n - 1)   # H0 : mu >= +delta
    return max(p_low, p_high)


def dz_ci90(t, n):
    """IC 90 % de dz par la loi t non centree."""
    from scipy.optimize import brentq
    df = n - 1
    f = lambda nc, q: stats.nct.cdf(t, df, nc) - q
    try:
        lo = brentq(f, t - 8, t + 8, args=(0.95,)) / np.sqrt(n)
        hi = brentq(f, t - 8, t + 8, args=(0.05,)) / np.sqrt(n)
        return lo, hi
    except (ValueError, RuntimeError):
        # approximation normale (t tres grand)
        dz = t / np.sqrt(n)
        se = np.sqrt(1 / n + dz ** 2 / (2 * n))
        return dz - 1.645 * se, dz + 1.645 * se


def contrasts(w):
    """w : DataFrame sujets x 4 conditions -> dict effet -> Series."""
    return {
        "Reality": (w.REAL_APP + w.REAL_DISG) / 2 - (w.VIRT_APP + w.VIRT_DISG) / 2,
        "Valence": (w.REAL_DISG + w.VIRT_DISG) / 2 - (w.REAL_APP + w.VIRT_APP) / 2,
        "Interaction": (w.REAL_APP - w.VIRT_APP) - (w.REAL_DISG - w.VIRT_DISG),
    }


def wide(df, value, subject_col="subject"):
    w = df.pivot_table(index=subject_col, columns="condition", values=value)
    w = w[[c for c in CONDS if c in w.columns]].dropna()
    if w.shape[1] < 4:
        return None
    incl = [inclusion_number(s) for s in w.index]
    keep = [i not in EXCLUDED_INCLUSIONS for i in incl]
    return w[keep]


# ==========================================================
# CHARGEMENT DES MESURES
# ==========================================================

measures = {}     # nom -> (DataFrame sujets x conditions, unite, famille)

if os.path.exists(TFR_FILE):
    t = pd.read_csv(TFR_FILE)
    t = t[(t.roi == ROI) & (t.band == BAND) & (t.window == WINDOW)]
    measures["MU_R"] = (wide(t, "power_db"), "dB", "Time-frequency")
else:
    print("Absent :", TFR_FILE)

if os.path.exists(GFP_PEAK_FILE):
    g = pd.read_csv(GFP_PEAK_FILE)
    measures["GFP_PEAK"] = (wide(g, "peak_gfp_uv"), "uV", "Global GFP")
else:
    print("Absent :", GFP_PEAK_FILE)

if os.path.exists(GFP_WINDOWS_FILE):
    gw = pd.read_csv(GFP_WINDOWS_FILE)
    for win in gw.sort_values("window_start_ms").microstate.unique():
        name = f"GFP_{win}"
        measures[name] = (wide(gw[gw.microstate == win], "mean_gfp_fixed_uv"),
                          "uV", "GFP fixed windows")
        PRIMARY += [(name, "Reality"), (name, "Interaction")]
else:
    print("Absent :", GFP_WINDOWS_FILE)

if os.path.exists(MS_PARAMS_FILE):
    ms = pd.read_csv(MS_PARAMS_FILE)
    for state in sorted(ms.microstate.unique()):
        if state in MS_EXCLUDED:
            continue
        for par in MS_PARAMS:
            sub = ms[ms.microstate == state]
            measures[f"{state}_{par}"] = (wide(sub, par), par, "Microstates")
else:
    print("Absent :", MS_PARAMS_FILE)

measures = {k: v for k, v in measures.items() if v[0] is not None and len(v[0]) > 3}
print(f"{len(measures)} mesures chargees :", ", ".join(measures))


# ==========================================================
# CALCULS
# ==========================================================

rows = []
for name, (w, unit, family) in measures.items():
    for effect, x in contrasts(w).items():
        x = x.values.astype(float)
        n = len(x)
        if x.std(ddof=1) == 0:
            print(f"  {name} / {effect} : contraste constant, ignore")
            continue
        tval, p = stats.ttest_1samp(x, 0)
        dz = x.mean() / x.std(ddof=1)
        lo, hi = dz_ci90(tval, n)
        bf10 = jzs_bf10(tval, n, PRIOR_R)
        row = {
            "family": family, "measure": name, "unit": unit, "effect": effect,
            "n": n, "mean": x.mean(), "sd": x.std(ddof=1),
            "t": tval, "p": p, "dz": dz, "dz_ci90_low": lo, "dz_ci90_high": hi,
            "BF10": bf10, "BF01": 1 / bf10,
            f"p_TOST_dz{EQ_BOUND_DZ}": tost_p(x, EQ_BOUND_DZ),
        }
        for r in PRIOR_R_ROBUST:
            row[f"BF01_r{r}"] = 1 / jzs_bf10(tval, n, r)
        rows.append(row)

res = pd.DataFrame(rows)


def verdict(bf01):
    if bf01 >= 10:
        return "H0 forte"
    if bf01 >= 3:
        return "H0 moderee"
    if bf01 > 1 / 3:
        return "non concluant"
    if bf01 > 1 / 10:
        return "H1 moderee"
    return "H1 forte"


res["evidence"] = res.BF01.apply(verdict)
res.to_csv(os.path.join(OUTDIR, "bayes_equivalence.csv"), index=False)


# ==========================================================
# RESUME
# ==========================================================

tost_col = f"p_TOST_dz{EQ_BOUND_DZ}"
cols = ["measure", "effect", "n", "dz", "dz_ci90_low", "dz_ci90_high", "p",
        "BF01", f"BF01_r{PRIOR_R_ROBUST[0]}", f"BF01_r{PRIOR_R_ROBUST[1]}",
        tost_col, "evidence"]
prim = pd.DataFrame(PRIMARY, columns=["measure", "effect"]).merge(res, on=["measure", "effect"])

lines = [
    f"Facteurs de Bayes (JZS, r = {PRIOR_R}) et TOST (bornes dz = +/- {EQ_BOUND_DZ})",
    f"Sujets exclus : {EXCLUDED_INCLUSIONS}",
    "BF01 > 3 : soutien a l'absence d'effet ; p_TOST < .05 : effet plus petit que la borne",
    "",
    "=== Mesures principales ===",
    prim[cols].round(3).to_string(index=False),
    "",
]
for fam in res.family.unique():
    sub = res[(res.family == fam) & (res.effect != "Valence")]
    lines += [f"=== {fam} : Reality et Interaction ===",
              sub[cols].round(3).to_string(index=False), ""]
lines += ["=== Valence (controle de sensibilite : l'effet connu doit donner BF10 eleve) ===",
          res[res.effect == "Valence"][["measure", "n", "dz", "p", "BF10", "evidence"]]
          .round(3).to_string(index=False)]

txt = "\n".join(lines)
pd.set_option("display.width", 200)
print(txt)
with open(os.path.join(OUTDIR, "bayes_summary.txt"), "w", encoding="utf-8") as f:
    f.write(txt)


# ==========================================================
# FIGURE : BF01 (log) pour Reality et Interaction
# ==========================================================

plot = res[res.effect.isin(["Reality", "Interaction"])
           & res.family.isin(["Time-frequency", "Global GFP", "GFP fixed windows"])]
if len(plot):
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.45 * plot.measure.nunique() + 1.6),
                             sharey=True)
    order = list(dict.fromkeys(plot.measure))
    for ax, eff in zip(axes, ["Reality", "Interaction"]):
        d = plot[plot.effect == eff].set_index("measure").reindex(order)
        y = np.arange(len(order))
        ax.scatter(d.BF01, y, color="black", zorder=3)
        for r in PRIOR_R_ROBUST:
            ax.scatter(d[f"BF01_r{r}"], y, marker="|", color="grey", s=80, zorder=2)
        for v, ls in [(1, "-"), (3, "--"), (1 / 3, "--"), (10, ":"), (0.1, ":")]:
            ax.axvline(v, color="grey", lw=.8, ls=ls)
        ax.set_xscale("log")
        ax.set_xlim(0.01, 100)
        ax.set_yticks(y)
        ax.set_yticklabels(order, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("BF01 (log)  ←  effet  |  absence d'effet  →", fontsize=8)
        ax.set_title(eff)
    fig.suptitle(f"Facteurs de Bayes JZS (r = {PRIOR_R} ; traits gris : r = "
                 f"{PRIOR_R_ROBUST[0]} et {PRIOR_R_ROBUST[1]})", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "bayes_forest.png"), dpi=300)
    plt.close(fig)

print(f"\nSorties dans {OUTDIR}")
