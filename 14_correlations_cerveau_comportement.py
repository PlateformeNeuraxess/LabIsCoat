# -*- coding: utf-8 -*-

"""
14 - Correlations cerveau-comportement - Lab Is Coat
====================================================

Question : les signatures neurales du reel et de la valence sont-elles
dissociables AU NIVEAU INDIVIDUEL ? (titre : "Dissociable neural
signatures of emotional value and physical reality in food perception")

Plan de double dissociation (defini AVANT de voir les resultats)
---------------------------------------------------------------

                         effet compt. du REEL     effet compt. de VALENCE
                         (EVA_MOY)                (ENVIE_VAL)
    MU_R  (EEG reel)     attendu  (diagonale)     nul attendu (croise)
    GFP_VAL (EEG val.)   nul attendu (croise)     attendu  (diagonale)

- Diagonale (2 correlations predites) : Spearman, correction de Holm sur
  ces 2 tests.
- Cases croisees (2 correlations predites nulles) : on ne peut pas
  "accepter" H0 avec une p > .05 -> on rapporte BF01 = 1 / BF10
  (BF01 > 3 = soutien a l'absence de correlation).
- Test formel de la dissociation : pour chaque indice EEG, la
  correlation avec "son" effet comportemental est-elle plus forte qu'avec
  l'autre ? Test de Steiger (1980) pour deux correlations dependantes
  partageant une variable, sur les rho de Spearman (approximation
  courante). 2 tests, correction de Holm.

Indices EEG
    MU_R     : effet du reel sur le mu 8-13 Hz, central droit, fenetre
               principale : (REAL_APP + REAL_DISG)/2 - (VIRT_APP + VIRT_DISG)/2
               (dB ; script 12 -> outputs_tfr/tfr_roi_band_windows.csv)
    GFP_VAL  : effet de valence sur le GFP moyen dans la derniere fenetre
               de la grande moyenne (MS5 tardive, ~478-600 ms) :
               (REAL_DISG + VIRT_DISG)/2 - (REAL_APP + VIRT_APP)/2
               (uV ; script 07 -> microstate_parameters_fixed_window.csv).
               Fenetre definie sur la grande moyenne toutes conditions
               -> orthogonale au contraste, pas de double dipping.
    MU_R_APP, MU_R_DISG, GFP_REAL (effet du reel sur ce meme GFP) :
               exploratoires.

Indices comportementaux (un par sujet)
    EVA_MOY    : echelle d'intensite post-session, moyenne APP/DISG
                 (0-10 ; 5 = pas de difference, > 5 = reel plus intense)
                 = effet comportemental du REEL
    ENVIE_VAL  : envie de manger, (RA + VA)/2 - (RD + VD)/2
                 = effet comportemental de la VALENCE
    ENVIE_APP, ENVIE_DISG, ENVIE_INTER, EVA_APP, EVA_DISG : exploratoires

Moderateurs (exploratoire) : FAIM, DUREE (questionnaire alimentaire).
FCQ-T-r : description de l'echantillon uniquement, non utilise ici.

Statistiques
    Spearman + IC 95 % bootstrap + BF10 (approximation de Jeffreys,
    a priori uniforme sur rho ; Ly, Verhagen & Wagenmakers, 2016),
    calcule sur les rangs. Les 4 correlations du plan sont calculees sur
    les MEMES sujets (donnees completes pour les 4 indices), condition
    du test de Steiger.

Correspondance des sujets
    Ligne k des fichiers comportementaux (k = 1..30) = sujet d'inclusion k.
    Numero d'inclusion = derniers chiffres du code EEG (VL29 -> 29).
    Verifier l'affichage de controle.

Sorties (OUTDIR)
    correlations.csv                 toutes les correlations
    dissociation_steiger.csv         tests de Steiger
    dissociation_matrix.png          figure 2 x 2 (double dissociation)
    correlations_data.csv            indices par sujet (controle)
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
GFP_FILE = "microstate_parameters_fixed_window.csv"   # sortie du script 07
GFP_METRIC = "mean_gfp_fixed_uv"
GFP_WINDOW = None   # None = derniere fenetre (debut le plus tardif) ;
                    # sinon le nom exact, ex. "MS5_478_600"

ENVIE_FILE = "BDDseparee_phasereco.xlsx"      # colonnes Moy_RA, Moy_VA, Moy_RD, Moy_VD
EVA_FILE = "BDDseparee_intensitePOST.xlsx"   # colonnes EVA_APP, EVA_DISG
QA_FILE = "BDDseparee_QuestionnaireAlimentaire.xlsx"  # Pre_Al_1_faim, Pre_Al_2_durée

# Sujets exclus de toutes les analyses EEG (sujet 12 : exclu du temps-frequence)
EXCLUDED_INCLUSIONS = [12]

OUTDIR = "outputs_correlations"
os.makedirs(OUTDIR, exist_ok=True)

ROI = "Central droit"
BAND = "mu"
WINDOW = "main"

# Plan de double dissociation : (indice EEG, indice comportemental, prediction)
DIAGONAL = [("MU_R", "EVA_MOY"), ("GFP_VAL", "ENVIE_VAL")]      # correlation attendue
CROSSED = [("MU_R", "ENVIE_VAL"), ("GFP_VAL", "EVA_MOY")]       # absence attendue
# Steiger : (indice EEG, comportement "propre", comportement "autre")
STEIGER = [("MU_R", "EVA_MOY", "ENVIE_VAL"), ("GFP_VAL", "ENVIE_VAL", "EVA_MOY")]

N_BOOT = 5000
SEED = 42


# ==========================================================
# OUTILS
# ==========================================================

def inclusion_number(code):
    """Numero d'inclusion = les 2 chiffres qui suivent les initiales
    (VL29 -> 29, BE09 -> 9), meme si le code porte un suffixe
    (ex. 'VL29-ICA-ICA', 'BE09_smooth'). A defaut : derniers chiffres."""
    s = str(code)
    m = re.search(r"[A-Za-z]{2}(\d{2})", s) or re.search(r"(\d+)\s*$", s)
    return int(m.group(1)) if m else np.nan


def bf10_correlation(r, n):
    """BF10 pour une correlation (Jeffreys, a priori uniforme sur rho)."""
    def lik(rho):
        return (1 - rho ** 2) ** ((n - 1) / 2) / (1 - rho * r) ** ((2 * n - 3) / 2)
    val, _ = integrate.quad(lik, -1, 1, limit=200)
    return 0.5 * val


def boot_ci(x, y, rng, n_boot):
    n = len(x)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        vals[b] = stats.spearmanr(x[i], y[i]).statistic
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    return np.percentile(vals, [2.5, 97.5])


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def steiger_z(r_jk, r_jh, r_kh, n):
    """Steiger (1980), Z2* : compare r_jk et r_jh (variable j commune),
    r_kh = correlation entre les deux autres variables. Bilateral."""
    z_jk, z_jh = np.arctanh(r_jk), np.arctanh(r_jh)
    rbar = (r_jk + r_jh) / 2
    psi = r_kh * (1 - 2 * rbar ** 2) - 0.5 * rbar ** 2 * (1 - 2 * rbar ** 2 - r_kh ** 2)
    c = psi / (1 - rbar ** 2) ** 2
    z = (z_jk - z_jh) * np.sqrt(n - 3) / np.sqrt(2 - 2 * c)
    return z, 2 * stats.norm.sf(abs(z))


def spearman_row(x, y, rng, label_e, label_b, status):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        print(f"  {label_e} x {label_b} : variable constante ou trop peu de sujets, ignore")
        return None
    r = stats.spearmanr(x, y)
    lo, hi = boot_ci(x, y, rng, N_BOOT)
    bf10 = bf10_correlation(r.statistic, len(x))
    return {"eeg": label_e, "behavior": label_b, "n": len(x),
            "rho": r.statistic, "p": r.pvalue, "ci_low": lo, "ci_high": hi,
            "BF10": bf10, "BF01": 1 / bf10, "status": status}


# ==========================================================
# INDICES EEG : mu (temps-frequence)
# ==========================================================

tfr = pd.read_csv(TFR_FILE)
d = tfr[(tfr.roi == ROI) & (tfr.band == BAND) & (tfr.window == WINDOW)]
w = d.pivot_table(index="subject", columns="condition", values="power_db")

mu = pd.DataFrame(index=w.index)
mu["MU_R"] = (w["REAL_APP"] + w["REAL_DISG"]) / 2 - (w["VIRT_APP"] + w["VIRT_DISG"]) / 2
mu["MU_R_APP"] = w["REAL_APP"] - w["VIRT_APP"]
mu["MU_R_DISG"] = w["REAL_DISG"] - w["VIRT_DISG"]
mu["inclusion"] = [inclusion_number(s) for s in mu.index]
mu = mu.reset_index().rename(columns={"subject": "eeg_code"})


# ==========================================================
# INDICES EEG : GFP en fenetre fixe (script 07)
# ==========================================================

gfp = pd.read_csv(GFP_FILE)
if GFP_WINDOW is None:
    last = gfp.sort_values("window_start_ms").microstate.iloc[-1]
else:
    last = GFP_WINDOW
g = gfp[gfp.microstate == last]
print(f"Fenetre GFP utilisee : {last} "
      f"({g.window_start_ms.iloc[0]:.0f}-{g.window_end_ms.iloc[0]:.0f} ms), {GFP_METRIC}")
gw = g.pivot_table(index="subject", columns="condition", values=GFP_METRIC)

gf = pd.DataFrame(index=gw.index)
gf["GFP_VAL"] = (gw["REAL_DISG"] + gw["VIRT_DISG"]) / 2 - (gw["REAL_APP"] + gw["VIRT_APP"]) / 2
gf["GFP_REAL"] = (gw["REAL_APP"] + gw["REAL_DISG"]) / 2 - (gw["VIRT_APP"] + gw["VIRT_DISG"]) / 2
gf["inclusion"] = [inclusion_number(s) for s in gf.index]
gf = gf.reset_index().rename(columns={"subject": "eeg_code_gfp"})

for df_, name_ in [(mu, "TFR"), (gf, "GFP")]:
    bad = df_.loc[df_.inclusion.isna()]
    if len(bad):
        print(f"ATTENTION ({name_}) : numero d'inclusion illisible pour :",
              list(bad.get("eeg_code", bad.index)))
mu = mu.dropna(subset=["inclusion"]).astype({"inclusion": int})
gf = gf.dropna(subset=["inclusion"]).astype({"inclusion": int})
eeg = mu.merge(gf, on="inclusion", how="outer")


# ==========================================================
# INDICES COMPORTEMENTAUX
# ==========================================================

env = pd.read_excel(ENVIE_FILE)
env.columns = [c.replace("Moy_", "") for c in env.columns]
beh = pd.DataFrame({"inclusion": np.arange(1, len(env) + 1)})
beh["ENVIE_VAL"] = (env["RA"] + env["VA"]) / 2 - (env["RD"] + env["VD"]) / 2
beh["ENVIE_APP"] = env["RA"] - env["VA"]
beh["ENVIE_DISG"] = env["RD"] - env["VD"]
beh["ENVIE_INTER"] = beh["ENVIE_APP"] - beh["ENVIE_DISG"]

eva = pd.read_excel(EVA_FILE)
beh["EVA_APP"] = eva["EVA_APP"].values
beh["EVA_DISG"] = eva["EVA_DISG"].values
beh["EVA_MOY"] = (beh["EVA_APP"] + beh["EVA_DISG"]) / 2

if QA_FILE:
    qa = pd.read_excel(QA_FILE)
    beh["FAIM"] = qa.iloc[:, 0].values      # Pre_Al_1_faim
    beh["DUREE"] = qa.iloc[:, 1].values     # Pre_Al_2_durée

data = eeg.merge(beh, on="inclusion", how="inner").sort_values("inclusion")
data = data[~data.inclusion.isin(EXCLUDED_INCLUSIONS)]

print("Correspondance code EEG -> numero d'inclusion (a verifier) :")
print("  " + ", ".join(f"{c}->{i}" for c, i in zip(data.eeg_code, data.inclusion)))
if eeg.inclusion.isna().any():
    print("  ATTENTION : codes EEG sans numero d'inclusion lisible")

# echantillon commun pour le plan (memes sujets pour les 4 correlations)
plan_vars = ["MU_R", "GFP_VAL", "EVA_MOY", "ENVIE_VAL"]
core = data.dropna(subset=plan_vars)
n = len(core)
print(f"\n{n} sujets avec les 4 indices du plan "
      f"(manquants : {sorted(set(data.inclusion) - set(core.inclusion))})")
data.to_csv(os.path.join(OUTDIR, "correlations_data.csv"), index=False)


# ==========================================================
# 1. PLAN DE DOUBLE DISSOCIATION
# ==========================================================

rng = np.random.default_rng(SEED)
rows = []
for e, b in DIAGONAL:
    rows.append(spearman_row(core[e].values, core[b].values, rng, e, b, "planned: diagonal"))
for e, b in CROSSED:
    rows.append(spearman_row(core[e].values, core[b].values, rng, e, b, "planned: crossed"))

# ==========================================================
# 2. EXPLORATOIRE (non corrige)
# ==========================================================

done = set(DIAGONAL + CROSSED)
eeg_x = ["MU_R", "MU_R_APP", "MU_R_DISG", "GFP_VAL", "GFP_REAL"]
beh_x = ["EVA_MOY", "EVA_APP", "EVA_DISG", "ENVIE_VAL", "ENVIE_APP", "ENVIE_DISG", "ENVIE_INTER"]
for e in eeg_x:
    for b in beh_x:
        if (e, b) in done or e not in data or b not in data:
            continue
        rows.append(spearman_row(data[e].values, data[b].values, rng, e, b, "exploratory"))
for m in [c for c in ["FAIM", "DUREE"] if c in data]:
    for t in ["EVA_MOY", "ENVIE_APP", "MU_R"]:
        rows.append(spearman_row(data[m].values, data[t].values, rng, m, t,
                                 "exploratory (moderator)"))

res = pd.DataFrame([r for r in rows if r is not None])
res["p_holm"] = np.nan
diag = res.status == "planned: diagonal"
res.loc[diag, "p_holm"] = holm(res.loc[diag, "p"])
res["evidence"] = pd.cut(res["BF10"], [0, 1 / 3, 3, np.inf],
                         labels=["pour H0 (BF01>3)", "non concluant", "pour H1 (BF10>3)"])
res.to_csv(os.path.join(OUTDIR, "correlations.csv"), index=False)


# ==========================================================
# 3. TESTS DE STEIGER (la dissociation elle-meme)
# ==========================================================

def rho(a, b):
    return stats.spearmanr(core[a], core[b]).statistic

st_rows = []
for e, own, other in STEIGER:
    r_own, r_other, r_beh = rho(e, own), rho(e, other), rho(own, other)
    z, p = steiger_z(r_own, r_other, r_beh, n)
    st_rows.append({"eeg": e, "own_behavior": own, "other_behavior": other,
                    "rho_own": r_own, "rho_other": r_other, "rho_behaviors": r_beh,
                    "n": n, "z": z, "p": p})
st = pd.DataFrame(st_rows)
st["p_holm"] = holm(st.p)
st.to_csv(os.path.join(OUTDIR, "dissociation_steiger.csv"), index=False)


# ==========================================================
# AFFICHAGE
# ==========================================================

pd.set_option("display.width", 180)
cols = ["eeg", "behavior", "n", "rho", "ci_low", "ci_high", "p", "p_holm", "BF10", "BF01", "evidence"]
print("\n=== Double dissociation : diagonale (correlations attendues, Holm) ===")
print(res[diag][cols].round(3).to_string(index=False))
print("\n=== Double dissociation : cases croisees (absence attendue -> lire BF01) ===")
print(res[res.status == "planned: crossed"][cols].round(3).to_string(index=False))
print("\n=== Tests de Steiger (rho_own > rho_other ?) ===")
print(st.round(3).to_string(index=False))
print("\n=== Exploratoire (non corrige) ===")
print(res[res.status.str.startswith("exploratory")][cols].round(3).to_string(index=False))

print("\nLecture :")
print("  dissociation soutenue si : diagonale significative (p Holm < .05),")
print("  cases croisees BF01 > 3, et Steiger significatif pour chaque indice EEG.")
print("  Avec N ~ 28, seule une correlation d'environ 0,5 est detectee avec 80 % de puissance.")


# ==========================================================
# FIGURE 2 x 2
# ==========================================================

eeg_rows = ["MU_R", "GFP_VAL"]
beh_cols = ["EVA_MOY", "ENVIE_VAL"]
ylab = {"MU_R": "μ power, real − picture (dB)",
        "GFP_VAL": "Late GFP (494–600 ms),\ndisgusting − appetizing (µV)"}
xlab = {"EVA_MOY": "Intensity rating (5 = no difference; > 5 = real)",
        "ENVIE_VAL": "Desire to eat, appetizing − disgusting"}

fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.5))
for i, e in enumerate(eeg_rows):
    for j, b in enumerate(beh_cols):
        ax = axes[i, j]
        r = res[(res.eeg == e) & (res.behavior == b)].iloc[0]
        is_diag = (e, b) in DIAGONAL
        ax.scatter(core[b], core[e], s=26, color="steelblue" if is_diag else "grey")
        if r.p < .05:
            k, c0 = np.polyfit(core[b], core[e], 1)
            xs = np.linspace(core[b].min(), core[b].max(), 50)
            ax.plot(xs, k * xs + c0, color="steelblue", lw=1)
        ax.axhline(0, color="k", lw=.5)
        if b == "EVA_MOY":
            ax.axvline(5, color="k", lw=.5, ls=":")
        ptxt = f"p Holm = {r.p_holm:.3f}" if is_diag else f"p = {r.p:.3f}"
        bftxt = f"BF01 = {r.BF01:.1f}"          # comme dans le texte (3.5)
        ax.text(-0.14, 1.12, "ABCD"[2 * i + j], transform=ax.transAxes, fontsize=12,
                fontweight="bold")
        ax.set_title(f"{'predicted' if is_diag else 'predicted null'}: "
                     f"ρ = {r.rho:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]\n{ptxt}, {bftxt}",
                     fontsize=8.5)
        if i == 1:
            ax.set_xlabel(xlab[b], fontsize=8.5)
        if j == 0:
            ax.set_ylabel(ylab[e], fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "dissociation_matrix.png"), dpi=300)
plt.close(fig)

print(f"\nSorties dans {OUTDIR}")
