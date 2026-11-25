# -*- coding: utf-8 -*-

"""
TANOVA (analyse topographique) - Lab Is Coat
============================================

Question : la CONFIGURATION des cartes de potentiel (independamment de
leur force) differe-t-elle entre conditions, et a quels instants ?

Complement des autres analyses evoquees :
    - GFP (scripts 03-04, 07)  -> force de la reponse
    - TANOVA (ce script)       -> configuration du champ sur le scalp
    - micro-etats (01-05)      -> sequence des configurations stables
(Murray, Brunet & Michel, 2008 ; Koenig & Melie-Garcia, 2010)

Methode
-------
1) ERP individuels par condition (memes regles que 01-05 : config.py,
   canaux communs, reference moyenne, fenetre config.TMIN-TMAX).
2) Plan 2 x 2 a mesures repetees -> chaque effet oppose deux "cotes",
   chacun moyenne de deux conditions :
       Reality     : (RA + RD)/2  contre (VA + VD)/2
       Valence     : (RD + VD)/2  contre (RA + VA)/2
       Interaction : (RD + VA)/2  contre (RA + VD)/2
                     [= (RD - RA) - (VD - VA)]
3) Statistique a chaque instant = dissimilarite globale (Lehmann &
   Skrandies, 1980) entre les cartes MOYENNES des deux cotes, apres
   normalisation de chacune par sa GFP (procedure de Murray et al., 2008).
   La normalisation est faite sur les moyennes de groupe, et non carte
   par carte chez chaque sujet : normaliser des cartes individuelles
   bruitees transforme une simple difference d'amplitude en fausse
   difference de forme (verifie sur donnees simulees). Une difference
   purement d'amplitude (meme carte, plus forte) ne produit donc pas
   d'effet TANOVA : elle est testee par la GFP.
4) Permutations : echange aleatoire des deux cotes chez chaque sujet
   (valide pour un effet a 1 degre de liberte). p par instant =
   proportion des permutations >= observe.
5) Correction pour les comparaisons multiples dans le temps : statistique
   de duree globale (Koenig & Melie-Garcia, 2010). Pour chaque
   permutation, on calcule la plus longue suite d'instants consecutifs
   a p < ALPHA ; le 95e centile de cette distribution donne la duree
   minimale qu'un effet doit atteindre pour etre significatif.
6) Test de coherence topographique (TCT ; Koenig & Melie-Garcia, 2010) :
   a chaque instant, la carte moyenne d'une condition est-elle plus
   coherente entre sujets que si les electrodes etaient melangees ?
   Une TANOVA n'est interpretable que pendant les periodes ou les cartes
   sont coherentes.

Entrees : *-epo.fif (config.EPOCHS_DIR)
Sorties (dans config.OUTPUT_DIR/tanova/) :
    tanova_timecourse.csv   instant x effet : dissimilarite, p
    tanova_windows.csv      periodes significatives (apres correction de duree)
    tct_timecourse.csv      p du TCT par condition et par instant
    tanova_summary.txt      resume pour les Methodes / Resultats
    figures : tanova_p_timecourse.png, tanova_difference_maps.png
"""

import os
import glob

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

import config


# ==========================================================
# PARAMETRES
# ==========================================================

N_PERMUTATIONS = 5000
ALPHA = 0.05              # seuil par instant
DURATION_QUANTILE = 0.95  # correction de duree globale
N_PERM_TCT = 1000
RANDOM_STATE = 42
BATCH = 250               # permutations traitees par lot (memoire)

# Sujets exclus (meme regle que les autres analyses). Si config.py
# definit EXCLUDED_SUBJECTS, il est utilise ; sinon liste ci-dessous.
EXCLUDED_SUBJECTS = getattr(config, "EXCLUDED_SUBJECTS", [])

OUTDIR = os.path.join(config.OUTPUT_DIR, "tanova")
os.makedirs(OUTDIR, exist_ok=True)

COND_ORDER = ["REAL_APP", "VIRT_APP", "REAL_DISG", "VIRT_DISG"]


# ==========================================================
# OUTILS
# ==========================================================

def gfp(x, axis):
    """GFP = ecart-type spatial (donnees en reference moyenne)."""
    return np.std(x, axis=axis)


def normalize_maps(data):
    """data : (..., n_channels, n_times). Centre et divise chaque carte
    (chaque instant) par sa GFP."""
    data = data - data.mean(axis=-2, keepdims=True)
    g = np.std(data, axis=-2, keepdims=True)
    g[g == 0] = 1.0
    return data / g


def longest_run(mask):
    """Longueur de la plus longue suite de True consecutifs."""
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def runs(mask):
    """Liste des (debut, fin_inclus) des suites de True."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def dissimilarity(a, b):
    """Dissimilarite globale entre cartes normalisees. a, b : (..., n_ch, n_t)."""
    return gfp(normalize_maps(a) - normalize_maps(b), axis=-2)


def tanova_1df(side_a, side_b, rng, n_perm):
    """side_a, side_b : (n_subj, n_ch, n_t), cartes NON normalisees des
    deux cotes de l'effet. Renvoie la dissimilarite observee (n_t), les p
    par instant (n_t), et la distribution des durees maximales sous H0."""
    n_subj, n_ch, n_t = side_a.shape
    M = ((side_a + side_b) / 2).mean(axis=0)                  # moyenne commune
    D = (side_a - side_b).reshape(n_subj, -1)                 # differences

    obs = dissimilarity(side_a.mean(axis=0), side_b.mean(axis=0))

    # sous H0 : echange des cotes chez chaque sujet = signe du demi-ecart
    null = np.empty((n_perm, n_t))
    done = 0
    while done < n_perm:
        b = min(BATCH, n_perm - done)
        signs = rng.choice([-1.0, 1.0], size=(b, n_subj))
        half = (signs @ D / n_subj / 2).reshape(b, n_ch, n_t)
        null[done:done + b] = dissimilarity(M + half, M - half)
        done += b

    # p observe par instant (l'observe compte dans la distribution)
    p_obs = (1 + (null >= obs).sum(axis=0)) / (n_perm + 1)

    # p de chaque permutation par instant -> durees maximales sous H0
    # rang decroissant parmi les n_perm permutations + l'observe
    all_stats = np.vstack([obs[None, :], null])               # (n_perm+1, n_t)
    order = np.argsort(-all_stats, axis=0, kind="stable")
    ranks = np.empty_like(order)
    rows = np.arange(all_stats.shape[0])[:, None]
    ranks[order, np.arange(n_t)[None, :]] = rows + 1          # 1 = plus grand
    p_all = ranks / (n_perm + 1)
    max_dur = np.array([longest_run(p_all[k] < ALPHA) for k in range(1, n_perm + 1)])

    return obs, p_obs, max_dur


def tct(maps, rng, n_perm):
    """Test de coherence topographique pour une condition.
    maps : (n_subj, n_ch, n_t), cartes normalisees. Sous H0, on melange
    les electrodes de chaque sujet independamment."""
    n_subj, n_ch, n_t = maps.shape
    obs = gfp(maps.mean(axis=0), axis=0)
    count = np.zeros(n_t)
    for _ in range(n_perm):
        shuffled = np.empty_like(maps)
        for s in range(n_subj):
            shuffled[s] = maps[s][rng.permutation(n_ch)]
        count += gfp(shuffled.mean(axis=0), axis=0) >= obs
    return (1 + count) / (n_perm + 1)


# ==========================================================
# CHARGEMENT DES ERP INDIVIDUELS
# ==========================================================

files = sorted(glob.glob(os.path.join(config.EPOCHS_DIR, "*-epo.fif")))
files = [f for f in files
         if os.path.basename(f).replace("-epo.fif", "") not in EXCLUDED_SUBJECTS]
if config.DEBUG_MAX_SUBJECTS is not None:
    files = files[: config.DEBUG_MAX_SUBJECTS]
if not files:
    raise FileNotFoundError(f"Aucun fichier *-epo.fif dans {config.EPOCHS_DIR}")

common_channels, target_sfreq = config.get_dataset_specs(files)

data = {c: [] for c in COND_ORDER}
subjects = []
info = None
times = None

for file in files:
    subject = os.path.basename(file).replace("-epo.fif", "")
    epochs = config.prepare_epochs(file, common_channels, target_sfreq)

    evs = {}
    for cond in COND_ORDER:
        vis = config.CONDITIONS[cond]
        try:
            ep = mne.concatenate_epochs([epochs[vis[0]], epochs[vis[1]]], verbose=False)
        except KeyError:
            break
        ev = ep.average()
        ev.crop(config.TMIN, config.TMAX)
        ev.set_eeg_reference("average", projection=False, verbose=False)
        evs[cond] = ev

    if len(evs) < len(COND_ORDER):
        print(f"  {subject} : condition manquante, sujet ignore")
        continue

    for cond in COND_ORDER:
        data[cond].append(evs[cond].data)
    subjects.append(subject)
    if info is None:
        info = evs[COND_ORDER[0]].info
        times = evs[COND_ORDER[0]].times * 1000

n_subj = len(subjects)
print(f"\n{n_subj} sujets inclus dans la TANOVA")
if EXCLUDED_SUBJECTS:
    print("Exclus :", EXCLUDED_SUBJECTS)

raw_maps = {c: np.array(v) for c, v in data.items()}           # (n_subj, n_ch, n_t)
norm = {c: normalize_maps(v) for c, v in raw_maps.items()}     # pour le TCT

RA, VA, RD, VD = (raw_maps[c] for c in COND_ORDER)
CONTRASTS = {
    "Reality (REAL - VIRTUAL)": ((RA + RD) / 2, (VA + VD) / 2),
    "Valence (DISG - APP)": ((RD + VD) / 2, (RA + VA) / 2),
    "Interaction": ((RD + VA) / 2, (RA + VD) / 2),
}

dt = times[1] - times[0]
rng = np.random.default_rng(RANDOM_STATE)


# ==========================================================
# TEST DE COHERENCE TOPOGRAPHIQUE
# ==========================================================

print("\nTCT (coherence des cartes entre sujets)...")
tct_rows = []
tct_p = {}
for cond in COND_ORDER:
    p = tct(norm[cond], rng, N_PERM_TCT)
    tct_p[cond] = p
    for t, pv in zip(times, p):
        tct_rows.append({"condition": cond, "time_ms": t, "p_tct": pv})
    incoh = runs(p >= ALPHA)
    txt = ", ".join(f"{times[a]:.0f}-{times[b]:.0f}" for a, b in incoh) or "aucune"
    print(f"  {cond} : periodes NON coherentes (p >= {ALPHA}) : {txt}")
pd.DataFrame(tct_rows).to_csv(os.path.join(OUTDIR, "tct_timecourse.csv"), index=False)


# ==========================================================
# TANOVA
# ==========================================================

rows, windows, results = [], [], {}

for name, (side_a, side_b) in CONTRASTS.items():

    print(f"\nTANOVA : {name} ({N_PERMUTATIONS} permutations)...")
    obs, p, max_dur = tanova_1df(side_a, side_b, rng, N_PERMUTATIONS)

    dur_threshold = np.quantile(max_dur, DURATION_QUANTILE)   # en echantillons
    sig_runs = [(a, b) for a, b in runs(p < ALPHA) if (b - a + 1) > dur_threshold]
    results[name] = dict(obs=obs, p=p, dur_threshold=dur_threshold, sig_runs=sig_runs)

    print(f"  duree minimale significative : {dur_threshold * dt:.0f} ms "
          f"({dur_threshold:.0f} echantillons)")

    for t, d, pv in zip(times, obs, p):
        rows.append({"effect": name, "time_ms": t, "dissimilarity": d, "p": pv})

    if not sig_runs:
        print("  aucune periode significative apres correction de duree")
    for a, b in sig_runs:
        windows.append({
            "effect": name,
            "start_ms": round(times[a], 1),
            "end_ms": round(times[b], 1),
            "duration_ms": round((b - a + 1) * dt, 1),
            "min_p": p[a:b + 1].min(),
            "mean_dissimilarity": obs[a:b + 1].mean(),
            "duration_threshold_ms": round(dur_threshold * dt, 1),
        })
        print(f"  periode significative : {times[a]:.0f}-{times[b]:.0f} ms "
              f"(min p = {p[a:b + 1].min():.4f})")

pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "tanova_timecourse.csv"), index=False)
windows_df = pd.DataFrame(
    windows,
    columns=["effect", "start_ms", "end_ms", "duration_ms", "min_p",
             "mean_dissimilarity", "duration_threshold_ms"],
)
windows_df.to_csv(os.path.join(OUTDIR, "tanova_windows.csv"), index=False)


# ==========================================================
# FIGURE 1 : p par instant + periodes significatives
# ==========================================================

fig, axes = plt.subplots(len(CONTRASTS), 1, figsize=(9, 2.4 * len(CONTRASTS)), sharex=True)
for ax, (name, r) in zip(axes, results.items()):
    ax.plot(times, 1 - r["p"], color="black", lw=1)
    ax.axhline(1 - ALPHA, color="grey", ls="--", lw=0.8)
    for a, b in runs(r["p"] < ALPHA):
        ax.axvspan(times[a], times[b], color="lightgrey", alpha=0.6, lw=0)
    for a, b in r["sig_runs"]:
        ax.axvspan(times[a], times[b], color="gold", alpha=0.6, lw=0)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("1 - p")
    ax.set_title(f"{name}   (duree min. = {r['dur_threshold'] * dt:.0f} ms)", fontsize=10)
axes[-1].set_xlabel("Temps (ms)")
fig.suptitle(f"TANOVA, N = {n_subj}  |  gris : p < {ALPHA} par instant ; "
             f"dore : significatif apres correction de duree", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "tanova_p_timecourse.png"), dpi=300)
plt.close(fig)


# ==========================================================
# FIGURE 2 : cartes de difference (non normalisees, en uV)
# moyennees sur chaque periode significative ; a defaut, sur les
# fenetres de la grande moyenne (grand_average_segments.csv)
# ==========================================================

RAW = {c: raw_maps[c].mean(axis=0) for c in COND_ORDER}       # (n_ch, n_t)
RAW_DIFF = {
    "Reality (REAL - VIRTUAL)": (RAW["REAL_APP"] + RAW["REAL_DISG"]) / 2
                                - (RAW["VIRT_APP"] + RAW["VIRT_DISG"]) / 2,
    "Valence (DISG - APP)": (RAW["REAL_DISG"] + RAW["VIRT_DISG"]) / 2
                            - (RAW["REAL_APP"] + RAW["VIRT_APP"]) / 2,
    "Interaction": (RAW["REAL_DISG"] - RAW["REAL_APP"])
                   - (RAW["VIRT_DISG"] - RAW["VIRT_APP"]),
}

try:
    ga_windows = config.load_ga_windows()
except FileNotFoundError:
    ga_windows = {}

panels = []
for name in CONTRASTS:
    wins = results[name]["sig_runs"]
    if wins:
        for a, b in wins:
            panels.append((name, f"{times[a]:.0f}-{times[b]:.0f} ms *", a, b))
    else:
        for wname, (t0, t1) in ga_windows.items():
            a = int(np.argmin(np.abs(times - t0)))
            b = int(np.argmin(np.abs(times - t1)))
            panels.append((name, f"{wname} (n.s.)", a, b))

if panels:
    n_cols = max(sum(1 for p in panels if p[0] == n) for n in CONTRASTS)
    fig, axes = plt.subplots(len(CONTRASTS), n_cols, figsize=(2.6 * n_cols, 2.8 * len(CONTRASTS)),
                             squeeze=False)
    vmax = max(np.abs(RAW_DIFF[n][:, a:b + 1].mean(axis=1)).max() for n, _, a, b in panels) * 1e6
    for i, name in enumerate(CONTRASTS):
        mine = [p for p in panels if p[0] == name]
        for j in range(n_cols):
            ax = axes[i, j]
            if j >= len(mine):
                ax.axis("off")
                continue
            _, label, a, b = mine[j]
            topo = RAW_DIFF[name][:, a:b + 1].mean(axis=1) * 1e6
            mne.viz.plot_topomap(topo, info, axes=ax, show=False, cmap="RdBu_r",
                                 vlim=(-vmax, vmax))
            ax.set_title(label, fontsize=8)
        axes[i, 0].set_ylabel(name, fontsize=9)
    fig.suptitle(f"Cartes de difference (uV, +/- {vmax:.2f}) ; * = periode TANOVA significative",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "tanova_difference_maps.png"), dpi=300)
    plt.close(fig)


# ==========================================================
# RESUME
# ==========================================================

lines = [
    f"TANOVA - N = {n_subj} participants, {times[0]:.0f}-{times[-1]:.0f} ms, "
    f"{len(common_channels)} canaux, {target_sfreq:.0f} Hz",
    f"{N_PERMUTATIONS} permutations (echange des conditions par sujet), "
    f"seuil par instant p < {ALPHA}, correction de duree globale au {DURATION_QUANTILE:.0%}",
    "",
]
for name, r in results.items():
    lines.append(f"{name} : duree minimale {r['dur_threshold'] * dt:.0f} ms")
    if r["sig_runs"]:
        for a, b in r["sig_runs"]:
            lines.append(f"   significatif {times[a]:.0f}-{times[b]:.0f} ms "
                         f"(min p = {r['p'][a:b + 1].min():.4f})")
    else:
        lines.append("   aucune periode significative")
lines.append("")
lines.append("TCT : periodes ou les cartes ne sont PAS coherentes entre sujets")
for cond, p in tct_p.items():
    txt = ", ".join(f"{times[a]:.0f}-{times[b]:.0f}" for a, b in runs(p >= ALPHA)) or "aucune"
    lines.append(f"   {cond} : {txt}")
lines.append("")
lines.append("Rappel : cartes moyennes normalisees par la GFP -> une difference d'amplitude "
             "seule (meme topographie) ne donne pas d'effet TANOVA ; elle est testee par la GFP.")

summary = "\n".join(lines)
with open(os.path.join(OUTDIR, "tanova_summary.txt"), "w", encoding="utf-8") as f:
    f.write(summary)
print("\n" + summary)
print("\nSorties dans :", OUTDIR)
