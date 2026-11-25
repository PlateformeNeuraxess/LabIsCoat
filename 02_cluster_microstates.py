# -*- coding: utf-8 -*-

"""
Clustering des micro-etats (k-means, polarite prise en compte)

- k-means sur les cartes aux pics de GFP (01), en tenant compte de la
  polarite (config.IGNORE_POLARITY = False) : convention ERP, une
  carte et son inverse de signe sont deux etats differents
  (Murray, Brunet & Michel, 2008). pycrostates n'offre pas cette
  option (invariance imposee) -> algorithme dans microstate_utils.py.

- Choix de K : critere de Krzanowski-Lai (KL maximal ; Murray et al.,
  2008), calcule sur la variance residuelle ponderee par le GFP
  (1 - GEV), reference pour les micro-etats d'ERP (config.K_CRITERION =
  "KL"). Silhouette, Calinski-Harabasz et Davies-Bouldin restent
  calcules a titre indicatif (ils favorisent presque toujours K = 2,
  surtout quand la polarite est prise en compte). Le GEV est donne a
  titre indicatif (il augmente mecaniquement avec K). Le choix
  automatique peut etre remplace par config.K_FORCED apres inspection.

- Les templates sont renumerotes par latence mediane des pics qui leur
  sont assignes (MS1 = le plus precoce) -> numerotation stable.

Entree :
    outputs_microstates/gfp_peak_maps.npy
    outputs_microstates/gfp_peak_metadata.csv
    outputs_microstates/reference_info.fif

Sorties :
    outputs_microstates/microstate_model.npz      (templates, noms, K, GEV)
    outputs_microstates/clustering_metrics.csv
    outputs_microstates/microstate_K_selection.png
    outputs_microstates/microstate_maps.png
"""

import os

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

import config
import microstate_utils as msu


OUTPUT_MODEL = os.path.join(config.OUTPUT_DIR, "microstate_model.npz")
OUTPUT_METRICS = os.path.join(config.OUTPUT_DIR, "clustering_metrics.csv")
OUTPUT_FIGURE = os.path.join(config.OUTPUT_DIR, "microstate_K_selection.png")
OUTPUT_MAPS_FIG = os.path.join(config.OUTPUT_DIR, "microstate_maps.png")

INPUT_MAPS = os.path.join(config.OUTPUT_DIR, "gfp_peak_maps.npy")
INPUT_METADATA = os.path.join(config.OUTPUT_DIR, "gfp_peak_metadata.csv")
INPUT_INFO = os.path.join(config.OUTPUT_DIR, "reference_info.fif")


# ==========================================================
# CHARGEMENT
# ==========================================================

maps = np.load(INPUT_MAPS)                        # (n_peaks, n_channels)
metadata = pd.read_csv(INPUT_METADATA)
info = mne.io.read_info(INPUT_INFO)
latencies = metadata["time_ms"].values

print("Cartes chargees :", maps.shape)
print("Polarite :", "ignoree" if config.IGNORE_POLARITY else "prise en compte")


# ==========================================================
# TEST DES DIFFERENTS K
# ==========================================================

results = []
models = {}
dispersion = {}

k_list = list(config.K_RANGE)
# KL_K a besoin de W_{K-1} et W_{K+1} -> on ajuste aussi K_max + 1
# (non propose comme solution). Pas de K = 1 : avec la polarite, une
# solution a un seul etat ne peut pas expliquer les cartes de signe
# oppose, ce qui gonflerait artificiellement KL a K = 2 -> KL n'est
# calcule qu'a partir de K_min + 1.
k_fit = k_list + [k_list[-1] + 1]

for K in k_fit:

    print("\nTest K =", K, "" if K in k_list else "(pour le critere KL uniquement)")

    fit = msu.fit_kmeans(
        maps,
        n_clusters=K,
        n_init=config.N_INIT,
        max_iter=config.MAX_ITER,
        random_state=config.RANDOM_STATE,
        ignore_polarity=config.IGNORE_POLARITY,
    )
    # dispersion = variance residuelle ponderee par le GFP (1 - GEV)
    dispersion[K] = 1 - fit["GEV"]
    if K not in k_list:
        continue

    sil, ch_score, db_score = msu.clustering_scores(
        maps, fit["templates"], fit["labels"], config.IGNORE_POLARITY
    )

    results.append(
        {
            "K": K,
            "GEV": fit["GEV"],
            "W_residual": dispersion[K],
            "silhouette": sil,
            "calinski_harabasz": ch_score,
            "davies_bouldin": db_score,
        }
    )
    models[K] = fit

    print(
        f"  GEV={fit['GEV']:.3f}  silhouette={sil:.3f}  "
        f"CH={ch_score:.1f}  DB={db_score:.3f}"
    )

metrics = pd.DataFrame(results)

kl = msu.krzanowski_lai(dispersion, n_channels=maps.shape[1])
metrics["KL"] = metrics["K"].map(kl)
metrics["GEV_gain"] = metrics["GEV"].diff()


# ==========================================================
# CHOIX DE K : rang moyen sur 3 indices independants du GEV
# (silhouette haut = mieux, CH haut = mieux, DB bas = mieux)
# ==========================================================

metrics["rank_silhouette"] = metrics["silhouette"].rank(ascending=False)
metrics["rank_calinski_harabasz"] = metrics["calinski_harabasz"].rank(ascending=False)
metrics["rank_davies_bouldin"] = metrics["davies_bouldin"].rank(ascending=True)
metrics["mean_rank"] = metrics[
    ["rank_silhouette", "rank_calinski_harabasz", "rank_davies_bouldin"]
].mean(axis=1)

metrics.to_csv(OUTPUT_METRICS, index=False)
print("\n", metrics)

rank_K = int(metrics.sort_values("mean_rank").iloc[0]["K"])
kl_K = int(metrics.loc[metrics["KL"].idxmax(), "K"])
print("\nK selon Krzanowski-Lai (KL max)            :", kl_K)
print("K selon le rang moyen (silhouette / CH / DB) :", rank_K)

auto_K = kl_K if config.K_CRITERION == "KL" else rank_K
print(f"-> critere retenu (config.K_CRITERION = '{config.K_CRITERION}') : K = {auto_K}")

if config.K_FORCED is not None:
    best_K = int(config.K_FORCED)
    print(f"K impose par config.K_FORCED : {best_K}"
          + ("" if best_K == auto_K else f"  (le critere automatique donnait {auto_K})"))
else:
    best_K = auto_K
    print("-> Inspectez", OUTPUT_FIGURE, "avant de valider ce choix "
          "(config.K_FORCED permet d'imposer un autre K).")

fit = models[best_K]

# renumerotation par latence (MS1 = le plus precoce)
templates, labels, med_lat = msu.order_templates_by_latency(
    fit["templates"], fit["labels"], latencies
)
names = [f"MS{i + 1}" for i in range(best_K)]

print("\nTemplates retenus (latence mediane des pics GFP assignes) :")
for n, m, k in zip(names, med_lat, range(best_K)):
    print(f"  {n} : {m:.0f} ms  ({np.sum(labels == k)} cartes)")
print(f"GEV (cartes aux pics GFP) : {fit['GEV']:.3f}")

msu.save_model(OUTPUT_MODEL, templates, names, config.IGNORE_POLARITY, fit["GEV"])
print("\nModele sauvegarde :", OUTPUT_MODEL)


# ==========================================================
# VISUALISATION CHOIX DE K
# ==========================================================

fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

axes[0].plot(metrics.K, metrics.GEV, "-o", label="GEV (indicatif seulement)")
axes[0].set_ylabel("GEV")
axes[0].legend()
axes[0].set_title("Selection du nombre de micro-etats K")

axes[2].plot(metrics.K, metrics.KL, "-o", color="black", label="Krzanowski-Lai (max = K optimal)")
axes[2].axvline(best_K, color="black", linestyle="--")
axes[2].set_ylabel("KL")
axes[2].legend(fontsize=8)

axes[1].plot(metrics.K, metrics.silhouette, "-o", label="Silhouette (haut=mieux)")
axes[1].plot(
    metrics.K,
    metrics.calinski_harabasz / metrics.calinski_harabasz.max(),
    "-o",
    label="Calinski-Harabasz (normalise, haut=mieux)",
)
axes[1].plot(
    metrics.K,
    1 - metrics.davies_bouldin / metrics.davies_bouldin.max(),
    "-o",
    label="1 - Davies-Bouldin (normalise, haut=mieux)",
)
axes[1].axvline(best_K, color="black", linestyle="--", label=f"K retenu = {best_K}")
if best_K != auto_K:
    axes[1].axvline(auto_K, color="grey", linestyle=":", label=f"K automatique = {auto_K}")
axes[2].set_xlabel("Nombre de micro-etats K")
axes[1].set_ylabel("Indices (normalises)")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_FIGURE, dpi=300)
plt.close()
print("Figure sauvegardee :", OUTPUT_FIGURE)

fig = msu.plot_templates(templates, names, info, title="Microstate templates")
fig.savefig(OUTPUT_MAPS_FIG, dpi=300)
plt.close(fig)
print("Figure sauvegardee :", OUTPUT_MAPS_FIG)

