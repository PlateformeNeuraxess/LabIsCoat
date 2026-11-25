# -*- coding: utf-8 -*-

"""
Outils de segmentation en micro-etats EEG, AVEC prise en compte de la
polarite (adapte aux ERP).

Pourquoi ce module :
    pycrostates impose l'invariance de polarite a toutes les etapes
    (clustering, backfitting, lissage, correlation moyenne) : une carte
    et son inverse de signe y sont toujours le meme etat. C'est la
    convention pour l'EEG spontane, mais pas pour les ERP, ou la
    polarite est porteuse d'information (une P1 posterieure positive et
    une composante tardive de topographie inverse sont deux etats
    differents ; Murray, Brunet & Michel, 2008). pycrostates n'offre pas
    d'option pour la desactiver, d'ou ce module.

    Les algorithmes reprennent ceux de pycrostates 0.6 (k-means modifie,
    lissage de Pascual-Marqui et al. 1995, rejet des segments courts,
    parametres, transitions) ; seule la regle d'appariement change :
        - ignore_polarity=False (defaut ERP) : correlation SIGNEE,
          une carte ne peut etre expliquee que par un template de meme
          signe (amplitude >= 0).
        - ignore_polarity=True : correlation en valeur absolue,
          comportement identique a pycrostates (utile pour comparer).

Toutes les cartes/donnees sont supposees en reference moyenne
(fait dans 01 / 03 / 05 via set_eeg_reference("average")).
"""

from itertools import groupby

import numpy as np
from scipy.signal import convolve2d


# ==========================================================
# OUTILS DE BASE
# ==========================================================


def _normalize_columns(data):
    """Centre (reference moyenne) et norme a 1 chaque colonne
    (= chaque carte / chaque instant). Colonnes nulles laissees a 0."""
    data = data - data.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(data, axis=0, keepdims=True)
    norm[norm == 0] = 1
    return data / norm


def _normalize_rows(maps):
    """Idem pour des templates (n_states, n_channels)."""
    return _normalize_columns(maps.T).T


def spatial_correlation(templates, data):
    """Correlation spatiale (signee) entre chaque template et chaque
    instant. templates : (K, n_ch) ; data : (n_ch, n_t) -> (K, n_t)."""
    return _normalize_rows(templates) @ _normalize_columns(data)


def _match(corr, ignore_polarity):
    """Valeur d'appariement a maximiser."""
    return np.abs(corr) if ignore_polarity else corr


# ==========================================================
# CLUSTERING (k-means modifie, avec ou sans polarite)
# ==========================================================


def _kmeans_single(data, n_clusters, max_iter, tol, rng, ignore_polarity):
    """Un run de k-means. data : (n_ch, n_maps), reference moyenne,
    NON normalise (les cartes de fort GFP pesent plus, comme dans
    pycrostates). Renvoie (templates (K, n_ch) normes, converged)."""

    n_ch, n_samples = data.shape
    data_sum_sq = np.sum(data ** 2)

    init = rng.choice(n_samples, size=n_clusters, replace=False)
    maps = data[:, init].T.copy()
    maps /= np.linalg.norm(maps, axis=1, keepdims=True)

    prev_residual = np.inf
    converged = False

    for _ in range(max_iter):

        activation = maps @ data                         # (K, n_maps)
        labels = np.argmax(_match(activation, ignore_polarity), axis=0)

        for k in range(n_clusters):
            idx = labels == k
            if not np.any(idx):
                maps[k] = 0
                continue
            act = activation[k, idx]
            if not ignore_polarity:
                # avec polarite : seules les cartes de meme signe que
                # le template contribuent (amplitude >= 0)
                act = np.clip(act, 0, None)
            # pas de "power iteration" (= moyenne ponderee par
            # l'activation, cf. pycrostates / Pascual-Marqui 1995)
            new = data[:, idx] @ act
            n = np.linalg.norm(new)
            maps[k] = new / n if n > 0 else 0

        act_assigned = np.sum(maps[labels].T * data, axis=0)
        if not ignore_polarity:
            act_assigned = np.clip(act_assigned, 0, None)
        residual = abs(data_sum_sq - np.sum(act_assigned ** 2))
        residual /= float(n_samples * (n_ch - 1))

        if (prev_residual - residual) < (tol * residual):
            converged = True
            break
        prev_residual = residual

    return maps, converged


def global_explained_variance(data, templates, labels, ignore_polarity):
    """GEV = sum(GFP^2 * r^2) / sum(GFP^2), r = correlation avec le
    template assigne (signee et bornee a 0 si on tient compte de la
    polarite : une carte de signe oppose n'est pas expliquee)."""
    gfp = np.std(data, axis=0)
    corr = spatial_correlation(templates, data)[labels, np.arange(data.shape[1])]
    corr = np.abs(corr) if ignore_polarity else np.clip(corr, 0, None)
    return np.sum((gfp * corr) ** 2) / np.sum(gfp ** 2)


def fit_kmeans(maps, n_clusters, n_init=100, max_iter=300, tol=1e-6,
               random_state=42, ignore_polarity=False):
    """k-means sur les cartes aux pics de GFP.

    maps : (n_maps, n_channels), reference moyenne.
    Renvoie un dict : templates (K, n_ch) normes, labels (n_maps,),
    GEV, converged. Le meilleur des n_init runs (GEV max) est retenu.
    """
    data = np.asarray(maps, dtype=float).T
    data = data - data.mean(axis=0, keepdims=True)
    data = data[:, np.linalg.norm(data, axis=0) > 0]

    rng = np.random.RandomState(random_state)

    best = None
    for _ in range(n_init):
        templates, converged = _kmeans_single(
            data, n_clusters, max_iter, tol, rng, ignore_polarity
        )
        labels = np.argmax(
            _match(spatial_correlation(templates, data), ignore_polarity), axis=0
        )
        gev = global_explained_variance(data, templates, labels, ignore_polarity)
        if best is None or gev > best["GEV"]:
            best = dict(templates=templates, labels=labels, GEV=gev,
                        converged=converged)

    best["templates"] = _normalize_rows(best["templates"])
    return best


def clustering_scores(maps, templates, labels, ignore_polarity):
    """Silhouette, Calinski-Harabasz, Davies-Bouldin sur les cartes
    normalisees. Sur des cartes centrees et de norme 1, la distance
    euclidienne au carre vaut 2(1 - r) : ces indices sont donc bases sur
    la correlation spatiale signee. En mode invariant, chaque carte est
    d'abord retournee pour etre de meme signe que son template."""
    from sklearn.metrics import (
        silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    )

    X = _normalize_columns(np.asarray(maps, dtype=float).T)    # (n_ch, n)
    if ignore_polarity:
        sign = np.sign(np.sum(templates[labels].T * X, axis=0))
        sign[sign == 0] = 1
        X = X * sign
    X = X.T

    if len(np.unique(labels)) < 2:
        return np.nan, np.nan, np.nan

    return (
        silhouette_score(X, labels, metric="euclidean"),
        calinski_harabasz_score(X, labels),
        davies_bouldin_score(X, labels),
    )


def krzanowski_lai(W, n_channels):
    """Critere de Krzanowski-Lai (Krzanowski & Lai, 1988 ; Murray et
    al., 2008).

    W : dict {K: W_K}, K consecutifs, W_K = dispersion de la solution a
    K etats. On utilise W_K = 1 - GEV_K, c.-a-d. la variance residuelle
    ponderee par le GFP, qui est la quantite minimisee par le k-means :
    les pics de faible GFP (bruit) pesent peu, ce qui stabilise le
    critere.
        diff_K = (K-1)^(2/p) W_{K-1} - K^(2/p) W_K
        KL_K   = |diff_K| / |diff_{K+1}|
    p = nombre d'electrodes. Le K optimal maximise KL. KL_K n'est
    calcule que si W_{K-1} et W_{K+1} existent.
    Renvoie un dict {K: KL_K} (NaN si non calculable)."""
    p = float(n_channels)

    def diff(K):
        return (K - 1) ** (2 / p) * W[K - 1] - K ** (2 / p) * W[K]

    out = {}
    for K in sorted(W):
        if (K - 1) in W and (K + 1) in W:
            d1, d2 = diff(K), diff(K + 1)
            out[K] = abs(d1) / abs(d2) if d2 != 0 else np.nan
        else:
            out[K] = np.nan
    return out


def order_templates_by_latency(templates, labels, latencies_ms):
    """Renumerote les templates par latence mediane des pics GFP qui leur
    sont assignes (MS1 = le plus precoce). Rend la numerotation stable et
    interpretable d'un run a l'autre (le k-means donne un ordre
    arbitraire)."""
    K = templates.shape[0]
    med = np.array([
        np.median(latencies_ms[labels == k]) if np.any(labels == k) else np.inf
        for k in range(K)
    ])
    order = np.argsort(med)
    new_labels = np.empty_like(labels)
    for new, old in enumerate(order):
        new_labels[labels == old] = new
    return templates[order], new_labels, med[order]


# ==========================================================
# BACKFITTING + LISSAGE
# ==========================================================


def _smooth_segmentation(data, states, labels, factor, tol, half_window_size,
                         ignore_polarity):
    """Lissage de Pascual-Marqui et al. (1995), repris de pycrostates.
    data : (n_ch, n_t) centre/reduit par instant ; states : (K, n_ch)
    centres/reduits. Avec polarite, l'amplitude d'un etat est bornee a
    0 : un template de signe oppose n'explique rien de l'instant."""

    Ne, Nt = data.shape
    Nu = states.shape[0]
    Vvar = np.sum(data * data, axis=0)
    rmat = np.tile(np.arange(Nu), (Nt, 1)).T

    def explained(act):
        return act ** 2 if ignore_polarity else np.clip(act, 0, None) ** 2

    act_all = states @ data                                     # (K, Nt)

    w = np.zeros((Nu, Nt))
    w[rmat == labels] = 1
    e = np.sum(Vvar - explained(act_all[labels, np.arange(Nt)])) / (Nt * (Ne - 1))
    window = np.ones((1, 2 * half_window_size + 1))

    S0 = 0
    while True:
        Nb = convolve2d(w, window, mode="same")
        x = (np.tile(Vvar, (Nu, 1)) - explained(act_all)) / (2 * e * (Ne - 1)) \
            - factor * Nb
        labels = np.argmin(x, axis=0)
        w = np.zeros((Nu, Nt))
        w[rmat == labels] = 1
        Su = np.sum(Vvar - explained(act_all[labels, np.arange(Nt)])) / (Nt * (Ne - 1))
        if np.abs(Su - S0) <= np.abs(tol * Su):
            break
        S0 = Su

    return labels


def _reject_short_segments(labels, data, min_segment_length, ignore_polarity):
    """Segments plus courts que min_segment_length (echantillons)
    reattribues aux voisins, echantillon par echantillon, du cote ou la
    carte est la plus correlee a sa voisine (repris de pycrostates).
    Les segments de bord sont conserves (comme pycrostates)."""

    def corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        r = float(a @ b / d) if d > 0 else 0.0
        return abs(r) if ignore_polarity else r

    labels = labels.copy()
    while True:
        segments = [list(g) for _, g in groupby(labels)]
        idx = 0
        for k, seg in enumerate(segments):
            if k in (0, len(segments) - 1) or len(seg) >= min_segment_length:
                idx += len(seg)
                continue
            left, right = idx, idx + len(seg) - 1
            while right >= left:
                lc = corr(data[:, left - 1], data[:, left])
                rc = corr(data[:, right], data[:, right + 1])
                if abs(rc - lc) <= 1e-8:
                    if right == left:
                        labels[left] = labels[left - 1]
                        left += 1
                    else:
                        labels[right] = labels[right + 1]
                        labels[left] = labels[left - 1]
                        right -= 1
                        left += 1
                elif lc < rc:
                    labels[right] = labels[right + 1]
                    right -= 1
                else:
                    labels[left] = labels[left - 1]
                    left += 1
            break  # re-lister les segments
        else:
            break
    return labels


def backfit(data, templates, factor=10, half_window_size=3, tol=1e-5,
            min_segment_length=0, ignore_polarity=False):
    """Attribue un micro-etat a chaque instant.

    data : (n_ch, n_t) en reference moyenne (ERP d'un sujet/condition).
    Renvoie labels (n_t,) entiers 0..K-1.
    """
    d = np.asarray(data, dtype=float)
    d = d - d.mean(axis=0, keepdims=True)
    std = d.std(axis=0)
    std[std == 0] = 1
    d = d / std

    states = np.asarray(templates, dtype=float)
    states = states - states.mean(axis=1, keepdims=True)
    states = states / states.std(axis=1, keepdims=True)

    labels = np.argmax(_match(states @ d, ignore_polarity), axis=0)

    if factor != 0:
        labels = _smooth_segmentation(
            d, states, labels, factor, tol, half_window_size, ignore_polarity
        )

    if min_segment_length > 0:
        labels = _reject_short_segments(
            labels, np.asarray(data, dtype=float), min_segment_length,
            ignore_polarity,
        )

    return labels


# ==========================================================
# PARAMETRES ET TRANSITIONS
# ==========================================================


def compute_parameters(data, labels, templates, sfreq, ignore_polarity=False):
    """Memes definitions que pycrostates Segmentation.compute_parameters :

    par etat k :
        mean_corr   : correlation spatiale moyenne (signee si polarite)
                      entre le template et les instants assignes
        gev         : sum(GFP^2 r^2) instants assignes / sum(GFP^2) total
                      (r borne a 0 si polarite)
        timecov     : proportion du temps (ratio)
        meandurs    : duree moyenne des segments (s)
        occurrences : nombre de segments par seconde
    Renvoie une liste de dicts (un par etat).
    """
    data = np.asarray(data, dtype=float)
    n_t = data.shape[1]
    gfp = np.std(data, axis=0)
    corr_all = spatial_correlation(templates, data)
    segments = [(s, len(list(g))) for s, g in groupby(labels)]

    out = []
    for k in range(templates.shape[0]):
        mask = labels == k
        if not np.any(mask):
            out.append(dict(mean_corr=0.0, gev=0.0, timecov=0.0,
                            meandurs=0.0, occurrences=0.0))
            continue
        r = corr_all[k, mask]
        r_used = np.abs(r) if ignore_polarity else r
        r_gev = np.abs(r) if ignore_polarity else np.clip(r, 0, None)
        durs = np.array([n for s, n in segments if s == k])
        out.append(dict(
            mean_corr=float(np.mean(r_used)),
            gev=float(np.sum((gfp[mask] * r_gev) ** 2) / np.sum(gfp ** 2)),
            timecov=float(durs.sum() / n_t),
            meandurs=float(np.mean(durs) / sfreq),
            occurrences=float(len(durs) / n_t * sfreq),
        ))
    return out


def transition_counts(labels, n_states):
    """Nombre de transitions i -> j entre segments successifs
    (auto-transitions ignorees), comme pycrostates
    compute_transition_matrix(stat="count", ignore_repetitions=True)."""
    seq = [s for s, _ in groupby(labels)]
    T = np.zeros((n_states, n_states), dtype=int)
    for i, j in zip(seq, seq[1:]):
        T[i, j] += 1
    return T


# ==========================================================
# SAUVEGARDE / CHARGEMENT DU MODELE
# ==========================================================


def save_model(path, templates, names, ignore_polarity, gev):
    np.savez(path, templates=templates, names=np.array(names),
             ignore_polarity=ignore_polarity, gev=gev)


def load_model(path):
    z = np.load(path, allow_pickle=False)
    return dict(
        templates=z["templates"],
        names=[str(n) for n in z["names"]],
        ignore_polarity=bool(z["ignore_polarity"]),
        gev=float(z["gev"]),
    )


def plot_templates(templates, names, info, title=None):
    """Topographies des templates (mne.viz.plot_topomap)."""
    import matplotlib.pyplot as plt
    import mne

    K = templates.shape[0]
    fig, axes = plt.subplots(1, K, figsize=(3 * K, 3.2))
    axes = np.atleast_1d(axes)
    vmax = np.max(np.abs(templates))
    for k, ax in enumerate(axes):
        mne.viz.plot_topomap(templates[k], info, axes=ax, show=False,
                             vlim=(-vmax, vmax), cmap="RdBu_r")
        ax.set_title(names[k])
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
