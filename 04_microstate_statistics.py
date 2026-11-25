# -*- coding: utf-8 -*-

"""
Analyse statistique des micro-etats


Ce que ce script corrige par rapport a la version precedente :

- Les transitions (comptages d'evenements) sont modelisees avec une
  GEE Poisson (groups=subject, correlation "exchangeable"), pas avec
  un modele lineaire mixte gaussien. Des comptages de transitions ne
  sont pas des donnees continues normalement distribuees : les
  utiliser dans un mixedlm() classique viole les hypotheses du modele,
  surtout avec beaucoup de zeros.

- Les parametres continus (coverage, duree, occurrence, GEV,
  correlation moyenne) restent modelises par mixedlm, ce qui est
  approprie ici.

- Facteurs en codage somme (-0.5 / +0.5) dans tous les modeles
  (mixedlm et GEE) : les effets Reality et Valence sont de vrais
  effets principaux, moyennes sur l'autre facteur, et non des effets
  simples au niveau de reference (cas du codage de traitement).

- mean_gfp_uv (amplitude GFP moyenne par etat) et peak_gfp_latency_ms /
  peak_gfp_uv (latence et amplitude du pic de GFP global de l'ERP)
  sont testes en plus des 5 parametres pycrostates de base. Ces deux
  derniers ne sont pas decomposes par micro-etat (une seule valeur par
  sujet/condition) et apparaissent dans la heatmap sous le
  pseudo-microstate "ERP".

Entrees :
    outputs_microstates/microstate_parameters.csv
    outputs_microstates/microstate_transitions.csv
    outputs_microstates/gfp_peak_latency.csv

Sorties :
    outputs_microstates/results_microstates.csv
    outputs_microstates/results_transitions.csv
    outputs_microstates/figures/heatmap_statistics.png
    outputs_microstates/figures/heatmap_statistics_p_raw.png
    outputs_microstates/figures/heatmap_gfp_peak.png
    outputs_microstates/figures/heatmap_gfp_peak_p_raw.png
    outputs_microstates/figures/significant_effects.png
    outputs_microstates/figures/transition_matrix.png
    outputs_microstates/figures/GEV_plot.png
"""

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Poisson
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.stats.multitest import multipletests

import config

# ==========================================================
# CODAGE SOMME (-0.5 / +0.5) DES FACTEURS
# ==========================================================
#
# Avec un terme d'interaction, un codage de traitement (0/1, niveau de
# reference) fait des coefficients "Reality" et "Valence" des effets
# SIMPLES (Reality chez APP seulement, Valence chez VIRTUAL seulement).
# Le codage somme -0.5/+0.5 donne de vrais effets principaux, moyennes
# sur l'autre facteur :
#   real_c  : REAL = +0.5, VIRTUAL = -0.5  -> beta = REAL - VIRTUAL
#   disg_c  : DISG = +0.5, APP     = -0.5  -> beta = DISG - APP
#   real_c:disg_c -> (REAL: DISG-APP) - (VIRTUAL: DISG-APP)
# Les noms d'effets en sortie sont conserves (Reality[T.REAL], ...)
# pour que les figures et les CSV restent compatibles.

EFFECT_TERMS = {
    "real_c": "Reality[T.REAL]",
    "disg_c": "Valence[T.DISG]",
    "real_c:disg_c": "Reality[T.REAL]:Valence[T.DISG]",
}


# Une paire de transitions n'est modelisee que si au moins ce nombre de
# sujets la presente au moins une fois. En dessous, la GEE Poisson ne
# converge pas (separation : beta de +/-15 a 30, p = 1 ou NaN).
MIN_SUBJECTS_WITH_TRANSITION = 10


def fdr(pvalues):
    """Benjamini-Hochberg en ignorant les p manquantes (multipletests
    renvoie NaN pour TOUTES les lignes des qu'une seule p vaut NaN)."""
    p = pd.Series(pvalues, dtype=float)
    out = pd.Series(np.nan, index=p.index)
    ok = p.notna()
    if ok.any():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out.values


def add_sum_coding(d):
    """Ajoute les colonnes real_c / disg_c (codage -0.5 / +0.5)."""
    d = d.copy()
    d["real_c"] = np.where(d["Reality"] == "REAL", 0.5, -0.5)
    d["disg_c"] = np.where(d["Valence"] == "DISG", 0.5, -0.5)
    return d



FIGDIR = os.path.join(config.OUTPUT_DIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)

PARAMETERS_FILE = os.path.join(config.OUTPUT_DIR, "microstate_parameters.csv")
TRANSITIONS_FILE = os.path.join(config.OUTPUT_DIR, "microstate_transitions.csv")

PARAMS = [
    "coverage_percent",
    "mean_duration_ms",
    "occurrence_per_sec",
    "mean_corr",
    "GEV",
    "mean_gfp_uv",  # amplitude GFP moyenne pendant l'etat (ajoute en 03)
]

# Pic de GFP global (latence + amplitude) : propriete de l'ERP entier,
# une seule valeur par sujet/condition (pas de decomposition par
# micro-etat) -> modelise a part, voir plus bas.
GFP_PEAK_FILE = os.path.join(config.OUTPUT_DIR, "gfp_peak_latency.csv")
GFP_PEAK_PARAMS = ["peak_gfp_latency_ms", "peak_gfp_uv"]


# ==========================================================
# CHARGEMENT
# ==========================================================

df = pd.read_csv(PARAMETERS_FILE)
df["Reality"] = df.condition.apply(config.reality)
df["Valence"] = df.condition.apply(config.valence)


# ==========================================================
# MODELES SUR LES PARAMETRES CONTINUS (mixedlm)
# ==========================================================

results = []

for ms in sorted(df.microstate.unique()):

    print("Microstate", ms)

    data_ms = df[df.microstate == ms]

    for param in PARAMS:

        data = data_ms[["subject", "Reality", "Valence", param]].dropna()

        # codage somme -0.5/+0.5 -> vrais effets principaux (voir
        # add_sum_coding en haut du fichier)
        data = add_sum_coding(data)

        if len(data) < 20:
            continue

        try:
            model = smf.mixedlm(
                f"{param} ~ real_c * disg_c", data, groups=data.subject
            )
            fit = model.fit(reml=False)

            for term, effect in EFFECT_TERMS.items():
                if term in fit.params.index:

                    beta = fit.params[term]
                    p = fit.pvalues[term]

                    sd = data[param].std()
                    effect_size = beta / sd if sd > 0 else np.nan

                    results.append(
                        {
                            "microstate": ms,
                            "parameter": param,
                            "effect": effect,
                            "beta": beta,
                            "effect_size": effect_size,
                            "p": p,
                        }
                    )

        except Exception as e:
            print(" ", param, ":", e)


# ==========================================================
# PIC DE GFP GLOBAL (latence + amplitude)
# ==========================================================
#
# peak_gfp_latency_ms / peak_gfp_uv sont des proprietes de l'ERP
# entier (une valeur par sujet/condition), pas d'un micro-etat en
# particulier -> pas de boucle "for ms in ...", un seul modele par
# parametre. Les resultats sont ajoutes a la meme liste "results" sous
# un pseudo-microstate "ERP" pour apparaitre dans la meme heatmap /
# le meme CSV que les parametres par micro-etat.

if os.path.exists(GFP_PEAK_FILE):

    gfp_peak_df = pd.read_csv(GFP_PEAK_FILE)
    gfp_peak_df["Reality"] = gfp_peak_df.condition.apply(config.reality)
    gfp_peak_df["Valence"] = gfp_peak_df.condition.apply(config.valence)

    for param in GFP_PEAK_PARAMS:

        data = gfp_peak_df[["subject", "Reality", "Valence", param]].dropna()

        # meme codage somme que plus haut
        data = add_sum_coding(data)

        if len(data) < 20:
            print(f"  {param} (GFP global) : pas assez de lignes, ignore")
            continue

        try:
            model = smf.mixedlm(
                f"{param} ~ real_c * disg_c", data, groups=data.subject
            )
            fit = model.fit(reml=False)

            for term, effect in EFFECT_TERMS.items():
                if term in fit.params.index:

                    beta = fit.params[term]
                    p = fit.pvalues[term]

                    sd = data[param].std()
                    effect_size = beta / sd if sd > 0 else np.nan

                    results.append(
                        {
                            "microstate": "ERP",
                            "parameter": param,
                            "effect": effect,
                            "beta": beta,
                            "effect_size": effect_size,
                            "p": p,
                        }
                    )

        except Exception as e:
            print(" ", param, "(GFP global) :", e)
else:
    print(f"\n{GFP_PEAK_FILE} introuvable -> relance le script 03 pour le generer.")


# colonnes fixees explicitement -> le DataFrame garde le bon schema
# meme si aucun modele n'a pu etre ajuste (pas assez de sujets/lignes),
# au lieu de planter plus loin sur un DataFrame sans colonnes.
results = pd.DataFrame(
    results, columns=["microstate", "parameter", "effect", "beta", "effect_size", "p"]
)

if results.empty:
    print(
        "\nATTENTION : aucun effet n'a pu etre estime (moins de 20 lignes "
        "disponibles par micro-etat/parametre -> augmente DEBUG_MAX_SUBJECTS "
        "ou passe-le a None). results_microstates.csv sera vide."
    )
    results["p_FDR"] = pd.Series(dtype=float)
    results["significant"] = pd.Series(dtype=bool)
    results["significant_uncorrected"] = pd.Series(dtype=bool)
else:
    results["p_FDR"] = np.nan
    # PARAMS (par micro-etat) + GFP_PEAK_PARAMS (pic GFP global,
    # pseudo-microstate "ERP") : chaque parametre reste sa propre
    # famille de correction (voir commentaire plus haut), mais les
    # deux groupes doivent bien etre parcourus, sinon les lignes
    # peak_gfp_latency_ms / peak_gfp_uv gardent un p_FDR = NaN et ne
    # sont jamais marquees significatives, quel que soit leur p brut.
    for param in PARAMS + GFP_PEAK_PARAMS:
        idx = results.parameter == param
        if idx.sum() > 0:
            results.loc[idx, "p_FDR"] = fdr(results.loc[idx, "p"])

    results["significant"] = results.p_FDR < 0.05

    # p brute (non corrigee) : signal exploratoire uniquement, risque
    # de faux positifs bien plus eleve qu'avec p_FDR -> a ne jamais
    # presenter comme une preuve, mais utile pour reperer une tendance
    # a re-tester (voir discussion sur l'effet Valence disparu avec
    # K=4 micro-etats).
    results["significant_uncorrected"] = results.p < 0.05

results.to_csv(os.path.join(config.OUTPUT_DIR, "results_microstates.csv"), index=False)


if not results.empty:

    # --- figure(s) : heatmap p_FDR ---
    #
    # Les parametres par micro-etat (GEV, coverage_percent...) et les
    # parametres globaux du pic de GFP (pseudo-microstate "ERP") n'ont
    # pas la meme forme (K colonnes vs 1 seule) : les melanger dans un
    # seul pivot_table produit une grille avec un enorme bloc de
    # cellules vides (NaN) en haut a droite et en bas a gauche - moche
    # et trompeur (l'echelle de couleur se recale aussi sur le max
    # global, ce qui peut faire paraitre "chaud" un effet non
    # significatif). On separe donc en deux figures independantes.

    SIG_THRESHOLD = -np.log10(0.05)  # ~1.30, correspond a p_FDR < .05

    def _plot_significance_heatmap(
        sub_results, columns_key, title, out_path, figsize,
        value_col="p_FDR", cbar_label="-log10(FDR)",
    ):
        """Heatmap -log10(value_col) + encadres rouge (FDR) / orange (p
        brut non corrige), sur un sous-ensemble homogene de resultats
        (meme jeu de colonnes pour toutes les lignes -> pas de trous).

        value_col choisit ce qui est affiche/colore dans les cellules
        (p_FDR ou p brute) ; les deux cadres de couleur (rouge = FDR
        significatif, orange = p brute < .05) restent bases sur les
        memes seuils quel que soit value_col, pour pouvoir comparer
        directement les deux versions de la heatmap."""

        pivot = sub_results.pivot_table(
            index=["parameter", "effect"], columns=columns_key, values=value_col
        )
        pivot_fdr = sub_results.pivot_table(
            index=["parameter", "effect"], columns=columns_key, values="p_FDR"
        )
        pivot_raw = sub_results.pivot_table(
            index=["parameter", "effect"], columns=columns_key, values="p"
        )

        fig, ax = plt.subplots(figsize=figsize)
        heat_data = -np.log10(pivot)
        sns.heatmap(
            heat_data,
            annot=True,
            cmap="viridis",
            # echelle de couleur fixee au moins jusqu'au seuil de
            # significativite -> une case non significative ne peut
            # jamais paraitre "chaude" juste parce qu'elle est la plus
            # grande valeur de ce sous-ensemble particulier.
            vmin=0,
            vmax=max(SIG_THRESHOLD * 1.5, np.nanmax(heat_data.values) if heat_data.size else SIG_THRESHOLD),
            cbar_kws={"label": cbar_label},
            ax=ax,
        )

        # Encadre en orange les cellules avec p brute < .05 (signal
        # exploratoire uniquement), puis en rouge par-dessus celles qui
        # survivent en plus a la correction FDR (significatif au sens
        # strict) -> les deux niveaux restent visuellement distincts,
        # et identiques que l'on affiche p_FDR ou p brute.
        for (row_idx, row_name) in enumerate(pivot.index):
            for (col_idx, col_name) in enumerate(pivot.columns):

                raw_p = pivot_raw.loc[row_name, col_name]
                if pd.notna(raw_p) and raw_p < 0.05:
                    ax.add_patch(
                        plt.Rectangle(
                            (col_idx, row_idx), 1, 1,
                            fill=False, edgecolor="orange", lw=2.0,
                        )
                    )

                fdr_val = pivot_fdr.loc[row_name, col_name]
                if pd.notna(fdr_val) and fdr_val < 0.05:
                    ax.add_patch(
                        plt.Rectangle(
                            (col_idx, row_idx), 1, 1,
                            fill=False, edgecolor="red", lw=2.5,
                        )
                    )

        ax.set_title(title)
        fig.text(
            0.5, 0.005,
            f"Seuil de significativite : -log10(FDR) > {SIG_THRESHOLD:.2f}  (equivaut a p_FDR < .05)  |  "
            "rouge = significatif (FDR)  |  orange = p brut < .05 non corrige (exploratoire)",
            ha="center", va="bottom", fontsize=9, style="italic",
        )
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    results_ms = results[results.microstate != "ERP"]
    results_erp = results[results.microstate == "ERP"]

    if not results_ms.empty:
        _plot_significance_heatmap(
            results_ms,
            columns_key="microstate",
            title="Microstate statistics (FDR)",
            out_path=os.path.join(FIGDIR, "heatmap_statistics.png"),
            figsize=(10, 8.6),
        )
        # Meme grille, mais avec la p brute (non corrigee) affichee et
        # coloree au lieu de p_FDR -> demande explicite pour voir la
        # valeur exacte, pas juste un cadre orange indiquant p < .05.
        _plot_significance_heatmap(
            results_ms,
            columns_key="microstate",
            title="Microstate statistics (p brute, non corrigee)",
            out_path=os.path.join(FIGDIR, "heatmap_statistics_p_raw.png"),
            figsize=(10, 8.6),
            value_col="p",
            cbar_label="-log10(p) non corrige",
        )

    if not results_erp.empty:
        # Une seule "colonne" (pas de decomposition par micro-etat) ->
        # figure etroite dediee, plutot que de forcer ces lignes dans
        # la grille MS1..MSk ou elles ne collent pas.
        _plot_significance_heatmap(
            results_erp,
            columns_key="microstate",  # vaut toujours "ERP" ici
            title="Pic de GFP global (latence / amplitude) - FDR",
            out_path=os.path.join(FIGDIR, "heatmap_gfp_peak.png"),
            figsize=(6.5, 4.5),
        )
        _plot_significance_heatmap(
            results_erp,
            columns_key="microstate",
            title="Pic de GFP global (latence / amplitude) - p brute",
            out_path=os.path.join(FIGDIR, "heatmap_gfp_peak_p_raw.png"),
            figsize=(6.5, 4.5),
            value_col="p",
            cbar_label="-log10(p) non corrige",
        )

    # --- figure : effets significatifs ---

    sig = results[results.significant].copy()
    SIG_FIG_PATH = os.path.join(FIGDIR, "significant_effects.png")

    if len(sig) > 0:
        # Etiquette combinant parametre + type d'effet (main effect
        # Reality/Valence ou interaction). Sans ca, x=microstate /
        # hue=parameter ne distinguent pas deux effets significatifs
        # differents tombant sur le meme (microstate, parametre)
        # (ex. Valence et l'interaction sur MS1/occurrence_per_sec) :
        # seaborn les moyennait silencieusement et n'affichait plus
        # qu'une barre d'erreur quasi invisible au lieu de deux barres.
        effect_labels = {
            "Reality[T.REAL]": "Reality",
            "Valence[T.DISG]": "Valence",
            "Reality[T.REAL]:Valence[T.DISG]": "Reality x Valence",
        }
        sig["effect_label"] = sig["effect"].map(effect_labels).fillna(sig["effect"])
        sig["param_effect"] = sig["parameter"] + "\n(" + sig["effect_label"] + ")"

        plt.figure(figsize=(max(8, 1.6 * sig["param_effect"].nunique()), 5))
        sns.barplot(data=sig, x="param_effect", y="effect_size", hue="microstate")
        plt.axhline(0, color="black")
        plt.title("Significant microstate effects")
        plt.xlabel("Parameter (effect)")
        plt.tight_layout()
        plt.savefig(SIG_FIG_PATH, dpi=300)
        plt.close()
    else:
        # Aucun effet significatif ce run-ci. Un fichier
        # significant_effects.png peut deja exister sur le disque
        # depuis un run precedent (K different, autre sous-ensemble de
        # sujets...) : le laisser tel quel donnerait l'impression
        # trompeuse qu'il reflete le resultat actuel. On le regenere
        # systematiquement avec un message explicite plutot que de
        # sauter la sauvegarde.
        if os.path.exists(SIG_FIG_PATH):
            os.remove(SIG_FIG_PATH)
        plt.figure(figsize=(8, 3))
        plt.axis("off")
        plt.text(
            0.5, 0.5,
            "Aucun effet significatif (p_FDR < .05)\n"
            "sur les parametres continus dans ce run.",
            ha="center", va="center", fontsize=13,
        )
        plt.savefig(SIG_FIG_PATH, dpi=300)
        plt.close()
        print(
            "Pas d'effet significatif -> significant_effects.png regenere "
            "avec un message explicite (l'ancien fichier eventuel a ete supprime)."
        )
else:
    print("Pas de figure generee pour les parametres continus (resultats vides).")


# ==========================================================
# TRANSITIONS : GEE Poisson (comptages) au lieu de mixedlm gaussien
# ==========================================================

trans = pd.read_csv(TRANSITIONS_FILE)
trans["Reality"] = trans.condition.apply(config.reality)
trans["Valence"] = trans.condition.apply(config.valence)

states = sorted(set(trans["from"]) | set(trans["to"]))

transition_results = []

for a in states:
    for b in states:

        if a == b:
            continue

        dat = trans[(trans["from"] == a) & (trans["to"] == b)].copy()

        # meme codage somme que pour les modeles mixedlm ci-dessus
        # (effets principaux sur l'echelle log de la GEE Poisson)
        dat = add_sum_coding(dat)

        n_subj_with = dat.groupby("subject")["count"].sum().gt(0).sum()
        if n_subj_with < MIN_SUBJECTS_WITH_TRANSITION:
            print(f"Transition {a} -> {b} : {n_subj_with} sujet(s) seulement, non modelisee")
            continue

        try:
            gee_model = GEE.from_formula(
                "count ~ real_c * disg_c",
                groups="subject",
                data=dat,
                family=Poisson(),
                cov_struct=Exchangeable(),
            )
            fit = gee_model.fit()

            for term, effect in EFFECT_TERMS.items():
                if term in fit.params.index:
                    transition_results.append(
                        {
                            "from": a,
                            "to": b,
                            "effect": effect,
                            "beta": fit.params[term],
                            "p": fit.pvalues[term],
                        }
                    )

        except Exception as e:
            print("Transition", a, "->", b, ":", e)


# colonnes fixees explicitement -> le CSV garde des en-tetes valides
# meme quand aucune transition n'a pu etre modelisee (moins de 3 sujets
# par paire d'etats), au lieu d'ecrire un fichier totalement vide que
# pandas refuse ensuite de relire (EmptyDataError).
transition_results = pd.DataFrame(
    transition_results, columns=["from", "to", "effect", "beta", "p"]
)

if len(transition_results) > 0:
    transition_results["p_FDR"] = fdr(transition_results["p"])
    transition_results["significant"] = transition_results.p_FDR < 0.05

transition_results.to_csv(
    os.path.join(config.OUTPUT_DIR, "results_transitions.csv"), index=False
)


# --- figure : matrice de transitions moyenne (comptages) ---

mean_counts = trans.groupby(["from", "to"])["count"].mean().reset_index()

matrix = pd.DataFrame(0.0, index=states, columns=states)
for _, row in mean_counts.iterrows():
    matrix.loc[row["from"], row["to"]] = row["count"]

plt.figure(figsize=(6, 5))
sns.heatmap(matrix, annot=True, fmt=".1f", cmap="magma")
plt.xlabel("To")
plt.ylabel("From")
plt.title("Mean microstate transitions (counts)")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "transition_matrix.png"), dpi=300)
plt.close()


# --- figure : GEV moyen par condition ---

gev = df.groupby(["condition", "microstate"])["GEV"].mean().reset_index()

plt.figure(figsize=(8, 5))
sns.barplot(data=gev, x="microstate", y="GEV", hue="condition")
plt.title("Mean GEV per condition")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "GEV_plot.png"), dpi=300)
plt.close()

print("\nAnalyse terminee")
