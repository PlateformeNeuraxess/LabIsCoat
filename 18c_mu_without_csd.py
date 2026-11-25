# -*- coding: utf-8 -*-
"""
18c - L'effet mu du reel existe-t-il SANS Laplacien (CSD) ? - Lab Is Coat

Pourquoi : l'effet mu (REAL > VIRT, central droit) a ete trouve sur des
donnees transformees en CSD. La localisation de source (DICS, script 18)
travaille, elle, sur le potentiel en reference moyenne, et ne retrouve pas
l'effet. Ce script refait la mesure du script 11/12 sur les memes epoques
(outputs_tfr/epochs, script 11b), avec et sans CSD, pour savoir si l'effet
est visible dans les donnees que voit le modele de source.

Mesure : puissance mu 8-13 Hz, 0.3-1.75 s, ROI centrale droite et gauche
(memes electrodes que le script 12), en dB / ligne de base commune
(-2.0 a -1.5 s, 4 conditions). Contraste REAL - VIRT, t apparie et dz.
"""

import glob
import os

import numpy as np
import pandas as pd
import mne
from scipy import stats

EPOCH_DIR = os.path.join("outputs_tfr", "epochs")
ROIS = {
    "Central droit": ["E183", "E184", "E196", "E182", "E164", "E195"],
    "Central gauche": ["E59", "E52", "E51", "E65", "E66", "E58"],
}
CONDITIONS = {"REAL_APP": ["VIS1", "VIS2"], "VIRT_APP": ["VIS3", "VIS4"],
              "REAL_DISG": ["VIS5", "VIS6"], "VIRT_DISG": ["VIS7", "VIS8"]}
FREQS = np.arange(8.0, 14.0, 1.0)
N_CYCLES = FREQS / 2.0          # comme le script 11
MAIN, BASE = (0.3, 1.75), (-2.0, -1.5)
EXCLUDED = ["DN12"]

rows = []
for f in sorted(glob.glob(os.path.join(EPOCH_DIR, "*-tfr-epo.fif"))):
    s = os.path.basename(f).replace("-tfr-epo.fif", "")
    if any(x in s for x in EXCLUDED):
        continue
    ep0 = mne.read_epochs(f, preload=True, verbose=False).pick("eeg")
    versions = {"reference moyenne": ep0}
    try:
        versions["CSD"] = mne.preprocessing.compute_current_source_density(ep0, verbose=False)
    except Exception as e:
        print(f"{s} : CSD impossible ({e})")
    for vname, ep in versions.items():
        picks = {r: [c for c in chs if c in ep.ch_names] for r, chs in ROIS.items()}
        pw = {}
        for cond, vis in CONDITIONS.items():
            e = ep[[v for v in vis if v in ep.event_id]]
            allpicks = sorted({c for v in picks.values() for c in v})
            tfr = e.compute_tfr("morlet", freqs=FREQS, n_cycles=N_CYCLES, picks=allpicks,
                                return_itc=False, average=True, decim=2, verbose=False)
            pw[cond] = tfr
        t = pw["REAL_APP"].times
        bmask = (t >= BASE[0]) & (t <= BASE[1])
        mmask = (t >= MAIN[0]) & (t <= MAIN[1])
        for roi, chs in picks.items():
            idx = [pw["REAL_APP"].ch_names.index(c) for c in chs]
            P = {c: pw[c].data[idx].mean(axis=(0, 1)) for c in CONDITIONS}   # temps
            base = np.mean([P[c][bmask].mean() for c in CONDITIONS])
            db = {c: 10 * np.log10(P[c][mmask].mean() / base) for c in CONDITIONS}
            rows.append({"subject": s, "version": vname, "roi": roi,
                         "REAL": (db["REAL_APP"] + db["REAL_DISG"]) / 2,
                         "VIRT": (db["VIRT_APP"] + db["VIRT_DISG"]) / 2})
    print(s, "ok")

df = pd.DataFrame(rows)
df.to_csv(os.path.join("outputs_tfr", "mu_with_without_csd.csv"), index=False)
print("\nEffet du reel sur mu (0.3-1.75 s), REAL - VIRT :")
for (v, roi), d in df.groupby(["version", "roi"]):
    diff = d.REAL - d.VIRT
    t, p = stats.ttest_1samp(diff, 0)
    print(f"  {v:18s} {roi:15s} REAL {d.REAL.mean():+.2f} | VIRT {d.VIRT.mean():+.2f} dB | "
          f"t({len(d)-1}) = {t:.2f}, p = {p:.4f}, dz = {diff.mean()/diff.std(ddof=1):.2f}")
