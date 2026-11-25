[README(2).md](https://github.com/user-attachments/files/35435247/README.2.md)
# LabIsCoat# Beyond pictures: EEG analysis code

Analysis code for the article:

> **Beyond pictures: distinct electrophysiological signatures of physical reality and emotional value in food processing.**
> Gabriel D., et al. (submitted to *NeuroImage*).

The study crossed **physical reality** (real food vs. a printed photograph of the same exemplar) with **emotional value** (appetizing vs. disgusting) in a within-participant design. Stimuli were presented with the Lab-Life device (Gabriel et al., 2026, *Journal of Neuroscience Methods*, 429, 110692, https://doi.org/10.1016/j.jneumeth.2026.110692). High-density EEG (256-channel HydroCel Geodesic Sensor Net, 1000 Hz) was recorded in 28 participants.

This repository contains the Python scripts used for preprocessing, statistical analysis and figures. **It contains no data.** The EEG recordings and questionnaire data are not publicly available because of participant privacy regulations (see *Data availability* in the article).

---

## Requirements

- Python 3.10
- The packages listed in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The source-localization scripts (`18_*`) download the FreeSurfer `fsaverage` template the first time they run (about 200 MB, through `mne.datasets.fetch_fsaverage`). The 3D brain figures require `pyvista` and `pyvistaqt`.

The preprocessing scripts open interactive MNE windows for visual inspection of channels and trials, which require a Qt backend (`PyQt5`).

---

## Repository structure and order of execution

Scripts are meant to be run in the order below. Each script states its inputs and outputs in its header. Paths to the data are set at the top of each preprocessing script and in `config.py`, and must be adapted to your own folder structure.

### 1. Preprocessing (one participant at a time)

| Order | Script | Step | Article section |
|---|---|---|---|
| 1 | `Reduction-mffA204elec_concat.py` | Reads the `.mff` recordings, keeps 204 channels (203 electrodes + vertex reference), concatenates the recording blocks | 2.6 |
| 2 | `FiltrageDonnees_LabIsCoat.py` | Band-pass 0.5–40 Hz, 50 Hz notch, visual inspection and spherical-spline interpolation of bad channels, average reference, ICA (FastICA, 64 components, seed 42) with automatic removal of ocular and muscle components | 2.6 |
| 3 | `CorrectionArtefactSingleTrial.py` | Epochs from −0.1 to 0.6 s (baseline −0.1 to 0 s); removal of the screen-switching artifact by subtracting its first principal component, estimated around screen-closing events (−30 to +50 ms) | 2.6, S4 |
| 4 | `LabIsCoat-epochs.py` | Trial rejection (visual inspection and ±100 µV peak-to-peak criterion) | 2.7 |
| 5 | `LissagePicPCA50ms.py` | Savitzky–Golay smoothing (third order, 20-ms window) of residual activity between 30 and 80 ms, with 10-ms cosine cross-fades | S4 |

The output of step 5 (one `*-epo.fif` file per participant) is the input of the evoked-activity analyses (`EPOCHS_DIR` in `config.py`).

### 2. Shared modules

| File | Content |
|---|---|
| `config.py` | Paths, conditions, analysis parameters, homogeneous loading of the epoch files |
| `microstate_utils.py` | Polarity-sensitive microstate functions: k-means clustering, back-fitting, temporal smoothing (Pascual-Marqui et al., 1995), parameters |

### 3. Quality control

| Script | Content |
|---|---|
| `00_check_channels.py` | Channels available for each participant |
| `00b_count_trials.py` | Number of trials retained per participant and condition |
| `00c_check_baseline.py` | Effect of an alternative evoked baseline (−100 to −30 ms) on the results (S4) |

### 4. Evoked activity: microstates, GFP and TANOVA (sections 2.7 and 3.2)

| Script | Content | Output |
|---|---|---|
| `01_extract_gfp_peaks.py` | Maps at the GFP peaks of individual ERPs (80–600 ms) | `outputs_microstates/` |
| `02_cluster_microstates.py` | Polarity-sensitive k-means clustering, Krzanowski–Lai criterion | `microstate_model.npz` |
| `03_backfit_segmentation_parameters.py` | Back-fitting to individual ERPs, microstate parameters and transitions, grand-average segmentation (defines the fixed GFP windows) | `microstate_parameters.csv` |
| `03b_gev_table.py` | Global explained variance of each template (Table 1) | |
| `04_microstate_statistics.py` | Linear mixed-effects models (Reality × Valence) on microstate parameters, GEE on transitions (S3), FDR correction | |
| `07_fixed_window_gfp_analysis.py` | GFP peak and mean GFP in the grand-average windows | `microstate_parameters_fixed_window.csv` |
| `13_tanova.py` | Topographic ANOVA with duration correction (Koenig & Melie-García, 2010) | `outputs_microstates/tanova/` |
| `05b_figure_microstates_gfp.py` | **Figure 5** | `outputs_microstates/publication_figures/` |

### 5. Time-frequency analysis (sections 2.8 and 3.3)

| Script | Content | Output |
|---|---|---|
| `11a_timeline_marqueurs.py` | Timing of the button press, screen opening and screen closing in each trial (Figure 2) | |
| `11_tfr_extract.py` | Epochs from −2.5 to +2.5 s, screen-switching artifact correction at opening and closing (Tukey-tapered), rejection at 150 µV, current source density, Morlet wavelet power in dB relative to the common baseline (−2.0 to −1.5 s) | `outputs_tfr/tfr/*_tfr.npz`, `tfr_roi_band_windows.csv` |
| `11b_save_tfr_epochs.py` | Cleaned time-frequency epochs (without CSD), used for source localization | `outputs_tfr/epochs/` |
| `12_tfr_stats.py` | Planned paired tests (Bonferroni), cluster-based permutation tests on time × frequency maps and on channels × time | |
| `12b_tfr_anova.py` | Repeated-measures ANOVA (Reality × Valence × Hemisphere) for each region and band (Table 2) | |
| `12c_figure_tfr.py` | **Figure 7** | `outputs_tfr/figures/` |

### 6. Pre-stimulus sensorimotor activity (Supplementary Material S1)

| Script | Content |
|---|---|
| `17_prestim_lateralization_mu.py` | Pre-stimulus low-beta lateralization (−0.9 to −0.3 s) and its correlation with the Reality effect on μ power |
| `17b_prestim_after_press.py` | Control analysis in a window fully after the button press (−0.55 to −0.3 s) |
| `17c_figure_S1.py` | **Figure S1** |

### 7. Bayesian and equivalence analyses (sections 2.9 and 3.4)

| Script | Content | Output |
|---|---|---|
| `15_bayes_interaction.py` | Bayesian one-sample t-tests (JZS prior, r = √2/2; robustness with r = 0.5 and r = 1) and TOST equivalence tests (bounds dz = ±0.5) | `outputs_bayes/` |

### 8. Brain–behavior correlations and trait food craving (sections 2.10 and 3.5)

| Script | Content | Output |
|---|---|---|
| `14_correlations_cerveau_comportement.py` | Planned Spearman correlations between neural and behavioral indices, bootstrap confidence intervals, Bayes factors (Jeffreys), Steiger tests; **Figure S6** | `outputs_correlations/` |
| `16_figure_fcq_mu.py` | Trait food craving (FCQ-T-r) and the Reality effect on right central μ power; **Figure 9** | `outputs_correlations/` |

### 9. Source localization (sections 2.11 and 3.2–3.3)

| Script | Content | Output |
|---|---|---|
| `18_source_localization.py` | eLORETA on the fsaverage template; spatial cluster-based permutation tests for the Valence effect (evoked amplitude) and the Reality effect (μ power); a priori precentral + postcentral regions of interest | `outputs_source/` |
| `18b_figure_sources.py` | **Figure 6** | `outputs_source/` |
| `18c_mu_without_csd.py` | Reality effect on μ power in average-referenced data, without the surface Laplacian | |

Figures 2, 3, 4 and 8 were drawn directly from the values reported in the article and are not produced by these scripts.

---

## Notes

- **Polarity.** Microstate clustering and back-fitting take map polarity into account, which is appropriate for event-related potentials. Because pycrostates (Férat et al., 2022) ignores polarity, these steps are implemented in `microstate_utils.py`, which reproduces pycrostates when polarity is ignored.
- **Random seeds.** ICA, k-means, bootstrap and permutation procedures use fixed seeds (42 unless stated otherwise in the script header), so that results are reproducible.
- **Excluded participants.** Participants excluded from the EEG analyses are listed in `config.py` and in `EXCLUDED_INCLUSIONS` in the scripts concerned; see section 2.1 of the article.
- **Behavioral analyses.** Wilcoxon tests on desire-to-eat and emotional-intensity ratings (section 2.5) were performed with [to be completed: software].
- **Language.** Comments and console messages in the scripts are partly in French.

---

## Citation

If you use this code, please cite the article above and:

Gabriel, D., Bertrand, G., Nicolier, M., & Giustiniani, J. (2026). Neuroimaging of reality: A new approach for investigating neural bases of decision-making with real-world objects. *Journal of Neuroscience Methods*, 429, 110692. https://doi.org/10.1016/j.jneumeth.2026.110692


## Contact

Damien Gabriel, CIC-IT Inserm 1431, CHU de Besançon, Neuraxess, and UMR Inserm 1322 LINC, Université Marie et Louis Pasteur, Besançon, France.

## Requirements

# Python 3.10
# Versions used for the analyses reported in the article (section 2.6)
mne==1.11.*
numpy==2.2.*
scipy==1.15.*
pandas==2.3.*
scikit-learn==1.7.*
statsmodels==0.14.*

# Figures
matplotlib>=3.9
seaborn>=0.13

# Reading the questionnaire files (.xlsx)
openpyxl>=3.1

# Interactive inspection windows (preprocessing)
PyQt5>=5.15
mne-qt-browser>=0.6

# Source localization (scripts 18_*): template MRI and 3D rendering
nibabel>=5.2
pyvista>=0.43
pyvistaqt>=0.11
vtk>=9.3

