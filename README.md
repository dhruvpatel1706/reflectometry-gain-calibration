# Absolute reflectance calibration and the separation of film index from thickness in spectral reflectometry

Code and data for the manuscript of that title by Dhruv Patel, Independent Researcher (ORCID 0009-0009-3375-4646). The manuscript
itself is not included. The scripts generate every simulated spectrum, run every analysis, and write every number
(`paper/mst/numbers.tex`, one LaTeX macro per quoted value) and every figure (`paper/mst/figures/`) of the manuscript.
The measured spectra of section 9 come from a public dataset (CC BY 4.0); an unchanged copy of the one file used is
included under `data/external_2026-09-29/`, with its SHA-256 in `MANIFEST.tsv` and its attribution in `ATTRIBUTION.md`.
Nothing is downloaded. Licence of the code: MIT (see `LICENSE`).

## Requirements

Python 3.14 with numpy, scipy, matplotlib and openpyxl at the versions in `requirements.txt` (tested: Python 3.14.6,
numpy 2.4.6, scipy 1.18.1, matplotlib 3.11.2, openpyxl 3.1.5). Everything runs on one CPU core; no GPU.

## Reproduce everything

```
bash run_all.sh                      # or: PYTHON=/path/to/python3 bash run_all.sh
```

This runs steps 1 to 11 below in order and then checks every shipped file against `CHECKSUMS.sha256`; the
regenerated outputs, including `paper/mst/numbers.tex` and the six figures, should all be reported OK. Times are
wall-clock times of a clean run on an Apple M4 Pro laptop.

| step | script | reads | writes | time |
|---|---|---|---|---|
| 1 | `code/rev_structural.py` | forward model only | `data/results/rev_structural.json` | 2 min 47 s |
| 2 | `code/rev_matched.py` | forward model only | `rev_matched.json`, `rev_matched_trials.npz` | 5 min 47 s |
| 3 | `code/rev_sweep.py` | forward model only | `rev_sweep.json`, `rev_sweep_trials.npz` | 1 min 47 s |
| 4 | `code/rev_fisher.py` | `rev_matched_trials.npz`, `rev_structural.json` | `rev_fisher.json` | 1 s |
| 5 | `code/rev_lownoise.py` | `rev_matched_trials.npz` | `rev_lownoise.json` | 3 s |
| 6 | `code/rev_modulation.py` | `rev_matched_trials.npz` | `rev_modulation.json` | under 1 s |
| 7 | `code/rev_dispersion.py` | forward model only | `rev_dispersion.json`, `rev_dispersion_trials.npz` | 4 min 44 s |
| 8 | `code/twin_dispersion.py` | forward model only | `twin_dispersion.json` | 28 s |
| 9 | `code/real_data_check.py` | `data/external_2026-09-29/zenodo_15342753/Data_Figure_7.xlsx` (checked against `MANIFEST.tsv`) | `real_data_check.json` | 45 s |
| 10 | `code/rev_figures.py` | `rev_structural.json`, `rev_fisher.json`, `rev_sweep.json`, `real_data_check.json` | `paper/mst/figures/figure1.pdf` to `figure6.pdf` | 2 s |
| 11 | `code/make_mst_numbers.py` | all seven `rev_*.json`, `twin_dispersion.json` and `real_data_check.json` | `paper/mst/numbers.tex` | under 1 s |

`run_all.sh` also runs `code/test_real_data_backside.py` after step 9 (under 1 s, no output file). It checks the
backside sum of `real_data_check.py`, R_f + T_f R_b T_r / (1 - R_r R_b) with the film's power coefficients seen from
air and from the substrate, against an independent characteristic-matrix computation, against the coherent reflectance
of a finite glass layer averaged over its phase, and for energy conservation, reciprocity, the lossless limit and
zero-thickness and index-matched limits, for clear and absorbing films; it exits non-zero on any failure.

`code/optics.py` (Airy reflectance) and `code/fastgrid.py` (grid likelihoods for the known, profiled and
marginalized gain) are libraries used by the scripts. Every random draw is seeded; the seeds and the design of each
experiment are stored in the `config` block of the corresponding JSON file.

## Where each result lives

| manuscript | file |
|---|---|
| Section 5 (exact structure: twin symmetry, noiseless profiles, index matching), figure 2 | `rev_structural.json` |
| Section 5.3 (the twin with a dispersive film; the phase of r and the ellipsometric angles of a film and its twin) | `twin_dispersion.json` |
| Section 6 (amplitude and phase channels, closed-form cost, numerical checks, gain-error tolerance), figures 3 and 4 | `rev_fisher.json` (figure 4 also uses `rev_structural.json`) |
| Sections 7.1 to 7.3 (E1, E2, E3, E3b, grid convergence), tables 1 and 3 | `rev_matched.json`, per-trial values in `rev_matched_trials.npz` |
| Table 2 (E1 films by true index: local bound against Monte Carlo) | `rev_fisher.json` |
| Section 7.1 (errors by fringe modulation) | `rev_modulation.json` |
| Section 7.4 (thickness, noise and substrate sweep), figure 5 | `rev_sweep.json`, `rev_sweep_trials.npz` |
| Section 7.5 (low noise: direct least-squares fits), table 4 | `rev_lownoise.json` |
| Section 8.1 (dispersion, D1 and D2) | `rev_dispersion.json`, `rev_dispersion_trials.npz` |
| Section 8.2 (offsets and wavelength-dependent gains) | `rev_structural.json` |
| Section 9 (measured spectra of a silicon nitride film), table 5, figure 6 | `real_data_check.json` |
| Figure 1 (measurement chain) | drawn by `rev_figures.py`, no data |

Two JSON keys keep names from an earlier version of the analysis: `fisher_theory_steps_max_rel_change_in_sd_ratio`
(in `rev_fisher.json`) is the derivative check with fixed steps of 1e-6 in index and 1e-4 nm in thickness, and
`reviewer_example_ns2p30` (in `rev_structural.json`) is the film with n = 1.40, d = 700 nm on a substrate of index
2.30 shown in figure 2.

## The number guard

The manuscript types no result by hand: every quoted value is a macro of `paper/mst/numbers.tex`. Its number guard,
which checks the manuscript text against the result files, ships with the manuscript.

## Verification

On 29 September 2026 the complete pipeline was rerun from a copy of this code in an empty directory (15 minutes
in all): the six JSON result files of steps 1 to 4, 6 and 7 and figures 1 to 5 were identical byte for byte to the
files shipped here, and all 719 arrays in the three `.npz` files exactly. The scripts of those steps and the two
libraries are unchanged since then apart from one reworded docstring in `rev_modulation.py`; `rev_figures.py` now
also draws figure 6 and writes figures 1 to 5 byte for byte as before. On 9 October 2026
this repository was copied into an empty directory, where `CHECKSUMS.sha256` verified and the seven fast steps
(`rev_fisher.py`, `rev_lownoise.py`, `rev_modulation.py`, `twin_dispersion.py`, `real_data_check.py`,
`rev_figures.py`, `make_mst_numbers.py`) reproduced the shipped outputs byte for byte, including `numbers.tex`
and the six figures, with `test_real_data_backside.py` passing.

## Measured spectra: sources and records not used

The measured spectra are sheet `7a` of `Data_Figure_7.xlsx` in the dataset of Schwarz, Dick, Beuer, Rommel and Hutzler
(Zenodo, doi:10.5281/zenodo.15342753, CC BY 4.0), which supplements their article in Micron 198 (2025) 103878. The
film thickness (536.4 nm, by HAADF-STEM and EDX of a lamella prepared by focused ion beam in the Micron study), the
numerical apertures of the four objectives, the silicon reference and which objectives collect the reflection from the
back of the wafer are stated in Schwarz, Bauer, Rommel and Hutzler, arXiv:2607.03077 (CC BY 4.0), which uses the same
samples and set-up. The sheet also holds the data authors'
simulated spectra; `real_data_check.py` fits them to confirm which spectra belong to the single film. The substrate
index is the BOROFLOAT 33 reference values of the SCHOTT data sheet "BOROFLOAT 33 Optical Properties" (2021), quoted
in the script; the data sheet itself is not redistributed.

### Public records examined and not used

1. Sanchez-Arriaga, Tiwari, Hutabarat, Leyland and Tiwari, dataset doi:10.15131/shef.data.23285603 (CC BY 4.0),
   supplementing Sensors 23 (2023) 5326: SiO2 films on silicon measured with low-cost sensors from 450 to 700 nm.
   The root-mean-square difference between each measured spectrum and the data authors' own model (their
   "Math Model Reflectance" and "Test" columns) is 0.02 to 0.09 in reflectance, and the reference thicknesses
   (476.3 and 198.7 nm) come from the measurement report of the commercial standard, a spectral fit from 220 to
   1000 nm with tabulated optical constants (appendix of the article), not from an independent method.
2. Schwarz et al., dataset doi:10.5281/zenodo.13753754 (CC BY 4.0), supplementing Adv. Opt. Mater. 13 (2025)
   2402502: a thin thermal SiO2 film on silicon. The dataset gives the spectra but not the reference thickness or the
   objective used; both are stated in the article, whose publisher page could not be read for this work, so neither
   could be checked at its source.

## Files

`CHECKSUMS.sha256` lists the SHA-256 of every file in the package. Check it with `shasum -a 256 -c CHECKSUMS.sha256`.
