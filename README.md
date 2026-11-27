# RF dosimetry threshold reproduction package

Reproduction package for the accompanying article, provided with it as Supplementary Data 1 and publicly available at https://github.com/aalbata/RF_Dosimetry.

The package converts Monte Carlo samples of a dosimetric endpoint, either the peak 10 g specific absorption rate (SAR) or the peak steady-state temperature rise, into incident power density thresholds from five bounds: the conformal quantile, which equals the empirical quantile at the sample size of the main results; a nonparametric tolerance limit; the conditional value-at-risk (CVaR); a Cantelli moment bound; and a Wasserstein CVaR bound. It reports bootstrap intervals for the thresholds and the critical Wasserstein radius at which a comparison with a reference level can change. It contains the forward models (transmission-line planar slab, multilayer Mie sphere, steady-state and transient Pennes bioheat solutions), two tissue-property uncertainty models with four input-model variants, the empirical tests, the verification checks including an independent finite-difference time-domain (FDTD) solution, and it regenerates every table and figure of the article and the computed values quoted in its text. The same threshold calculation can be applied to endpoint samples computed with any other solver (see "Thresholds for your own samples").

## Contents

| Path | Description |
| --- | --- |
| `reproduce.py` | Complete implementation: forward models, uncertainty models, bounds, experiments, verification, figures and tables |
| `requirements.txt` | Exact package versions used for the article |
| `data/samples/` | Endpoint samples (75 NumPy `.npz` files) and deterministic studies (7 JSON files) |
| `results/` | Outputs produced by `python reproduce.py` from the shipped samples |
| `LICENSE` | MIT License for the code |

## Environment

The article results were computed with Python 3.12.10 on Windows 11 (Intel Core Ultra 7 155U) with the packages pinned in `requirements.txt` (NumPy 2.5.2, SciPy 1.18.1, Matplotlib 3.11.1, Pillow 12.3.0, scattnlay 2.4). The computation runs on the CPU only, and no data leave the machine.

## Installation

```
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python reproduce.py --check
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`. scattnlay provides binary wheels for common platforms; where no wheel is available, a C++ compiler is required to build it.

## Usage

| Command | Purpose | Run time on the machine above |
| --- | --- | --- |
| `python reproduce.py --check` | Check the installed packages without simulation | seconds |
| `python reproduce.py` | Regenerate all results, tables, figures and computed in-text values from `data/samples` | less than 2 min |
| `python reproduce.py --simulate --jobs 6` | Regenerate the endpoint samples and deterministic studies from scratch in six processes (written to `results/samples`), then all results | about 6 h |
| `python reproduce.py --fast --jobs 2 --outdir quick` | Reduced-size check of the complete pipeline | about 18 min |

All random streams derive from the seed 20260711 (option `--seed`). The property draws start from the seed, and every input-model variant reuses the copula scores of the main analysis; the factor shared by all tissues uses the seed plus 13. The coverage splits, calibration subsets, shift resamples, threshold bootstrap and sampling-distance bootstrap start from the seed, the seed plus 7, 11, 17 and 19. With `--simulate`, the regenerated samples equal the shipped samples whatever the number of processes.

## Thresholds for your own samples

```
python reproduce.py --samples my_sar.csv --anchor 50 --limit 2 --levels 40,10 --nominal 1.59 --report my_report.json
```

`my_sar.csv` holds one endpoint value per line (or in the first comma-separated column; a header line is skipped), for example the peak 10 g SAR in W kg^-1 of each Monte Carlo draw, all computed at the same source strength `--anchor` (incident power density in W m^-2, or another quantity such as antenna input power in W). The endpoint must scale linearly with the source strength. `--limit` is the restriction on the endpoint, `--levels` lists comparison levels in the units of the source strength, and `--nominal` (optional) is the endpoint at nominal properties, used to express critical radii as a percentage. The command prints the thresholds of the quantile, conformal, tolerance, CVaR and Cantelli bounds, with 95% bootstrap intervals for all except the conformal threshold, the threshold at nominal properties when `--nominal` is given, and the critical Wasserstein radius for each level, and `--report` writes them to a JSON file. The bootstrap uses the same random stream as the article (seed plus 17). NumPy `.npy` and `.npz` files are also accepted.

## Outputs

| Path | Content |
| --- | --- |
| `results/results/<model>/thresholds.csv` | Endpoint bounds, thresholds with 95% bootstrap intervals, nominal-property threshold, critical Wasserstein radii and sampling distance for each geometry and frequency |
| `results/results/<model>/coverage.csv` | Mean, median and standard error of the held-out exceedance of Gaussian plug-in and conformal quantiles over 200 random splits |
| `results/results/<model>/calibration_size.csv` | Mean threshold and mean held-out exceedance of the empirical quantile, conformal quantile and tolerance limit versus calibration size (25 to 1600 draws, 120 subsets each); tolerance entries are nan where the limit does not exist |
| `results/results/<model>/shift_test.csv` | Distribution-shift experiment, including the share of the worst-case CVaR increase |
| `results/results/<model>/input_variants.csv` | Thresholds of the four input-model variants, their change relative to the main analysis, the exceedance of the main conformal threshold and the Wasserstein distance to the main sample |
| `results/results/<model>/wasserstein_check.json` | Closed-form worst-case CVaR compared with the explicit transport construction for the 0.9 GHz slab sample |
| `results/results/<model>/temperature_thresholds.csv` | Temperature bounds, thresholds with bootstrap intervals, SAR-temperature rank correlation and rank correlations of the sampled thermal inputs, at 0.9, 2.45 and 5.8 GHz |
| `results/results/grid_averaging.csv` | Nominal sphere peak 10 g SAR on nine grids for the averaging rule of the article and its variants |
| `results/results/grid_subset.csv` | Draw-by-draw effect of the 1 mm grid on the first 150 reference-model sphere draws |
| `results/results/geometry.csv`, `perfusion.csv`, `transient.csv` | Geometry variations, perfusion settings and step response at nominal properties |
| `results/results/fdtd_check.json`, `verification.json` | FDTD check, solver verification quantities and run configuration |
| `results/results/model_distance.csv` | Wasserstein distance between the endpoint samples of the two uncertainty models |
| `results/tables/` | Article tables as displayed (rounded), one CSV file per table |
| `results/figures/` | Article figures as 300 dpi TIFF files and vector PDF files |
| `results/numbers.json` | Computed result values quoted in the article text |

`<model>` is `reference` (wide dielectric uncertainty model) or `within_tissue` (within-tissue model with permittivity and conductivity multipliers between 1/1.10 and 1.10, log half-width ln 1.10).

## Results

Incident power density thresholds (W m^-2) at a target exceedance probability of 0.05 from 2000 draws per case.

| Model | Frequency (GHz) | Geometry | Nominal | Quantile | Tolerance | CVaR | Cantelli |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Reference | 0.9 | Slab | 158.6 | 99.6 | 98.5 | 94.4 | 65.1 |
| Reference | 0.9 | Sphere | 41.6 | 27.3 | 26.4 | 24.1 | 18.2 |
| Reference | 2.45 | Slab | 62.9 | 54.4 | 54.1 | 52.7 | 37.5 |
| Reference | 2.45 | Sphere | 26.2 | 19.5 | 19.2 | 18.0 | 13.4 |
| Reference | 5.8 | Slab | 162.5 | 80.4 | 78.9 | 75.5 | 49.9 |
| Reference | 5.8 | Sphere | 107.6 | 52.9 | 52.5 | 50.9 | 32.3 |
| Within-tissue | 0.9 | Slab | 158.6 | 148.2 | 147.8 | 146.2 | 133.4 |
| Within-tissue | 0.9 | Sphere | 41.6 | 36.2 | 36.1 | 35.7 | 29.7 |
| Within-tissue | 2.45 | Slab | 62.9 | 59.9 | 59.8 | 59.4 | 55.0 |
| Within-tissue | 2.45 | Sphere | 26.2 | 24.8 | 24.7 | 24.5 | 22.3 |
| Within-tissue | 5.8 | Slab | 162.5 | 148.3 | 147.9 | 146.9 | 129.1 |
| Within-tissue | 5.8 | Sphere | 107.6 | 99.8 | 99.5 | 98.5 | 86.0 |

Column definitions. Model is the dielectric uncertainty model. Frequency is the plane-wave frequency in GHz. Geometry is the four-layer planar slab or the 30 mm layered sphere. Nominal is the threshold at nominal tissue properties. Quantile is the empirical and conformal quantile threshold (identical at 2000 draws). Tolerance is the nonparametric tolerance threshold with 95% content and 95% confidence. CVaR is the conditional value-at-risk threshold. Cantelli is the threshold from the Cantelli bound on the sample mean and standard deviation. The thresholds follow from the general-public local SAR restriction of 2 W kg^-1 (10 g, 6 min) in the guidelines of the International Commission on Non-Ionizing Radiation Protection (ICNIRP), Health Phys. 118 (2020) 483-524, doi:10.1097/HP.0000000000001210; the corresponding local reference levels for incident power density in the same guidelines are 20.1, 40.0 and 40.0 W m^-2 and the whole-body reference levels 4.5, 10.0 and 10.0 W m^-2 at 0.9, 2.45 and 5.8 GHz. The thresholds describe the modeled geometries, plane-wave illumination and input distributions; they are not exposure limits.

## Data sources

Nominal dielectric spectra use the four-term Cole-Cole parameters of Gabriel, Lau and Gabriel, Phys. Med. Biol. 41 (1996) 2271-2293, doi:10.1088/0031-9155/41/11/003. Density, thermal property and heat capacity values and ranges are taken from the IT'IS database for thermal and electromagnetic parameters of biological tissues, version 5.0 (IT'IS Foundation, 2025, doi:10.13099/VIP21000-05-0); the skin layer uses the database row for skin, and the fat layer takes its perfusion and metabolic heat ranges from the row for subcutaneous fat, because the row for non-infiltrated fat lists zero for both. Only the numerical values used by the model are stored in `reproduce.py`, and the database itself is not redistributed. The multilayer Mie fields are computed with scattnlay (Peña and Pal, Comput. Phys. Commun. 180 (2009) 2348-2354, doi:10.1016/j.cpc.2009.07.010; Ladutenko, Pal, Rivera and Peña-Rodríguez, Comput. Phys. Commun. 214 (2017) 225-230, doi:10.1016/j.cpc.2017.01.017).

## Citation

If you use this package, please cite the accompanying article: Al Bataineh A, AlZaabi F, Alkaabi N. Radiofrequency Exposure Thresholds from Statistical Upper Bounds under Uncertain Tissue Properties (2026). Citation metadata are given in `CITATION.cff`.

## License

The code is released under the MIT License (see `LICENSE`). The license covers the code and the generated samples and results; it does not extend to the third-party data sources listed above.
