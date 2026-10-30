# Data Shapley

## Assignment 9 - Statistical Methods for Machine Learning Course University of Milan

Based on Amirata Ghorbani and James Zou, [Data Shapley: Equitable Valuation of Data for Machine Learning](https://proceedings.mlr.press/v97/ghorbani19c.html), ICML 2019.

The project estimates how much each training point is worth to a classifier, using Data Shapley with Monte Carlo permutation sampling. It then checks whether removing the lowest-value points improves test accuracy compared with removing points at random. Everything is written from scratch with NumPy and Matplotlib only.

The full report is [report/main.pdf](report/main.pdf).

## What is in it

- **Datasets:** Palmer Penguins (species from four measurements) and a two-class Gaussian dataset. Three conditions: clean Penguins, Penguins with 20% of training labels flipped, and Gaussian data with 20% flipped.
- **Classifier:** our own 3-nearest-neighbour, with features scaled using the training data only.
- **Shapley values:** complete random permutations, no truncation. The utility is 3-NN validation accuracy. Each run uses at least 2000 permutations, chosen by a pilot on validation data, then continues until the convergence rule from the paper is met. The rule is that the average relative change over the last 100 permutations must be below 0.05.
- **Removal:** remove 10, 20, 30 or 40% of the training points with the lowest values (LOW), the highest values (HIGH) or at random (RANDOM, 30 draws). Then retrain and score on the test set.
- **Repeats:** ten seeds (100 to 109), with bootstrap intervals over runs.

## Main results

Test accuracy change at 20% removal, LOW minus RANDOM, in percentage points:

| Condition | LOW minus RANDOM | 95% interval | LOW wins |
|---|---:|---:|---:|
| Clean Penguins | -0.04 | [-0.39, +0.27] | 6 of 10 |
| Noisy Penguins | +8.51 | [+7.05, +9.96] | 10 of 10 |
| Noisy Gaussian | +8.88 | [+7.87, +9.96] | 10 of 10 |

- On clean data, removing low-value points does no better than random removal.
- With label noise, it clearly helps at every removal fraction.
- Removing high-value points lowers accuracy in every condition.
- Low values find the flipped labels well: average AUROC is 0.993 for noisy Penguins and 0.851 for noisy Gaussian.

The full tables and figures are in [notebooks/results.ipynb](notebooks/results.ipynb).

## How to run

You need Python 3.11 or newer; the results were made with Python 3.13. No GPU and no internet are needed after download.

**1. Get the project and install it**

```
git clone https://github.com/Dianaamrn/Data-Shapely-ML.git
cd Data-Shapely-ML
python -m venv .venv
.venv\Scripts\activate          (Windows)
source .venv/bin/activate       (Mac or Linux)
pip install -e .
```

This installs the exact NumPy and Matplotlib versions used for the results.

**2. Check the saved results (about 3 minutes)**

```
python run_checks.py
```

This runs the unit tests and then rebuilds every saved result from the seeds and compares it with what is stored. It should end with `All checks passed.`

**3. Recompute everything from scratch (optional, about 1 hour)**

```
python run_pipeline.py --fresh
```

This empties `results/` and runs every step again: pilot, ten seeds, summary tables, notebooks and checks. The numbers come out identical. Add `--workers 3` to run three seeds at a time. Without `--fresh`, finished steps are skipped, so an interrupted run can be resumed.

**4. Open the notebooks (optional)**

```
pip install -e ".[notebooks]"
cd notebooks
jupyter notebook
```

Run them from the `notebooks/` folder. The appendix takes about 15 to 20 minutes; the others are quick.

## What is in each folder

```
data/raw/
  penguins.csv                  the Palmer Penguins data, unchanged
  penguins.provenance.json      where it was downloaded from, its fingerprint and licence

data_shapley/                   all the project code
  data.py                       loading, cleaning, splitting, scaling, synthetic data, label noise
  knn.py                        our 3-nearest-neighbour classifier
  shapley.py                    exact Shapley values (for checks) and the permutation estimators
  metrics.py                    accuracy, macro-F1 and ranking comparisons
  experiments.py                LOW, HIGH and RANDOM removal and flipped-label detection
  pilot.py                      step 1: picks the minimum number of permutations
  study.py                      step 2: Shapley values, removal and detection for each seed
  study_summary.py              step 3: combines the ten seeds into summary tables
  provenance.py                 file fingerprints that tie results to the code

scripts/
  verify_full_study.py          rebuilds every saved study result from the seeds
  verify_study_summaries.py     recomputes the summary tables a second way

tests/                          unit tests for each code file

notebooks/
  data_preparation.ipynb        how the data is built
  algorithm_checks.ipynb        k-NN and Shapley checks, fast evaluator, pilot
  results.ipynb                 ten-seed results and all figures
  appendix.ipynb                extra checks

results/
  pilot/                        the pilot's record and samples
  study/seed100 ... seed109/    each seed's Shapley values, permutations, removal results and settings
  study/study_*.csv             the ten seeds combined (the tables in the report)
  study/study_*.json            the study settings and the verification records
  figures/                      the five report figures

report/                         the report (main.pdf) and its LaTeX source
run_pipeline.py                 runs the whole project
run_checks.py                   runs the tests and all checks
pyproject.toml                  install settings and pinned versions
```

## Data

Palmer Penguins data by Kristen B. Gorman and Palmer Station Antarctica LTER, CC0. Horst, Hill and Gorman (2020), [doi:10.5281/zenodo.3960218](https://doi.org/10.5281/zenodo.3960218). Gorman, Williams and Fraser (2014), [doi:10.1371/journal.pone.0090081](https://doi.org/10.1371/journal.pone.0090081).
