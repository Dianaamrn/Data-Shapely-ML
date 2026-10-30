# Data Shapley

## Assignment 9 - Statistical Methods for Machine Learning Course University of Milan

Based on Amirata Ghorbani and James Zou, [Data Shapley: Equitable Valuation of Data for Machine Learning](https://proceedings.mlr.press/v97/ghorbani19c.html), ICML 2019.

The project estimates how much each training point is worth to a classifier, using Data Shapley with Monte Carlo permutation sampling. It then checks whether removing the lowest-value points improves test accuracy compared with removing points at random. Everything is written from scratch with NumPy and Matplotlib only. (as the assignment requested it)

The full report is [report/Report.pdf](report/Report.pdf).

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

- On this specific clean dataset, removing low-value points does no better than random removal.
- With label noise, it clearly helps at every removal fraction.
- Removing high-value points lowers accuracy in every condition.
- Low values find the flipped labels well: average AUROC is 0.993 for noisy Penguins and 0.851 for noisy Gaussian.


## How to run

You need Python 3.12 or newer; the results were made with Python 3.13.

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

**3. Recompute everything from scratch (optional runtime roughly about 1 hour)**

```
python run_pipeline.py --fresh
```

**4. Open the notebooks (optional)**

```
pip install -e ".[notebooks]"
cd notebooks
jupyter notebook
```

Run them from the `notebooks/` folder. The appendix takes about 15 to 20 minutes; the others are quick.

## Data

Palmer Penguins data by Kristen B. Gorman and Palmer Station Antarctica LTER, CC0. Horst, Hill and Gorman (2020), [doi:10.5281/zenodo.3960218](https://doi.org/10.5281/zenodo.3960218). Gorman, Williams and Fraser (2014), [doi:10.1371/journal.pone.0090081](https://doi.org/10.1371/journal.pone.0090081).
