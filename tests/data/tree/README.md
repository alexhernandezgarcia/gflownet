# Classification / regression datasets for the tree environments

Every dataset lives in its own directory and follows the layout that
`gflownet.envs.tree.tree.Tree._load_dataset` and
`class_baselines/common.py::load_split` expect:

```
tests/data/tree/<name>/<name>_<seed>.csv     # seed in 1..5
```

with columns `<feature_1>, ..., <feature_d>, <target>, Split`, where `Split`
holds `train`/`test` and the target column is `class` (an integer label in
`0..K-1`) for classification or `target` (a float) for regression. All splits
are 80/20 with seeds 1-5; the classification scripts stratify by class
(`StratifiedShuffleSplit`), the regression scripts use a plain
`train_test_split`.

Each `<name>/prepare_<name>.py` regenerates its dataset's split CSVs.
`prepare_utils.py` holds the shared split logic used by the newer scripts
(credit, jannis); the older scripts are self-contained and follow
`magic/prepare_magic.py`.

## Small classification datasets

| name            | source                                   |     n |  d | K | classes (counts) |
|-----------------|------------------------------------------|------:|---:|--:|------------------|
| `iris`          | sklearn `load_iris` (UCI Iris)           |   150 |  4 | 3 | balanced (50/50/50) |
| `wine`          | UCI Wine (`wine.data`, tracked)          |   178 | 13 | 3 | 59/71/48 |
| `raisin`        | UCI Raisin (`Raisin_Dataset.xlsx`, tracked) | 900 |  7 | 2 | balanced; 0=Kecimen, 1=Besni |
| `breast_cancer` | sklearn `load_breast_cancer` (UCI WDBC)  |   569 | 30 | 2 | 212/357; 0=malignant, 1=benign |
| `magic`         | UCI MAGIC Gamma Telescope (`magic04.data`, tracked) | 19020 | 10 | 2 | 12332/6688; 0=gamma, 1=hadron |

Notes:

- `iris` and `breast_cancer` ship with scikit-learn — no raw file, no
  download. `wine`, `raisin` and `magic` read their raw file from the dataset
  directory (all three are tracked in git). `wine` remaps the raw labels 1-3
  to 0-2; `magic` maps g→0, h→1.
- `iris/prepare_iris.py` only writes seeds 2-5: `iris_1.csv` predates the
  script (it came from the original DT-GFN repository) and is *not*
  reproducible by rerunning with seed 1 (verified) — treat it as a fixed
  reference file and do not regenerate or overwrite it.
- `magic` is the only large one (15216 train samples) and is mildly
  imbalanced (65/35); the rest are small enough that single-split results are
  noisy — always average over the 5 splits.
- The `averaged_results_depth{2,3}.json` files in `iris/`, `wine/` and
  `breast_cancer/` are legacy evaluation artifacts of the original DT-GFN
  code, kept for reference; nothing in the current pipeline reads them.

## Regression datasets

All use a `target` float column and plain (unstratified) 80/20 splits.

| name             | source                                       |    n |  d | target (range) |
|------------------|----------------------------------------------|-----:|---:|----------------|
| `diabetes`       | sklearn `load_diabetes(scaled=False)`        |  442 | 10 | disease progression (25-346) |
| `energy`         | UCI Energy Efficiency (`ENB2012_data.xlsx`, tracked) | 768 | 8 | **heating** load Y1 (6.0-43.1) |
| `energy_cooling` | same raw file as `energy`                    |  768 |  8 | **cooling** load Y2 (10.9-48.0) |
| `concrete`       | UCI Concrete Compressive Strength (`Concrete_Data.xls`, tracked) | 1030 | 8 | strength in MPa (2.3-82.6) |

Notes:

- **There are two energy variants.** The UCI Energy Efficiency dataset has
  two targets; `energy` predicts the heating load (Y1) and `energy_cooling`
  the cooling load (Y2). Their splits are generated with the same seeds from
  the same rows, so `energy_<i>.csv` and `energy_cooling_<i>.csv` contain
  exactly the same buildings in the same train/test partition — results on
  the two are paired per split (same X, different y), not independent
  datasets. `energy_cooling/prepare_energy_cooling.py` reuses
  `../energy/ENB2012_data.xlsx` if present.
- `diabetes` uses the raw (unscaled) feature values on purpose: the Tree env
  min-max scales the data itself.
- `energy` and `concrete` auto-download their UCI zip on first run if the
  Excel file is missing (using certifi for TLS, since the cluster Python has
  no CA bundle); reading them needs `openpyxl` (xlsx) / `xlrd` (legacy xls).

## The tabular-benchmark datasets (credit, jannis)

Two datasets from the "classification on numerical features" suite of
Grinsztajn et al. (2022), which is numeric-only by construction.

| name      | source                    |     n |  d | K | reported best acc. |
|-----------|---------------------------|------:|---:|--:|--------------------|
| `credit`  | OpenML 44089 / task 361055 | 16714 | 10 | 2 | ~0.77 |
| `jannis2` | OpenML 44079 / task 361274 | 57580 | 54 | 2 | ~0.79 |
| `jannis4` | OpenML 41168 (ChaLearn AutoML) | 83733 | 54 | 4 | ~0.72 |

`jannis2` is the Grinsztajn binarization of `jannis4` (original class 1 against
the rest, subsampled to exact balance). `jannis4` is the version used by
Gorishniy et al. (2021), whose Table 2 gives MLP / ResNet / FT-Transformer /
NODE / XGBoost / CatBoost numbers.

### Generating the splits

Download the raw files once from a node with internet access (the compute nodes
on Mila and Trillium have none), then run each script. The scripts resolve all
paths relative to themselves, so the working directory does not matter.

```bash
module load python/3.10 && source ~/scratch/venvs/gflownet-env/bin/activate
cd tests/data/tree

curl -L -o credit/credit.csv \
  "https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/clf_num/credit.csv"
curl -L -o jannis2/jannis.csv \
  "https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/clf_num/jannis.csv"
curl -L -o jannis4/jannis_41168.pq \
  "https://data.openml.org/datasets/0004/41168/dataset_41168.pq"

python credit/prepare_credit.py --transform quantile   # see "credit" below
python jannis2/prepare_jannis_2classes.py
python jannis4/prepare_jannis_4classes.py
```

This writes 80/20 stratified splits for seeds 1-5, matching
`magic/prepare_magic.py`. Shared options (`prepare_utils.add_common_args`):

- `--seeds 1 2 3 4 5`, `--test-size 0.2` — split geometry.
- `--max-train N` — stratified-subsample the *training* set of each split to
  `N` (the test set is untouched). Use `--max-train 10000` to match the
  "medium-sized" regime that Grinsztajn's published numbers correspond to.
- `--transform quantile` — per-feature empirical-CDF transform, fit on the
  training split only. Writes to `<name>_quantile/` so both variants coexist.

## Loose files

`iris.csv` at this level is a legacy single-split copy of the iris dataset
(same column layout) predating the per-dataset directories. `trivial.csv`,
`trivial2d.csv` and `trivial_regression.csv` (generated by
`prepare_trivial_regression.py`) are tiny synthetic sanity-check datasets.

## Known issues

**`credit` needs `--transform quantile`.** The discrete tree node min-max
scales the data and picks thresholds from an equally-spaced grid on `[0, 1]`
(`n_thresholds: 99` in `config/experiments/tree/discrete_classification_tree.yaml`).
`RevolvingUtilizationOfUnsecuredLines` (max 22000, median 0.44) and `DebtRatio`
(max 61106, median 0.32) put 99.9% and 99.1% of their samples in the lowest
grid cell, so neither feature can be split at all, and the median feature still
has 59% of its mass in one cell. The quantile transform is monotone per
feature, so CART/RF/GBT baselines are invariant to it and published numbers
still apply. `prepare_utils` prints a warning whenever a feature crosses 95%
mass in one cell; neither jannis version triggers it. Both jannis versions are
already scaled to roughly `[0, 1]` upstream and need no transform.

**`credit` sentinel codes.** 89 rows use the Give Me Some Credit codes 96/98
in the three `NumberOfTime*DaysPastDue*` columns to mean "not available" /
"other"; they are read as counts. `--drop-sentinel-rows` removes them. The
published baselines keep them, so the default keeps them too.

**`jannis4` is imbalanced and partly binary-only.** Class counts are
1687 / 28790 / 14734 / 38522, so report macro-F1 alongside accuracy. `Tree`,
`CategoricalTreeProxy`, `eval_tree` and the CART/RF/GBT baselines in
`class_baselines/` are class-generic and handle K=4 (verified; multi-class
metrics are macro-F1, macro one-vs-rest AUC and multi-class log-loss). The
binarized-feature baselines (MAPTree, BCART MCMC/SMC) are binary by design
and skip `jannis4` with a message.

**Split protocol differs from the published baselines.** These are fresh
stratified 80/20 splits, not the OpenML task folds (Grinsztajn) or the fixed
train/val/test split (Gorishniy). Numbers lifted from those papers are
therefore indicative, not directly comparable — most sharply for
`--max-train`: Grinsztajn's medium regime trains on 10k samples, so quoting
those numbers next to a run trained on all 46k is unfair in our favour.

**The structure prior becomes negligible at this scale.** With
`prior_type: node_count` the prior contributes `-(log 4 + log d)` per internal
node: -5.4 for d=54, so at most -81 for a depth-5 tree. The marginal
log-likelihood is around -32000 (`jannis2`) and -74000 (`jannis4`), against
-9900 on magic — the prior is 0.25% of the objective and trees will grow to
`max_depth` regardless. Consider `proxy.normalize_likelihood: True` (divides
the log-likelihood by `N_train`, leaving the prior unscaled), or a BIC-style
`beta` that scales with `log n`, and report the choice explicitly.

**Repository size.** Five splits are ~134 MB for `jannis2` and ~195 MB for
`jannis4`. `.gitignore` in this directory keeps those and the raw downloads out
of git; `credit` (~0.9 MB per split) is tracked like `magic`. Regenerate the
jannis splits on each machine, or rsync them.
