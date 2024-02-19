# Evaluation
All scripts related to structure evaluation are maintained here.
The default output directory, where standard data, plots, and cached results are stored is `data/crystals/eval_data/`.
## Short overview

To evaluate data, compare with other works, and create the summary plots, run the following scripts in order (be sure to be in root-dir `gflownet-dev/`):

```
python scripts/crystal_eval/convert_CGFN_samples.py --file_path data/crystals/crystalgfn_sgfirst_10ksamples_20231130.pkl
python scripts/crystal_eval/eval_CGFN.py
```
If needed, the first script can be run several times on different CGFN samples to compare them (changing --file_path).
Before running the second script (`eval_CGFN`) be sure that `data/crystals/eval_data/` contains all datasets in `.pkl` that need to be compared.

1. Read the Crystal GFlowNet samples and convert them to standard format:
    ```
    python scripts/crystal_eval/convert_CGFN_samples.py --file_path data/crystals/crystalgfn_sgfirst_10ksamples_20231130.pkl --out_dir data/crystals/eval_data/ --n_rand_struct 4
    ```
    You can provide the path to the CGFN samples with `--file_path` (defaults to `data/crystals/crystalgfn_sgfirst_10ksamples_20231130.pkl`)

    Optionally, add the `--n_rand_struct` followed by the number of structures you want to generate through pyXtal. This will create n additional structures with random Wyckoff positions. By default, it is set to 0. Be aware that this is not guaranteed to work and some entries might be None. See [Structure Generation with pyXtal](#structure-generation-with-pyxtal).

    The data is saved by default to `data/crystals/eval_data/`. This can be changed by using `--out_dir`.




2. **[Optional, not yet implemented]** You can download, convert save data from other works with the following script:
    ```
    python scripts/crystal_eval/pull_data.py
    ```
    As above, you have the `--out_dir` argument to change the output directory.

3.  ```
    python scripts/crystal_eval/eval_CGFN.py --sample_dir data/crystals/ --out_dir data/crystals/eval_data/ --force_compute
    ```
    Run the evaluation script. This will compute a bunch of metrics defined in `metrics.py`. Provide the pickled data from step one in `--sample_dir`. Results will be saved to `--out_dir`. Intermediate computations are cached to file (also in `--out_dir`), so subsequent runs will be fast. `--force_compute` will ignore any cached metrics.

## Structure Generation with pyXtal

You can generate structures with `--n_random_struct` and relax them with `relax.py`. However, it is not recommended using at this point, as it will result in many empty (None) structures. This will be addressed in a future PR.

## SMACT

The SMACT metric requires the python library "SMACT", that can be installed using `pip install smact`