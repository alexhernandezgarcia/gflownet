# DT-GFN tree experiments on the Mila cluster

| file | what it does |
| --- | --- |
| `run_classification_tree_training.sh` | Train + evaluate a classification tree. One Slurm array task per dataset split (1-5), CPU. |
| `run_regression_tree_training.sh` | Same for regression trees. |
| `run_*_tree_training_gpu.sh` | GPU (l40s) variants of the two scripts above. |
| `resume_crashed_treeclass_run_worker.sh` | Resume ONE crashed/timed-out run (classification or regression) from its newest checkpoint. Training only. |
| `resume_crashed_treeclass_run.py` | Scan whole campaigns and submit the worker above for every unfinished run. |
| `treeclass_run_eval.sh` | Evaluate every finished run that has no `eval_results.json` yet. |
| `class_baselines.sh` | Classification baselines (BCART, MAPTree, CART, RF, boosters). Submits itself: `bash mila/tree/class_baselines.sh`. |
| `reg_baselines.sh`, `reg_bcart_baselines.sh` | Regression baselines (BART, boosters, GP, linear / BCART). Submit themselves like the one above. |

Every script documents all of its options in its header.

## Launch a training run

1. One-time setup:
   - In the scripts you use, replace `/home/mila/a/arnit` with your own paths
     in the `#SBATCH --output=` line and in `REPO=`.
   - Create the virtualenvs and log in to wandb:
     ```bash
     cd ~/gflownet && module load python/3.10
     source install.sh --cpu  --envpath $SCRATCH/venvs/gflownet-env
     source install.sh --cuda --envpath $SCRATCH/venvs/gflownet-env-gpu   # only for GPU runs
     wandb login
     mkdir -p $SCRATCH/gflownet-logs/slurm
     ```
2. Choose the settings:
   - `DATASET`: a folder in `tests/data/tree/` (e.g. `magic`, `diabetes`).
   - `EXP_CONFIG`: a file in `config/experiments/tree/`, e.g.
     `tree/classification_tree` (MLP policy) or `tree/trfm_classification_tree`
     (transformer policy); the `regression_tree` variants for regression.
   - `EXP_NAME`: campaign name, i.e. the folder the runs are written to.
3. Submit from the repo root. Any hydra overrides go after the script name:
   ```bash
   sbatch --time=36:00:00 \
     --export=ALL,EXP_NAME=MY_CAMPAIGN,DATASET=magic,EXP_CONFIG=tree/classification_tree \
     mila/tree/run_classification_tree_training.sh \
     gflownet.optimizer.n_train_steps=20000 env.max_depth=5
   ```
   Defaults: splits 1-5, 4 CPUs, 24G (regression: 32G), 18 h per task. Use `--array=1,3` for
   some splits only. Big runs need more `--time` (magic, MLP, 20k steps: about
   35 h), and transformer runs need `--mem=64G`.
4. Monitor with `squeue -u $USER` and
   `tail $SCRATCH/gflownet-logs/slurm/<jobname>-<arrayjobid>_<split>.out`.
5. Results are in `$SCRATCH/gflownet-logs/<EXP_NAME>/<run_name>/`. The run is
   complete once `eval_results.json` exists.

## Resume a crashed run

The worker restarts a run from the newest checkpoint in `<run_dir>/ckpts/`. It
reads everything else (config, device, venv) from the run's own
`.hydra/config.yaml`. Runs that already finished exit immediately, so
resubmitting is always safe. It only trains: evaluation is step 5.

1. Find out why the run died, and check that it has a checkpoint:
   ```bash
   sacct -j <JOBID> --format=JobID%14,JobName%16,State%16,Elapsed,ExitCode
   ls $SCRATCH/gflownet-logs/<CAMPAIGN>/<RUN>/ckpts/
   ```
   Walltime to request: remaining steps × s/it (read it off the progress bar in
   the old log) + ~15 min.
2. Resubmit, either **one run**:
   ```bash
   cd ~/gflownet
   sbatch --job-name=<CAMPAIGN>_resume \
          --partition=long-cpu,long-cpu-eek \
          --time=6:00:00 --cpus-per-task=4 --mem=64G \
          --export=ALL,RUN_DIR=$SCRATCH/gflownet-logs/<CAMPAIGN>/<RUN> \
          mila/tree/resume_crashed_treeclass_run_worker.sh
   ```
   For GPU runs (`device: cuda` in the config), use
   `--partition=long --gres=gpu:l40s:1` instead (`a100l` for depth ≥ 6).

   or **every unfinished run of one or more campaigns**. The script picks the
   partition, GPU and walltime for each run:
   ```bash
   python mila/tree/resume_crashed_treeclass_run.py <CAMPAIGN> ... --dry-run   # check the plan first
   python mila/tree/resume_crashed_treeclass_run.py <CAMPAIGN> ... --sec-per-step <s/it> --mem 64G
   ```
3. After ~2 min, check that the resume actually started. A broken resume fails
   in under 60 s and disappears from `squeue`:
   ```bash
   squeue -u $USER
   tail -20 $SCRATCH/gflownet-logs/slurm/<JOBNAME>-<JOBID>.out
   # good: ">>> Resuming from .../ckpts" followed by a progress bar
   # bad:  "exited with code 1"
   ```
4. Out of time again? Repeat step 2. The run continues from its newest
   checkpoint.
5. Once training has finished, evaluate:
   ```bash
   sbatch --export=ALL,ROOT=$SCRATCH/gflownet-logs/<CAMPAIGN> mila/tree/treeclass_run_eval.sh
   ```
