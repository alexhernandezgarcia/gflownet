# DT-GFN tree experiments on DRAC (Trillium, Rorqual)

These clusters are not Mila: you need the **account name of a professor** to
submit anything (`--account=<account_name>`). How to get one and how to log in:
<https://docs.mila.quebec/technical_reference/clusters/drac/?h=drac>.

| file | what it is |
| --- | --- |
| `cls_tree_worker.sh`, `reg_tree_worker.sh` | the actual work: run naming, config hash, train / resume from `ckpts/`, final eval. Every launcher calls these. |
| `rorqual_example_runs/` | Rorqual launchers: one Slurm array task per dataset split. |
| `trillium_example_runs/` | Trillium launchers: one whole node running the full grid concurrently. Copy the closest one and edit the grid. |
| `resume_all_unfinished_runs.sh` | finish every unfinished run of a campaign in one new allocation. |

Runs land in `$SCRATCH/gflownet-logs/<EXP_NAME>/<run_name>/`.

## 1. Launch

In the launcher you use, edit the two `#SBATCH` lines that cannot expand
variables: `--account=<account_name>` and `--output=/scratch/<user>/...`.
`--time` sets the walltime; on the command line it (like `--mem`,
`--cpus-per-task`, `--array`) beats the header.

```bash
cd $HOME/gflownet && git pull
mkdir -p $SCRATCH/gflownet-logs/slurm

# Rorqual: per-core scheduling like Mila; ask for any CPUs / memory / walltime.
sbatch --time=36:00:00 --mem=64G \
  --export=ALL,EXP_NAME=<campaign>,DATASET=magic,EXP_CONFIG=tree/trfm_classification_tree \
  drac/rorqual_example_runs/rorqual_cls_tree.sh gflownet.optimizer.n_train_steps=20000

# Trillium: submit from $SCRATCH ($HOME is read-only on compute nodes).
cd $SCRATCH
sbatch --account=<account_name> --time=24:00:00 \
  $HOME/gflownet/drac/trillium_example_runs/trillium_reg_tree.sh
```

**Trillium** schedules by NODE: the smallest job is one full node (192 CPUs,
768 GB) and the walltime is **24 h maximum**, so put the whole grid on that one
node. **Rorqual** is much more flexible — take what you need.

Everything after the script name is a hydra override. `--time` too short?
Resubmit the SAME command: finished runs are skipped (`eval_results.json`
exists), unfinished ones resume from `ckpts/`. Check with `squeue -u $USER`;
`head -30 $SCRATCH/gflownet-logs/slurm/<job>.out` prints the run directory.

## 2. wandb (compute nodes have no internet, runs are written offline)

From a **login node**, after `wandb login` once:

```bash
module load StdEnv/2023 python/3.10
source $SCRATCH/venvs/gflownet-env/bin/activate
cd $SCRATCH/gflownet-logs/wandb
wandb sync --sync-all --include-offline
```

Faster, only some runs by name:

```bash
cd $SCRATCH/gflownet-logs/wandb/wandb
ls -d offline-run-20260827_035248-*   # confirm it's the ones you expect
wandb sync offline-run-20260827_035248-*
```

Re-runnable; already-synced runs are skipped. Sync **before** resuming a run on
another cluster, otherwise the unsynced steps are lost.

## 3. Results → Mila (cluster → your machine → Mila)

DRAC and Mila share no SSH keys, so your own machine is the hop. Everything
below runs there (macOS rsync is 2.6.9: `--progress`, not `--info=progress2`).
Swap `rorqual` for `trillium` as needed. Trailing slashes on both sides are
required, `$SCRATCH` in single quotes expands on the remote side, and `ckpts/`
must never be excluded.

```bash
# 0. what's there / how big
ssh rorqual 'du -sh $SCRATCH/gflownet-logs/*'

# 1. cluster -> your machine (skip wandb: tens of GB of offline runs)
mkdir -p ~/gflownet-logs-rorqual && cd ~/gflownet-logs-rorqual
rsync -ahz --progress --partial --partial-dir=.rsync-partial \
  --exclude 'wandb/' --exclude 'slurm/' \
  rorqual:'$SCRATCH/gflownet-logs/' ./

# 2. what would be OVERWRITTEN on Mila (dry run; same campaign + same config
#    hash = same run dir). Empty output = only new files = safe.
rsync -ahn --itemize-changes ./ mila:'$SCRATCH/gflownet-logs/' \
  | grep '^>f' | grep -v '+++'

# 3. -> Mila  (--backup-dir = undo button; drop it if step 2 printed nothing.
#    Spell it out: option values are not shell-expanded on the remote side.)
rsync -ahz --progress --partial --partial-dir=.rsync-partial \
  --backup --backup-dir=/network/scratch/<first-letter>/<user>/gflownet-logs-overwritten \
  ./ mila:'$SCRATCH/gflownet-logs/'

# 3b. or send a colliding campaign somewhere separate instead
rsync -ahz --progress ./<CAMPAIGN>/ mila:'$SCRATCH/gflownet-logs-rorqual/<CAMPAIGN>/'

# 4. check the run counts match
ssh rorqual 'for d in $SCRATCH/gflownet-logs/*/; do printf "%-45s %s\n" "$(basename $d)" "$(ls $d|wc -l)"; done'
ssh mila   'for d in $SCRATCH/gflownet-logs/*/; do printf "%-45s %s\n" "$(basename $d)" "$(ls $d|wc -l)"; done'
```

All of it is re-runnable: only new/changed files move.

## 4. Evaluations and results tables

On Mila, with the venv active:

```bash
# which finished runs have no eval_results.json yet (--dry-run only lists them)
python gflownet/envs/tree/helpers_for_experiments/run_missing_evals.py \
  $SCRATCH/gflownet-logs/<CAMPAIGN> --dry-run

# run them: drop --dry-run (--force re-evaluates runs that already have one)
python gflownet/envs/tree/helpers_for_experiments/run_missing_evals.py \
  $SCRATCH/gflownet-logs/<CAMPAIGN>

# results: one mean +/- std table per dataset, runs grouped by training config
python gflownet/envs/tree/helpers_for_experiments/aggregate_treeclass_results.py \
  $SCRATCH/gflownet-logs/<CAMPAIGN>
```

The aggregator prints two independent sources side by side: `eval` (the
`eval_results.json` on disk) and `wandb` (last logged value of each run, so
runs without a final eval are visible too). Useful flags: `--source eval|wandb`,
`--dataset iris,wine`, `--task regression`, `--min-splits 1` (configs with
fewer than 3 splits are hidden by default), `--diff-configs` (which config keys
separate two groups) and one filter per settings column (`--steps 10000
--depth 5 --policy mlp --prior bcart`). `--help` and the script's docstring
document the rest.
