# Rorqual: DT-GFN tree runs

Per-core scheduling like Mila: one Slurm array task per dataset split.
`drac/rorqual_cls_tree.sh` / `drac/rorqual_reg_tree.sh` are the Slurm headers
(account def-alexhg, 4 CPUs, 64 GB / 32 GB, 36 h / 24 h, `--array=1-5`);
`drac/cls_tree_worker.sh` / `drac/reg_tree_worker.sh` are the bodies of the
Mila launchers without the Mila-only header (run naming, config hash, resume
from `ckpts/`, final eval). Runs land in `$SCRATCH/gflownet-logs/<EXP_NAME>/<run_name>/`.

## 1. Launch (login node, like on Mila)

```bash
cd $HOME/gflownet && git pull          # same branch/commit as on Mila
mkdir -p $SCRATCH/gflownet-logs/slurm

# CREDITQUANT_trillium (fe19ccfb) and MAGIC_STAB2_B0.1_EPSANNEAL (32f39131) recipes;
# shared_weights=False = transformer backward policy with its own trunk.
CREDIT="gflownet.optimizer.n_train_steps=20000 gflownet.optimizer.lr=0.001 gflownet.optimizer.lr_decay_period=1000000 gflownet.optimizer.batch_size.forward=18 gflownet.optimizer.batch_size.backward_replay=2 gflownet.optimizer.batch_size.backward_dataset=0 policy.backward.shared_weights=False"
MAGIC="gflownet.optimizer.n_train_steps=20000 gflownet.optimizer.lr=0.001 env.max_depth=5 gflownet.optimizer.batch_size.forward=45 gflownet.optimizer.batch_size.backward_replay=5 gflownet.epsilon_annealing.enabled=True policy.backward.shared_weights=False"
TRFM=EXP_CONFIG=tree/trfm_classification_tree

# 25 transformer runs
sbatch --export=ALL,EXP_NAME=TRFM_CREDITQUANT_rorqual,DATASET=credit_quantile,$TRFM              drac/rorqual_cls_tree.sh $CREDIT
sbatch --export=ALL,EXP_NAME=TRFM_CREDITQUANT_B0.1_rorqual,DATASET=credit_quantile,$TRFM         drac/rorqual_cls_tree.sh $CREDIT proxy.reward_function_kwargs.beta=0.1
sbatch --export=ALL,EXP_NAME=TRFM_MAGIC_B0.1_EPSANNEAL_rorqual,DATASET=magic,$TRFM               drac/rorqual_cls_tree.sh $MAGIC proxy.reward_function_kwargs.beta=0.1
sbatch --export=ALL,EXP_NAME=TRFM_MAGIC_B1_EPSANNEAL_rorqual,DATASET=magic,$TRFM                 drac/rorqual_cls_tree.sh $MAGIC proxy.reward_function_kwargs.beta=1.0
sbatch --export=ALL,EXP_NAME=TRFM_MAGIC_B0.1_EPSANNEAL_REPLAY1K_rorqual,DATASET=magic,$TRFM      drac/rorqual_cls_tree.sh $MAGIC proxy.reward_function_kwargs.beta=0.1 buffer.replay_capacity=1000

# 5 MLP runs: every classification_tree.yaml default (fixed replay buffer, beta 1, batch 45/5), only 20k steps
sbatch --time=48:00:00 --mem=32G --export=ALL,EXP_NAME=CREDITQUANT_DEFAULTS_rorqual,DATASET=credit_quantile drac/rorqual_cls_tree.sh gflownet.optimizer.n_train_steps=20000
```

`$CREDIT` uses the fixed replay buffer; the fe19ccfb MLP runs had the old one
(`buffer.check_diversity=False buffer.diversity_check_reward_similarity=0.1`).

Other campaigns: change `EXP_NAME`, `DATASET`, `EXP_CONFIG` (default
`tree/classification_tree` = MLP) and the hydra overrides; resources on the
command line beat the header (`--array=1,3 --time=... --mem=...`).

`--time` too short? Resubmit the SAME line: finished runs are skipped
(`eval_results.json` exists), unfinished ones resume from `ckpts/` (every 500
steps). 20k transformer steps on 4 cores will not fit in 36 h (10k steps took
more than 32 h on 4 Mila cores on much smaller datasets), so expect 2-3 rounds;
a 20k-step MLP run with batch 45/5 took ~35 h on magic.

Check: `squeue -u $USER`; `head -30 $SCRATCH/gflownet-logs/slurm/cls_tree-<job>_<split>.out`
prints the run directory.

## 2. wandb (compute nodes have no internet, runs are written offline)

```bash
# login node; `module load StdEnv/2023 python/3.10 && source $SCRATCH/venvs/gflownet-env/bin/activate`; `wandb login` once
cd $SCRATCH/gflownet-logs/wandb && wandb sync --sync-all    # finds ./wandb/offline-run-*
```

Re-runnable; already-synced runs are skipped. Sync BEFORE resuming a run on
another cluster, otherwise the unsynced steps are dropped.

## 3. Results -> Mila

Rorqual and Mila share no SSH keys: go through the Mac with agent forwarding.

```bash
ssh -A rorqual
SRC=/scratch/arnit/gflownet-logs
DST=arnit@login.server.mila.quebec:/network/scratch/a/arnit/gflownet-logs
# would anything on Mila be overwritten? (same campaign + same config hash = same run dir); empty = safe
rsync -ahn --itemize-changes --exclude 'wandb/' $SRC/*_rorqual $DST/ | grep '^>f' | grep -v '+++'
rsync -ahz --progress --partial --exclude 'wandb/' $SRC/*_rorqual $DST/
```

No trailing slash on the sources (they are copied as directories). Re-runnable:
only new/changed files move. Then on Mila:

```bash
ROOT=$SCRATCH/gflownet-logs/<CAMPAIGN> bash mila/tree/aggregate_treeclass_results.sh
# a run copied before it finished: python mila/tree/relocate_run.py <run_dir>  BEFORE resuming it on Mila
```
