# Trillium -> Mac -> Mila run sync

Mac rsync is 2.6.9: use `--progress`, not `--info=progress2`.

```bash
# 0. what's there / how big
ssh trillium 'du -sh /scratch/arnit/gflownet-logs/*'

# 1. Trillium -> Mac  (skip wandb: ~28 GB of offline runs)
mkdir -p ~/gflownet-logs-trillium && cd ~/gflownet-logs-trillium
rsync -ahz --progress --partial --partial-dir=.rsync-partial \
  --exclude 'wandb/' --exclude 'slurm/' \
  trillium:/scratch/arnit/gflownet-logs/ ./

# 2. what would be OVERWRITTEN on Mila (dry run; same campaign names exist on both clusters)
rsync -ahn --itemize-changes ./ mila:/network/scratch/a/arnit/gflownet-logs/ \
  | grep '^>f' | grep -v '+++'
# empty output = only new files = safe. Anything printed = existing Mila file gets replaced.

# 3. Mac -> Mila  (--backup-dir = undo button, drop it if step 2 was empty)
rsync -ahz --progress --partial --partial-dir=.rsync-partial \
  --backup --backup-dir=/network/scratch/a/arnit/gflownet-logs-overwritten \
  ./ mila:/network/scratch/a/arnit/gflownet-logs/

# 3b. or send a colliding campaign somewhere separate instead
rsync -ahz --progress ./TRFM_REG_CONTROL/ \
  mila:/network/scratch/a/arnit/gflownet-logs-trillium/TRFM_REG_CONTROL/

# 4. check counts match
ssh trillium 'for d in /scratch/arnit/gflownet-logs/*/; do printf "%-30s %s\n" "$(basename $d)" "$(ls $d|wc -l)"; done'
ssh mila 'cd /network/scratch/a/arnit/gflownet-logs && for d in */; do printf "%-30s %s\n" "${d%/}" "$(ls $d|wc -l)"; done'
```

Trailing slashes on both sides are required. All of it is re-runnable: only new/changed files move.

Then on Mila (needs `ckpts/`, so never exclude those):

```bash
python gflownet/envs/tree/helpers_for_experiments/run_missing_evals.py $SCRATCH/gflownet-logs/<CAMPAIGN> --dry-run
ROOT=$SCRATCH/gflownet-logs/<CAMPAIGN> bash mila/tree/aggregate_treeclass_results.sh
```

wandb offline runs: sync from the Trillium login node, `wandb sync --sync-all` in `/scratch/arnit/gflownet-logs`.
