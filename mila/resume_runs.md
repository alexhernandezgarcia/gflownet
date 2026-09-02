# Resume crashed / timed-out classification runs

`mila/tree/resume_crashed_magic_worker.sh` resumes ONE run from its newest
checkpoint. You pass only `RUN_DIR`; everything else (device, venv, all hydra
overrides) is read back from the run's own `.hydra/config.yaml`, so the
continued run is exactly the run that died. Training only, no eval.

```bash
# 0. why did it die? (jobs gone from squeue are still in sacct)
sacct -j <JOBID> --format=JobID%14,JobName%16,State%16,Elapsed,ExitCode
ls $SCRATCH/gflownet-logs/<CAMPAIGN>/<RUN>/ckpts/   # where it will restart from

# 1. resubmit ONE run
cd ~/gflownet
sbatch --job-name=<CAMPAIGN>_resume \
       --partition=long-cpu,long-cpu-eek \
       --time=6:00:00 --cpus-per-task=4 --mem=64G \
       --export=ALL,RUN_DIR=$SCRATCH/gflownet-logs/<CAMPAIGN>/<RUN> \
       mila/tree/resume_crashed_magic_worker.sh

# 2. ALWAYS verify after ~2 min: a broken run fails in <60 s and vanishes
squeue -u $USER
tail -20 $SCRATCH/gflownet-logs/slurm/<JOBNAME>-<JOBID>.out
# want: ">>> Resuming from .../ckpts" + a progress bar
# not:  "exited with code 1"

# 3. the rest of the campaign
for d in $SCRATCH/gflownet-logs/<CAMPAIGN>/*/; do
  sbatch --job-name=<CAMPAIGN>_resume \
         --partition=long-cpu,long-cpu-eek \
         --time=6:00:00 --cpus-per-task=4 --mem=64G \
         --export=ALL,RUN_DIR="${d%/}" \
         mila/tree/resume_crashed_magic_worker.sh
done
```

GPU runs (`device: cuda` in the config) instead need
`--partition=long --gres=gpu:l40s:1` (`a100l` for depth >= 6); the script picks
the matching venv by itself.

Re-runnable by design: finished runs (`ckpts/final.ckpt` or
`samples/gfn_samples.pkl`) exit 0 immediately, and a checkpoint truncated by the
kill is quarantined before loading. So if 6 h is not enough, just run the same
loop again -- it continues from the newest checkpoint. Sizing: remaining steps x
s/it (read the s/it off the old log's progress bar) + ~15 min margin.

`mila/tree/resume_crashed_magic.py --dry-run` does all of the above for the
hard-coded magic campaigns (scan, classify, size the walltime); the manual
`sbatch` above is for everything else.

Gotcha seen on 2026-08-27: a `pip install` in `$SCRATCH/venvs/gflownet-env` had
pulled wandb 0.29, which dropped `Run.get_url()` used in
`gflownet/utils/logger.py` -> every resume died after 27 s. Environment drift
breaks resumes silently; step 2 is what catches it.
