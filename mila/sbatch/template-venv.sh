#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={outdir}/{job_name}-%j.out
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --gres={gres}
#SBATCH --partition={partition}
#SBATCH --time={time}

echo
printf '%.0s-' {1..30} && echo
printf '%.0s-' {1..30} && echo
echo "\nStarting job $SLURM_JOB_ID at: `date`\n"

module load {modules}
source {venv}/bin/activate

cd {code_dir}

echo
printf '%.0s-' {1..30} && echo
printf '%.0s-' {1..30} && echo

python main.py {main_args}
