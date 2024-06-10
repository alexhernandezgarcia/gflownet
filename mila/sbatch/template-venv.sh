#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={outdir}/{job_name}-%j.out
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --gres={gres}
#SBATCH --partition={partition}
#SBATCH --time={time}

python3 -c 'print("\n" + "-" * 40 + "\n" + "-" * 40)'
echo "Starting job $SLURM_JOB_ID at: `date`"

module load {modules}
source {venv}/bin/activate

cd {code_dir}

python3 -c 'print("\n" + "-" * 40 + "\n" + "-" * 40)'
echo

python main.py {main_args}
