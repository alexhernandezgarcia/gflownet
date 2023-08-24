#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={outdir}/{job_name}-%j.out
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --gres={gres}
#SBATCH --partition={partition}

module load {modules}
source {venv}/bin/activate

cd {code_dir}

python main.py {main_args}