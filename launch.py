from pathlib import Path
from os import popen
from os.path import expandvars
from argparse import ArgumentParser
import datetime
from yaml import safe_load
import re
from textwrap import dedent
import sys

HELP = dedent(
    """
    Fill in an sbatch template and submit a job.

    Examples:

    $ python launch.py --main_args='user=$USER logger.do.online=False'

    $ python launch.py --template=sbatch/template-venv.sh \\
        --venv='~/.venvs/gfn' \\
        --modules='python/3.7 cuda/11.3' \\
        --main_args='user=$USER logger.do.online=False'

    $ python launch.py --runs=runs/comp-sg-lp/v0" --mem=32G
        where the file ./runs/comp-sg-lp/v0.yaml contains:
        ```
        shared:
          job:
            gres: gpu:1
            mem: 16G
            cpus_per_task: 2
          main_args: "user=$USER +experiments=neurips23/crystal-comp-sg-lp.yaml"

        runs:
        - main_args: "gflownet.policy.backward=null gflownet=flowmatch"
        - main_args: "gflownet=trajectorybalance"
        ```

        This will execute 2 jobs with the following args order update:
            * SLURM params:
                1. shared_job params
                2. run.job params
                3. command-line args
            * main.py args:
                1. shared_main_args
                2. run.main_args
                3. command-line --main_args='[...]'
        """
)


def resolve(path):
    """
    Resolves a path with environment variables and user expansion.
    All paths will end up as absolute paths.

    Args:
        path (str | Path): The path to resolve

    Returns:
        Path: resolved path
    """
    if path is None:
        return None
    return Path(expandvars(str(path))).expanduser().resolve()


def now_str():
    """
    Returns a string with the current date and time.
    Eg: "20210923_123456"

    Returns:
        str: current date and time
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def load_runs(yaml_path):
    """
    Loads a yaml file with run configurations and turns it into a list of runs.

    Example yaml file:

    ```
    shared:
      job:
        gres: gpu:1
        mem: 16G
        cpus_per_task: 2
      main_args: "user=$USER +experiments=neurips23/crystal-comp-sg-lp.yaml"

    runs:
    - main_args: "gflownet.policy.backward=null gflownet=flowmatch"
    - job:
        partition: main
      main_args: "gflownet.policy.backward=null gflownet=flowmatch"
    ```

    Args:
        yaml_path (str | Path): Where to fine the yaml file

    Returns:
        list[dict]: List of run configurations as dicts
    """
    if yaml_path is None:
        return []
    with open(yaml_path, "r") as f:
        run_config = safe_load(f)

    job = run_config.get("shared", {}).get("job", {})
    sma = run_config.get("shared", {}).get("main_args", "")
    if sma:
        sma = sma + " "

    runs = []
    for r in run_config["runs"]:
        rj = job.copy()
        if "job" in r:
            rj.update(r["job"])
            r["job"] = rj
        else:
            r["job"] = rj
        assert "main_args" in r, f"main_args not in {r}"
        r["main_args"] = sma + r["main_args"].strip()

        runs.append(r)
    return runs


def find_run_conf(args):
    if not args.get("runs"):
        return None
    if args["runs"].endswith(".yaml"):
        args["runs"] = args["runs"][:-5]
    if args["runs"].endswith(".yml"):
        args["runs"] = args["runs"][:-4]
    if args["runs"].startswith("external/"):
        args["runs"] = args["runs"][9:]
    if args["runs"].startswith("runs/"):
        args["runs"] = args["runs"][5:]
    yamls = [
        str(y) for y in (root / "external" / "runs").glob(f"**/{args['runs']}.y*ml")
    ]
    if len(yamls) == 0:
        raise ValueError(f"Could not find {args['runs']}.y(a)ml in ./external/runs/")
    if len(yamls) > 1:
        print(">>> Warning: found multiple matches:\n  •" + "\n  •".join(yamls))
    runs_conf_path = Path(yamls[0])
    print("🗂 Using run file: ./" + str(runs_conf_path.relative_to(Path.cwd())))
    print()
    return runs_conf_path


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "--help", action="store_true", help="show this help message and exit"
    )
    parser.add_argument("--job_name", type=str)
    parser.add_argument("--outdir", type=str, help="where to write the slurm .out file")
    parser.add_argument("--cpus_per_task", type=int)
    parser.add_argument("--mem", type=str)
    parser.add_argument("--gres", type=str)
    parser.add_argument("--partition", type=str)
    parser.add_argument("--modules", type=str, help="string after 'module load'")
    parser.add_argument("--conda_env", type=str, help="conda environment name")
    parser.add_argument("--venv", type=str, help="path to venv (without bin/activate)")
    parser.add_argument(
        "--code_dir", type=str, help="cd before running main.py (defaults to here)"
    )
    parser.add_argument("--main_args", type=str, help="main.py args")
    parser.add_argument(
        "--runs", type=str, help="run file name in external/runs (without .yaml)"
    )
    parser.add_argument(
        "--dev", action="store_true", help="Don't run just show what it would have run"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print templated sbatch after running it"
    )

    args = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    root = Path(__file__).resolve().parent

    defaults = {
        "job_name": "crystal-gfn",
        "outdir": "$SCRATCH/crystals/logs/slurm",
        "cpus_per_task": 2,
        "mem": "32G",
        "gres": "gpu:1",
        "partition": "long",
        "modules": "anaconda/3 cuda/11.3",
        "conda_env": "gflownet",
        "code_dir": "~/ocp-project/gflownet",
        "main_args": None,
        "runs": None,
        "dev": False,
        "verbose": False,
        "venv": None,
        "template": root / "sbatch" / "template-conda.sh",
    }

    if args.get("help"):
        print(HELP)
        sys.exit(0)

    # load sbatch template file to format
    template = resolve(args.get("template", defaults["template"])).read_text()
    # find the required formatting keys
    template_keys = set(re.findall(r"{(\w+)}", template))

    # in dev mode: no mkdir, no sbatch etc.
    dev = args.get("dev")

    # where to write the slurm output file
    outdir = resolve(args.get("outdir", defaults["outdir"]))
    if not dev:
        outdir.mkdir(parents=True, exist_ok=True)

    # find runs config file in external/runs as a yaml file
    runs_conf_path = find_run_conf(args)
    # load yaml file as list of dicts. May be empty if runs_conf_path is None
    runs = load_runs(runs_conf_path)
    # No run passed in the CLI args or in the associated yaml file so run the
    # CLI main_args, if any.
    if not runs:
        assert args["main_args"], "main_args must be specified if no runs are given"
        runs = [{}]

    # Save submitted jobs ids
    job_ids = []

    # A unique datetime identifier for the runs about to be submitted
    now = now_str()

    for i, run in enumerate(runs):
        job = defaults.copy()
        job.update(run.pop("job", {}))
        run_args = {**job, **run, **args}

        run_args["code_dir"] = str(resolve(run_args["code_dir"]))
        run_args["outdir"] = str(resolve(run_args["outdir"]))
        run_args["venv"] = str(resolve(run_args["venv"]))

        # filter out useless args for the template
        run_args = {k: str(v) for k, v in run_args.items() if k in template_keys}
        # Make sure all the keys in the template are in the args
        if set(template_keys) != set(run_args.keys()):
            print(f"template keys: {template_keys}")
            print(f"template args: {run_args}")
            raise ValueError(
                "template keys != template args (see details printed above)"
            )

        # format template for this run
        templated = template.format(**run_args)

        # set output path for the sbatch file to execute in order to submit the job
        if runs_conf_path is not None:
            sbatch_path = (
                runs_conf_path.parent / f"{runs_conf_path.stem}_{now}_{i}.sbatch"
            )
        else:
            sbatch_path = root / f"various/{run_args['job_name']}_{now}.sbatch"

        if not dev:
            # make sure the sbatch file parent directory exists
            sbatch_path.parent.mkdir(parents=True, exist_ok=True)
            # write template
            sbatch_path.write_text(templated)
            print(f"  🏷 Created ./{sbatch_path.relative_to(Path.cwd())}")
            # Submit job to SLURM
            out = popen(f"sbatch {sbatch_path}").read()
            # Identify printed-out job id
            job_id = re.findall(r"Submitted batch job (\d+)", out)[0]
            job_ids.append(job_id)
            print("  ✅ " + out)
            # Write job ID & output file path in the sbatch file
            templated += (
                "\n# SLURM_JOB_ID: "
                + job_id
                + "\n# Output file: "
                + str(outdir / f"{run_args['job_name']}-{job_id}.out")
                + "\n"
            )
            sbatch_path.write_text(templated)

        # final prints for dev & verbose mode
        if dev or args.get("verbose"):
            if dev:
                print("\nDEV: would have writen in sbatch file:", str(sbatch_path))
            print("#" * 40 + " <sbatch> " + "#" * 40)
            print(templated)
            print("#" * 40 + " </sbatch> " + "#" * 39)
            print()

    # Recap submitted jobs. Useful for scancel for instance.
    if job_ids:
        print("\n🔥 All jobs submitted: " + " ".join(job_ids) + "\n")
