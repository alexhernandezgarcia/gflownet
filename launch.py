from pathlib import Path
from os import popen
from os.path import expandvars
from argparse import ArgumentParser
import datetime
from yaml import safe_load
import re
from textwrap import dedent
import sys
from copy import deepcopy

HELP = dedent(
    """
    >>> HOW TO USE:

    Fill in an sbatch template and submit a job.

    Examples:

    # using default job configuration, with script args from the command-line:
    $ python launch.py user=$USER logger.do.online=False

    # overriding the default job configuration and adding script args:
    $ python launch.py --template=sbatch/template-venv.sh \\
        --venv='~/.venvs/gfn' \\
        --modules='python/3.7 cuda/11.3' \\
        user=$USER logger.do.online=False

    # using a yaml file to specify multiple jobs to run:
    $ python launch.py --runs=runs/comp-sg-lp/v0" --mem=32G

        Explanation:
        ------------

        Say the file ./extenal/runs/comp-sg-lp/v0.yaml contains:

        ```
        shared:
          job:
            gres: gpu:1
            mem: 16G
            cpus_per_task: 2
          script:
            user: $USER
            +experiments: neurips23/crystal-comp-sg-lp.yaml
            gflownet:
              __value__: flowmatch
              optimizer:
                lr: 0.0001

        runs:
        - script:
            gflownet:
              policy:
                backward: null
        - script:
            gflownet:
              __value__: trajectorybalance
        ```

        Then the command-line ^ will execute 2 jobs with the following
        configurations:
            * SLURM params:
                1. shared.job params
                2. run.job params
                3. command-line args (eg: --mem=32G in this example)
            * Python script (main.py) args:
                1. shared.main_args
                2. run.main_args (appended to shared.main_args if present)
                3. command-line args (eg: --main_args='[...]', absent in this example)
            * All of the above are optional granted they are defined at least once
                somewhere.

        1. -> python main.py user=$USER +experiments=neurips23/crystal-comp-sg-lp.yaml gflownet=flowmatch gflownet.optimizer.lr=0.0001 gflownet.policy.backward=None
        2. -> python main.py user=$USER +experiments=neurips23/crystal-comp-sg-lp.yaml gflownet=trajectorybalance gflownet.optimizer.lr=0.0001
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
      script:
        user: $USER
        +experiments: neurips23/crystal-comp-sg-lp.yaml
        gflownet:
          __value__: tranjectorybalance

    runs:
    - {}
    - script:
        gflownet:
            __value__: flowmatch
            policy:
                backward: null
    - job:
        partition: main
      script:
        gflownet.policy.backward: null
        gflownet: flowmatch
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

    shared_job = run_config.get("shared", {}).get("job", {})
    shared_script = run_config.get("shared", {}).get("script", {})
    runs = []
    for run_dict in run_config["runs"]:
        run_job = deep_update(shared_job, run_dict.get("job", {}))
        run_script = deep_update(shared_script, run_dict.get("script", {}))
        run_dict["job"] = run_job
        run_dict["script"] = run_script
        runs.append(run_dict)
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


def script_args_dict_to_main_args_str(script_dict, is_first=True, nested_key=""):
    """
    Recursively turns a dict of script args into a string of main.py args
    as `nested.key=value` pairs

    Args:
        script_dict (dict): script dictionary of args
        previous_str (str, optional): base string to append to. Defaults to "".
    """
    if not isinstance(script_dict, dict):
        return nested_key + "=" + str(script_dict) + " "
    new_str = ""
    for k, v in script_dict.items():
        if k == "__value__":
            new_str += nested_key + "=" + str(v) + " "
            continue
        new_key = k if not nested_key else nested_key + "." + str(k)
        new_str += script_args_dict_to_main_args_str(
            v, nested_key=new_key, is_first=False
        )
    if is_first:
        new_str = new_str.strip()
    return new_str


def deep_update(a, b, path=None, verbose=None):
    """
    https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries/7205107#7205107

    Args:
        a (dict): dict to update
        b (dict): dict to update from

    Returns:
        dict: updated copy of a
    """
    if path is None:
        path = []
        a = deepcopy(a)
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                deep_update(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                if verbose:
                    print(">>> Warning: Overwriting", ".".join(path + [str(key)]))
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "-h", "--help", action="store_true", help="show this help message and exit"
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
    parser.add_argument("--force", action="store_true", help="skip user confirmation")

    known, unknown = parser.parse_known_args()

    cli_script_args = " ".join(unknown) if unknown else ""

    args = {k: v for k, v in vars(known).items() if v is not None}
    root = Path(__file__).resolve().parent

    defaults = {
        "code_dir": "~/ocp-project/gflownet",
        "conda_env": "gflownet",
        "cpus_per_task": 2,
        "dev": False,
        "force": False,
        "gres": "gpu:1",
        "job_name": "crystal-gfn",
        "main_args": None,
        "mem": "32G",
        "modules": "anaconda/3 cuda/11.3",
        "outdir": "$SCRATCH/crystals/logs/slurm",
        "partition": "long",
        "runs": None,
        "template": root / "sbatch" / "template-conda.sh",
        "venv": None,
        "verbose": False,
    }

    if args.get("help"):
        print(parser.format_help())
        print(HELP)
        sys.exit(0)

    # load sbatch template file to format
    template = resolve(args.get("template", defaults["template"])).read_text()
    # find the required formatting keys
    template_keys = set(re.findall(r"{(\w+)}", template))

    # in dev mode: no mkdir, no sbatch etc.
    dev = args.get("dev")

    # in force mode, no confirmation is asked
    force = args.get("force", defaults["force"])

    # where to write the slurm output file
    outdir = resolve(args.get("outdir", defaults["outdir"]))
    if not dev:
        outdir.mkdir(parents=True, exist_ok=True)

    # find runs config file in external/runs as a yaml file
    runs_conf_path = find_run_conf(args)
    # load yaml file as list of dicts. May be empty if runs_conf_path is None
    run_dicts = load_runs(runs_conf_path)
    # No run passed in the CLI args or in the associated yaml file so run the
    # CLI main_args, if any.
    if not run_dicts:
        run_dicts = [{}]

    # Save submitted jobs ids
    job_ids = []

    # A unique datetime identifier for the runs about to be submitted
    now = now_str()

    if not force and not dev:
        if "y" not in input(f"🚨 Submit {len(run_dicts)} jobs? [y/N] ").lower():
            print("🛑 Aborted")
            sys.exit(0)
        print()

    local_out_dir = root / "external" / "launched_sbatch_scripts"
    if runs_conf_path is not None:
        local_out_dir = local_out_dir / runs_conf_path.parent.relative_to(
            root / "external" / "runs"
        )
    else:
        local_out_dir = local_out_dir / "_other_"

    for i, run_dict in enumerate(run_dicts):
        run_args = defaults.copy()
        run_args = deep_update(run_args, run_dict.pop("job", {}))
        run_args = deep_update(run_args, run_dict)
        run_args = deep_update(run_args, args)

        run_args["code_dir"] = str(resolve(run_args["code_dir"]))
        run_args["outdir"] = str(resolve(run_args["outdir"]))
        run_args["venv"] = str(resolve(run_args["venv"]))
        run_args["main_args"] = (
            script_args_dict_to_main_args_str(run_args["script"]) + cli_script_args
        )

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
            sbatch_path = local_out_dir / f"{runs_conf_path.stem}_{now}_{i}.sbatch"
        else:
            sbatch_path = local_out_dir / f"{run_args['job_name']}_{now}.sbatch"

        if not dev:
            # make sure the sbatch file parent directory exists
            sbatch_path.parent.mkdir(parents=True, exist_ok=True)
            # write template
            sbatch_path.write_text(templated)
            print(f"  🏷  Created ./{sbatch_path.relative_to(Path.cwd())}")
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
    jobs_str = "⚠️ No job submitted!"
    if job_ids:
        jobs_str = "All jobs submitted: " + " ".join(job_ids)
        print(f"\n🚀 Submitted job {i+1}/{len(run_dicts)}")

    # make copy of original yaml conf and append all the sbatch info:
    if runs_conf_path is not None:
        conf = runs_conf_path.read_text()
        new_conf_path = local_out_dir / f"{runs_conf_path.stem}_{now}.yaml"
        new_conf_path.parent.mkdir(parents=True, exist_ok=True)
        conf += "\n# " + jobs_str + "\n"
        new_conf_path.write_text(conf)
        rel = new_conf_path.relative_to(Path.cwd())
        if not dev:
            print(f"   Created summary YAML in ./{rel}")

    if job_ids:
        print(f"   {jobs_str}\n")
