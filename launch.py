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
    ## 🥳 User guide

    Fill in an sbatch template and submit a job.

    Examples:

    ```sh
    # using default job configuration, with script args from the command-line:
    $ python launch.py user=$USER logger.do.online=False

    # overriding the default job configuration and adding script args:
    $ python launch.py --template=sbatch/template-venv.sh \\
        --venv='~/.venvs/gfn' \\
        --modules='python/3.7 cuda/11.3' \\
        user=$USER logger.do.online=False

    # using a yaml file to specify multiple jobs to run:
    $ python launch.py --jobs=jobs/comp-sg-lp/v0" --mem=32G
    ```

    ### 📝 Explanation

    Case-study:

    ```sh
    python launch.py --jobs=crystals/explore-losses --mem=32G
    ```

    Say the file `./external/jobs/crystals/explore-losses.yaml` contains:

    ```yaml
    # Contents of external/jobs/crystals/explore-losses.yaml

    # Shared section across jobs
    shared:
      # job params
      slurm:
          template: sbatch/template-conda.sh # which template to use
          modules: anaconda/3 cuda/11.3      # string of the modules to load
          conda_env: gflownet                # name of the environment
          code_dir: ~/ocp-project/gflownet   # where to find the repo
          gres: gpu:1                        # slurm gres
          mem: 16G                           # node memory
          cpus_per_task: 2                   # task cpus

      # main.py params
      script:
        user: $USER
        +experiments: neurips23/crystal-comp-sg-lp.yaml
        gflownet:
          __value__: flowmatch               # special entry if you want to see `gflownet=flowmatch`
        optimizer:
          lr: 0.0001                     # will be translated to `gflownet.optimizer.lr=0.0001`

    # list of slurm jobs to execute
    jobs:
      - {}                                   # empty dictionary = just run with the shared params
      - slurm:                               # change this job's slurm params
          partition: unkillable
        script:                              # change this job's script params
          gflownet:
            policy:
              backward: null
      - script:
          gflownet:
            __value__: trajectorybalance
    ```

    Then the command-line ^ will execute 3 jobs with the following
    configurations:

    ```bash
    python main.py user=$USER +experiments=neurips23/crystal-comp-sg-lp.yaml gflownet=flowmatch gflownet.optimizer.lr=0.0001

    python main.py user=$USER +experiments=neurips23/crystal-comp-sg-lp.yaml gflownet=flowmatch gflownet.optimizer.lr=0.0001 gflownet.policy.backward=None

    python main.py user=$USER +experiments=neurips23/crystal-comp-sg-lp.yaml gflownet=trajectorybalance gflownet.optimizer.lr=0.0001
    ```

    And their SLURM configuration will be similar as the `shared.slurm` params, with the following differences:

    1. The second job will have `partition: unkillable` instead of the default (`long`).
    2. They will all have `64G` of memory instead of the default (`32G`) because the `--mem=64G` command-line
        argument overrides everything.
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


def load_jobs(yaml_path):
    """
    Loads a yaml file with run configurations and turns it into a list of jobs.

    Example yaml file:

    ```
    shared:
      slurm:
        gres: gpu:1
        mem: 16G
        cpus_per_task: 2
      script:
        user: $USER
        +experiments: neurips23/crystal-comp-sg-lp.yaml
        gflownet:
          __value__: tranjectorybalance

    jobs:
    - {}
    - script:
        gflownet:
            __value__: flowmatch
            policy:
                backward: null
    - slurm:
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
        jobs_config = safe_load(f)

    shared_slurm = jobs_config.get("shared", {}).get("slurm", {})
    shared_script = jobs_config.get("shared", {}).get("script", {})
    jobs = []
    for job_dict in jobs_config["jobs"]:
        job_slurm = deep_update(shared_slurm, job_dict.get("slurm", {}))
        job_script = deep_update(shared_script, job_dict.get("script", {}))
        job_dict["slurm"] = job_slurm
        job_dict["script"] = job_script
        jobs.append(job_dict)
    return jobs


def find_jobs_conf(args):
    if not args.get("jobs"):
        return None
    if args["jobs"].endswith(".yaml"):
        args["jobs"] = args["jobs"][:-5]
    if args["jobs"].endswith(".yml"):
        args["jobs"] = args["jobs"][:-4]
    if args["jobs"].startswith("external/"):
        args["jobs"] = args["jobs"][9:]
    if args["jobs"].startswith("jobs/"):
        args["jobs"] = args["jobs"][5:]
    yamls = [
        str(y) for y in (root / "external" / "jobs").glob(f"**/{args['jobs']}.y*ml")
    ]
    if len(yamls) == 0:
        raise ValueError(f"Could not find {args['jobs']}.y(a)ml in ./external/jobs/")
    if len(yamls) > 1:
        print(">>> Warning: found multiple matches:\n  •" + "\n  •".join(yamls))
    jobs_conf_path = Path(yamls[0])
    print("🗂 Using jobs file: ./" + str(jobs_conf_path.relative_to(Path.cwd())))
    print()
    return jobs_conf_path


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


def print_md_help(parser, defaults):
    global HELP

    print("# 🤝 Gflownet Launch tool help\n")
    print("## 💻 Command-line help\n")
    print(parser.format_help())
    print("## 🎛️ Default values\n")
    print(
        "```yaml\n"
        + "\n".join(
            [
                f"{k:{max(len(d) for d in defaults)+1}}: {str(v)}"
                for k, v in defaults.items()
            ]
        )
        + "\n```"
    )
    print(HELP, end="")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    defaults = {
        "code_dir": "~/ocp-project/gflownet",
        "conda_env": "gflownet",
        "cpus_per_task": 2,
        "dev": False,
        "force": False,
        "gres": "gpu:1",
        "job_name": "crystal-gfn",
        "jobs": None,
        "main_args": None,
        "mem": "32G",
        "modules": "anaconda/3 cuda/11.3",
        "outdir": "$SCRATCH/crystals/logs/slurm",
        "partition": "long",
        "template": root / "sbatch" / "template-conda.sh",
        "venv": None,
        "verbose": False,
    }

    parser = ArgumentParser(add_help=False)
    parser.add_argument(
        "-h", "--help", action="store_true", help="show this help message and exit"
    )
    parser.add_argument(
        "--help-md",
        action="store_true",
        help="Show an extended help message as markdown. Can be useful to overwrite "
        + "LAUNCH.md with `$ python launch.py --help-md > LAUNCH.md`",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        help="slurm job name to show in squeue."
        + f" Defaults to {defaults['job_name']}",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="where to write the slurm .out file."
        + f" Defaults to {defaults['outdir']}",
    )
    parser.add_argument(
        "--cpus_per_task",
        type=int,
        help="number of cpus per SLURM task."
        + f" Defaults to {defaults['cpus_per_task']}",
    )
    parser.add_argument(
        "--mem",
        type=str,
        help="memory per node (e.g. 32G)." + f" Defaults to {defaults['mem']}",
    )
    parser.add_argument(
        "--gres",
        type=str,
        help="gres per node (e.g. gpu:1)." + f" Defaults to {defaults['gres']}",
    )
    parser.add_argument(
        "--partition",
        type=str,
        help="slurm partition to use for the job."
        + f" Defaults to {defaults['partition']}",
    )
    parser.add_argument(
        "--modules",
        type=str,
        help="string after 'module load'." + f" Defaults to {defaults['modules']}",
    )
    parser.add_argument(
        "--conda_env",
        type=str,
        help="conda environment name." + f" Defaults to {defaults['conda_env']}",
    )
    parser.add_argument(
        "--venv",
        type=str,
        help="path to venv (without bin/activate)."
        + f" Defaults to {defaults['venv']}",
    )
    parser.add_argument(
        "--code_dir",
        type=str,
        help="cd before running main.py (defaults to here)."
        + f" Defaults to {defaults['code_dir']}",
    )
    parser.add_argument(
        "--jobs",
        type=str,
        help="run file name in external/jobs (without .yaml)."
        + f" Defaults to {defaults['jobs']}",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Don't run just, show what it would have run."
        + f" Defaults to {defaults['dev']}",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print templated sbatch after running it."
        + f" Defaults to {defaults['verbose']}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="skip user confirmation." + f" Defaults to {defaults['force']}",
    )

    known, unknown = parser.parse_known_args()

    cli_script_args = (" " + " ".join(unknown)) if unknown else ""

    args = {k: v for k, v in vars(known).items() if v is not None}

    if args.get("help_md"):
        print_md_help(parser, defaults)
        sys.exit(0)
    if args.get("help"):
        print(parser.format_help())
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

    # find jobs config file in external/jobs as a yaml file
    jobs_conf_path = find_jobs_conf(args)
    # load yaml file as list of dicts. May be empty if jobs_conf_path is None
    job_dicts = load_jobs(jobs_conf_path)
    # No run passed in the CLI args or in the associated yaml file so run the
    # CLI main_args, if any.
    if not job_dicts:
        job_dicts = [{}]

    # Save submitted jobs ids
    job_ids = []

    # A unique datetime identifier for the jobs about to be submitted
    now = now_str()

    if not force and not dev:
        if "y" not in input(f"🚨 Submit {len(job_dicts)} jobs? [y/N] ").lower():
            print("🛑 Aborted")
            sys.exit(0)
        print()

    local_out_dir = root / "external" / "launched_sbatch_scripts"
    if jobs_conf_path is not None:
        local_out_dir = local_out_dir / jobs_conf_path.parent.relative_to(
            root / "external" / "jobs"
        )
    else:
        local_out_dir = local_out_dir / "_other_"

    for i, job_dict in enumerate(job_dicts):
        job_args = defaults.copy()
        job_args = deep_update(job_args, job_dict.pop("slurm", {}))
        job_args = deep_update(job_args, job_dict)
        job_args = deep_update(job_args, args)

        job_args["code_dir"] = str(resolve(job_args["code_dir"]))
        job_args["outdir"] = str(resolve(job_args["outdir"]))
        job_args["venv"] = str(resolve(job_args["venv"]))
        job_args["main_args"] = (
            script_args_dict_to_main_args_str(job_args["script"]) + cli_script_args
        )

        # filter out useless args for the template
        job_args = {k: str(v) for k, v in job_args.items() if k in template_keys}
        # Make sure all the keys in the template are in the args
        if set(template_keys) != set(job_args.keys()):
            print(f"template keys: {template_keys}")
            print(f"template args: {job_args}")
            raise ValueError(
                "template keys != template args (see details printed above)"
            )

        # format template for this run
        templated = template.format(**job_args)

        # set output path for the sbatch file to execute in order to submit the job
        if jobs_conf_path is not None:
            sbatch_path = local_out_dir / f"{jobs_conf_path.stem}_{now}_{i}.sbatch"
        else:
            sbatch_path = local_out_dir / f"{job_args['job_name']}_{now}.sbatch"

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
                + str(outdir / f"{job_args['job_name']}-{job_id}.out")
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
        print(f"\n🚀 Submitted job {i+1}/{len(job_dicts)}")

    # make copy of original yaml conf and append all the sbatch info:
    if jobs_conf_path is not None:
        conf = jobs_conf_path.read_text()
        new_conf_path = local_out_dir / f"{jobs_conf_path.stem}_{now}.yaml"
        new_conf_path.parent.mkdir(parents=True, exist_ok=True)
        conf += "\n# " + jobs_str + "\n"
        new_conf_path.write_text(conf)
        rel = new_conf_path.relative_to(Path.cwd())
        if not dev:
            print(f"   Created summary YAML in ./{rel}")

    if job_ids:
        print(f"   {jobs_str}\n")
