import datetime
import re
import sys
from argparse import ArgumentParser
from copy import deepcopy
from os import popen
from os.path import expandvars
from pathlib import Path
from textwrap import dedent

from git import Repo
from yaml import safe_load

ROOT = Path(__file__).resolve().parent.parent

GIT_WARNING = True

HELP = dedent(
    """
    ## 🥳 User guide

    In a word, use `launch.py` to fill in an sbatch template and submit either
    a single job from the command-line, or a list of jobs from a `yaml` file.

    Examples:

    ```bash
    # using default job configuration, with script args from the command-line:
    $ python mila/launch.py user=$USER logger.do.online=False

    # overriding the default job configuration and adding script args:
    $ python mila/launch.py --template=mila/sbatch/template-venv.sh \\
        --venv='~/.venvs/gfn' \\
        --modules='python/3.7 cuda/11.3' \\
        user=$USER logger.do.online=False

    # using a yaml file to specify multiple jobs to run:
    $ python mila/launch.py --jobs=jobs/comp-sg-lp/v0" --mem=32G
    ```

    ### 🤓 How it works

    1. All experiment files should be in `external/jobs`
        1. Note that all the content in `external/` is **ignored by git**
    2. You can nest experiment files infinitely, let's say you work on crystals and call your experiment `explore-losses.yaml` then you could put your config in `external/jobs/crystals/explore-losses.yaml`
    3. An experiment file contains 2 main sections:
        1. `shared:` contains the configuration that will be, you guessed it, shared across jobs.
        2. `jobs:` lists configurations for the SLURM jobs that you want to run. The `shared` configuration will be loaded first, then updated from the `run`'s.
    4. Both `shared` and `job` dicts contain (optional) sub-sections:
        1. `slurm:` contains what's necessary to parameterize the SLURM job
        2. `script:` contains a dict version of the command-line args to give `main.py`

        ```yaml
        script:
          gflownet:
            optimizer:
              lr: 0.001

        # is equivalent to
        script:
          gflownet.optimizer.lr: 0.001

        # and will be translated to
        python main.py gflownet.optimizer.lr=0.001
        ```

    5. Launch the SLURM jobs with `python mila/launch.py --jobs=crystals/explore-losses`
        1. `launch.py` knows to look in `external/jobs/` and add `.yaml` (but you can write `.yaml` yourself)
        2. You can overwrite anything from the command-line: the command-line arguments have the final say and will overwrite all the jobs' final dicts. Run mila/`python mila/launch.py -h` to see all the known args.
        3. You can also override `script` params from the command-line: unknown arguments will be given as-is to `main.py`. For instance `python mila/launch.py --jobs=crystals/explore-losses --mem=32G env.some_param=value` is valid
    6. `launch.py` loads a template (`mila/sbatch/template-conda.sh`) by default, and fills it with the arguments specified, then writes the filled template in `external/launched_sbatch_scripts/crystals/` with the current datetime and experiment file name.
    7. `launch.py` executes `sbatch` in a subprocess to execute the filled template above
    8. A summary yaml is also created there, with the exact experiment file and appended `SLURM_JOB_ID`s returned by `sbatch`

    ### 📝 Case-study

    Let's study the following example:

    ```
    $ python mila/launch.py --jobs=crystals/explore-losses --mem=64G

    🗂 Using run file: ./external/jobs/crystals/explore-losses.yaml

    🚨 Submit 3 jobs? [y/N] y

      🏷  Created ./external/launched_sbatch_scripts/example_20230613_194430_0.sbatch
      ✅  Submitted batch job 3301572

      🏷  Created ./external/launched_sbatch_scripts/example_20230613_194430_1.sbatch
      ✅  Submitted batch job 3301573

      🏷  Created ./external/launched_sbatch_scripts/example_20230613_194430_2.sbatch
      ✅  Submitted batch job 3301574


    🚀 Submitted job 3/3
    Created summary YAML in ./external/launched_sbatch_scripts/example_20230613_194430.yaml
    All jobs submitted: 3301572 3301573 3301574
    ```

    Say the file `./external/jobs/crystals/explore-losses.yaml` contains:

    ```yaml
    # Contents of external/jobs/crystals/explore-losses.yaml

    {yaml_example}
    ```

    Then the launch command-line ^ will execute 3 jobs with the following configurations:

    ```bash
    python main.py user=$USER +experiments=neurips23/crystal-comp-sg-lp.yaml gflownet=flowmatch gflownet.optimizer.lr=0.0001

    python main.py user=$USER +experiments=neurips23/crystal-comp-sg-lp.yaml gflownet=flowmatch gflownet.optimizer.lr=0.0001 gflownet.policy.backward=None

    python main.py user=$USER +experiments=neurips23/crystal-comp-sg-lp.yaml gflownet=trajectorybalance gflownet.optimizer.lr=0.0001
    ```

    And their SLURM configuration will be similar as the `shared.slurm` params, with the following differences:

    1. The second job will have `partition: unkillable` instead of the default (`long`).
    2. They will all have `64G` of memory instead of the default (`32G`) because the `--mem=64G` command-line
        argument overrides everything.

    ## Updating the launcher

    When updating the launcher, you should:

    1. Update this markdown text **in launch.py:HELP** (do not edit this `LAUNCH.md`)
    2. Run `$ python mila/launch.py --help-md > LAUNCH.md` to update this `LAUNCH.md` from the new `launch.py:HELP` text, new flags etc.
    """.format(
        yaml_example="\n".join(
            [
                # need to indend those lines because of dedent()
                "    " + l if i else l  # first line is already indented
                for i, l in enumerate(
                    (ROOT / "mila/sbatch/example-jobs.yaml")
                    .read_text()
                    .splitlines()[6:]  # ignore first lines which are just comments
                )
            ]
        )
    )
)


def resolve(path):
    """
    Resolves a path with environment variables and user expansion.
    All paths will end up as absolute paths.

    Parameters
    ----------
    path : str | Path
        The path to resolve

    Returns
    -------
    Path
        resolved path
    """
    if path is None:
        return None
    path = str(path).replace("$root", str(ROOT))
    return Path(expandvars(path)).expanduser().resolve()


def now_str():
    """
    Returns a string with the current date and time.
    Eg: "20210923_123456"

    Returns
    -------
    str
        current date and time
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def load_jobs(yaml_path):
    """
    Loads a yaml file with run configurations and turns it into a list of jobs.

    Example yaml file:

    .. code-block:: yaml

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

    Parameters
    ----------
    yaml_path : str | Path
        Where to fine the yaml file

    Returns
    -------
    list[dict]
        List of run configurations as dicts
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
    local_out_dir = ROOT / "external" / "launched_sbatch_scripts"
    if not args.get("jobs"):
        return None, local_out_dir / "_other_"

    if resolve(args["jobs"]).is_file():
        assert args["jobs"].endswith(".yaml") or args["jobs"].endswith(
            ".yml"
        ), "jobs file must be a yaml file"
        jobs_conf_path = resolve(args["jobs"])
        local_out_dir = local_out_dir / jobs_conf_path.parent.name
    else:
        if args["jobs"].endswith(".yaml"):
            args["jobs"] = args["jobs"][:-5]
        if args["jobs"].endswith(".yml"):
            args["jobs"] = args["jobs"][:-4]
        if args["jobs"].startswith("external/"):
            args["jobs"] = args["jobs"][9:]
        if args["jobs"].startswith("jobs/"):
            args["jobs"] = args["jobs"][5:]
        yamls = [
            str(y) for y in (ROOT / "external" / "jobs").glob(f"**/{args['jobs']}.y*ml")
        ]
        if len(yamls) == 0:
            raise ValueError(
                f"Could not find {args['jobs']}.y(a)ml in ./external/jobs/"
            )
        if len(yamls) > 1:
            print(">>> Warning: found multiple matches:\n  •" + "\n  •".join(yamls))
        jobs_conf_path = Path(yamls[0])
        local_out_dir = local_out_dir / jobs_conf_path.parent.relative_to(
            ROOT / "external" / "jobs"
        )
    print("🗂  Using jobs file: ./" + str(jobs_conf_path.relative_to(Path.cwd())))
    print()
    return jobs_conf_path, local_out_dir


def quote(value):
    v = str(value)
    v = v.replace("(", r"\(").replace(")", r"\)")
    if " " in v or "=" in v:
        v = v.replace('"', r"\"")
        if "'" in v:
            v = v.replace("'", r"\'")
        v = f'"{v}"'
    return v


def script_dict_to_main_args_str(script_dict, is_first=True, nested_key=""):
    """
    Recursively turns a dict of script args into a string of main.py args
    as ``nested.key=value`` pairs

    Parameters
    ----------
    script_dict : dict
        script dictionary of args
    is_first : bool, optional
        whether this is the first call in the recursion
    nested_key : str, optional
        prefix to add to the keys as ``nested.key``

    Returns
    -------
    str
        string of main.py args (eg: ``"key=value nested.key2=value2"``)
    """
    if not isinstance(script_dict, dict):
        candidate = f"{nested_key}={quote(script_dict)}"
        if candidate.count("=") > 1 or " " in candidate:
            assert "'" not in candidate, """Keys cannot contain ` ` and `'` and `=` """
            candidate = f"'{candidate}'"
        return candidate + " "
    new_str = ""
    for k, v in script_dict.items():
        if k == "__value__":
            value = str(v)
            if " " in value:
                value = f"'{value}'"
            candidate = f"{nested_key}={quote(v)} "
            if candidate.count("=") > 1:
                assert (
                    "'" not in candidate
                ), """Keys cannot contain ` ` and `'` and `=` """
                candidate = f"'{candidate}'"
            new_str += candidate
            continue
        new_key = k if not nested_key else nested_key + "." + str(k)
        new_str += script_dict_to_main_args_str(v, nested_key=new_key, is_first=False)
    if is_first:
        new_str = new_str.strip()
    return new_str


def deep_update(a, b, path=None, verbose=None):
    """
    https://stackoverflow.com/questions/7204805/how-to-merge-dictionaries-of-dictionaries/7205107#7205107

    Parameters
    ----------
    a : dict
        dict to update
    b : dict
        dict to update from

    Returns
    -------
    dict
        updated copy of a
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
    print("In the following, `$root` refers to the root of the current repository.\n")
    print("```sh")
    print(parser.format_help())
    print("```\n")
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


def ssh_to_https(url):
    """
    Converts a ssh git url to https.
    Eg:
    """
    if "https://" in url:
        return url
    if "git@" in url:
        path = url.split(":")[1]
        return f"https://github.com/{path}"
    raise ValueError(f"Could not convert {url} to https")


def code_dir_for_slurm_tmp_dir_checkout(git_checkout, is_private=False):
    global GIT_WARNING

    repo = Repo(ROOT)
    if git_checkout is None:
        git_checkout = repo.active_branch.name
        if GIT_WARNING:
            print("💥 Git warnings:")
            print(
                f"  • `git_checkout` not provided. Using current branch: {git_checkout}"
            )
        # warn for uncommitted changes
        if repo.is_dirty() and GIT_WARNING:
            print(
                "  • Your repo contains uncommitted changes. "
                + "They will *not* be available when cloning happens within the job."
            )
        if GIT_WARNING and "y" not in input("Continue anyway? [y/N] ").lower():
            print("🛑 Aborted")
            sys.exit(0)
        GIT_WARNING = False

    repo_url = ssh_to_https(repo.remotes.origin.url) if not is_private else str(ROOT)
    repo_name = repo_url.split("/")[-1].split(".git")[0]

    return dedent(
        """\
        $SLURM_TMPDIR
        git clone {git_url} tmp-{repo_name}
        cd tmp-{repo_name}
        {git_checkout}
        echo "Current commit: $(git rev-parse HEAD)"
    """
    ).format(
        git_url=repo_url,
        git_checkout=f"git checkout {git_checkout}" if git_checkout else "",
        repo_name=repo_name,
    )


def clean_sbatch_params(templated):
    """
    Removes all SBATCH params that have an empty value.

    Args:
        templated (str): templated sbatch file

    Returns:
        str: cleaned sbatch file
    """
    new_lines = []
    for line in templated.splitlines():
        if not line.startswith("#SBATCH"):
            new_lines.append(line)
            continue
        if "=" not in line:
            new_lines.append(line)
            continue
        if line.split("=")[1].strip():
            new_lines.append(line)
    return "\n".join(new_lines)


if __name__ == "__main__":
    defaults = {
        "code_dir": "$root",
        "conda_env": "gflownet",
        "cpus_per_task": 2,
        "dry-run": False,
        "force": False,
        "git_checkout": None,
        "gres": "",
        "is_private": False,
        "job_name": "gflownet",
        "jobs": None,
        "main_args": None,
        "mem": "32G",
        "modules": "anaconda/3 cuda/11.3",
        "outdir": "$SCRATCH/gflownet/logs/slurm",
        "partition": "long",
        "template": "$root/mila/sbatch/template-conda.sh",
        "time": "",
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
        + "LAUNCH.md with `$ python mila/launch.py --help-md > LAUNCH.md`",
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
        "--time",
        type=str,
        help="wall clock time limit (e.g. 2-12:00:00). "
        + "See: https://slurm.schedmd.com/sbatch.html#OPT_time"
        + f" Defaults to {defaults['time']}",
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
        "--template",
        type=str,
        help="path to sbatch template." + f" Defaults to {defaults['template']}",
    )
    parser.add_argument(
        "--code_dir",
        type=str,
        help="cd before running main.py (defaults to here)."
        + f" Defaults to {defaults['code_dir']}",
    )
    parser.add_argument(
        "--git_checkout",
        type=str,
        help="Branch or commit to checkout before running the code."
        + " This is only used if --code_dir='$SLURM_TMPDIR'. If not specified, "
        + " the current branch is used."
        + f" Defaults to {defaults['git_checkout']}",
    )
    parser.add_argument(
        "--jobs",
        type=str,
        help="jobs (nested) file name in external/jobs (with or without .yaml)."
        + " Or an absolute path to a yaml file anywhere"
        + f" Defaults to {defaults['jobs']}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't run just, show what it would have run."
        + f" Defaults to {defaults['dry-run']}",
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
    parser.add_argument(
        "--is_private",
        action="store_true",
        help="Whether the code is private (i.e. not on github or private repo)."
        + "In this case, the code is cloned to $SLURM_TMPDIR "
        + "from the current local repo path."
        + f" Defaults to {defaults['is_private']}",
    )

    known, unknown = parser.parse_known_args()

    cli_script_args = " ".join(unknown) if unknown else ""

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

    # in dry run mode: no mkdir, no sbatch etc.
    dry_run = args.get("dry_run")

    # in force mode, no confirmation is asked
    force = args.get("force", defaults["force"])

    # where to write the slurm output file
    outdir = resolve(args.get("outdir", defaults["outdir"]))
    if not dry_run:
        outdir.mkdir(parents=True, exist_ok=True)

    # find jobs config file in external/jobs as a yaml file
    jobs_conf_path, local_out_dir = find_jobs_conf(args)
    # load yaml file as list of dicts. May be empty if jobs_conf_path is None
    job_dicts = load_jobs(jobs_conf_path)
    # No run passed in the CLI args or in the associated yaml file so run the
    # CLI main_args, if any.
    if not job_dicts:
        job_dicts = [{}]

    # Save submitted jobs ids
    job_ids = []
    job_out_files = []

    # A unique datetime identifier for the jobs about to be submitted
    now = now_str()

    if not force and not dry_run:
        if "y" not in input(f"🚨 Submit {len(job_dicts)} jobs? [y/N] ").lower():
            print("🛑 Aborted")
            sys.exit(0)
        print()

    for i, job_dict in enumerate(job_dicts):
        job_args = defaults.copy()
        job_args = deep_update(job_args, job_dict.pop("slurm", {}))
        job_args = deep_update(job_args, job_dict)
        job_args = deep_update(job_args, args)

        job_args["code_dir"] = (
            str(resolve(job_args["code_dir"]))
            if "SLURM_TMPDIR" not in job_args["code_dir"]
            else code_dir_for_slurm_tmp_dir_checkout(
                job_args.get("git_checkout"), args.get("is_private")
            )
        )
        job_args["outdir"] = str(resolve(job_args["outdir"]))
        job_args["venv"] = str(resolve(job_args["venv"]))
        job_args["main_args"] = script_dict_to_main_args_str(job_args.get("script", {}))
        if job_args["main_args"] and cli_script_args:
            job_args["main_args"] += " "
        job_args["main_args"] += cli_script_args

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
        templated = clean_sbatch_params(templated)

        # set output path for the sbatch file to execute in order to submit the job
        if jobs_conf_path is not None:
            sbatch_path = local_out_dir / f"{jobs_conf_path.stem}_{now}_{i}.sbatch"
        else:
            sbatch_path = local_out_dir / f"{job_args['job_name']}_{now}.sbatch"

        if not dry_run:
            # make sure the sbatch file parent directory exists
            sbatch_path.parent.mkdir(parents=True, exist_ok=True)
            # write template
            sbatch_path.write_text(templated)
            print()
            # Submit job to SLURM
            out = popen(f"sbatch {sbatch_path}").read().strip()
            # Identify printed-out job id
            job_id = re.findall(r"Submitted batch job (\d+)", out)[0]
            job_ids.append(job_id)
            print("  ✅ " + out)
            # Rename sbatch file with job id
            parts = sbatch_path.stem.split(f"_{now}")
            new_name = f"{parts[0]}_{job_id}_{now}.sbatch"
            sbatch_path = sbatch_path.rename(sbatch_path.parent / new_name)
            print(f"  🏷  Created ./{sbatch_path.relative_to(Path.cwd())}")
            # Write job ID & output file path in the sbatch file
            job_output_file = str(outdir / f"{job_args['job_name']}-{job_id}.out")
            job_out_files.append(job_output_file)
            print("  📝  Job output file will be: " + job_output_file)
            templated += (
                "\n# SLURM_JOB_ID: "
                + job_id
                + "\n# Output file: "
                + job_output_file
                + "\n"
            )
            sbatch_path.write_text(templated)

        # final prints for dry_run & verbose mode
        if dry_run or args.get("verbose"):
            if dry_run:
                print("\nDRY RUN: would have writen in sbatch file:", str(sbatch_path))
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
        conf += (
            "\n# Job Output files:\n#"
            + "\n#".join([f"  • {f}" for f in job_out_files])
            + "\n"
        )
        wandb_query = f"({'|'.join(job_ids)})"
        conf += f"\n# Wandb RegEx query:\n#  • {wandb_query}\n"
        scancel = f"scancel {' '.join(job_ids)}"
        conf += f"\n# Cancel all jobs:\n#  • {scancel}\n"
        rel = new_conf_path.relative_to(Path.cwd())
        if not dry_run:
            new_conf_path.write_text(conf)
            print(f"   Created summary YAML in {rel}")

    if job_ids:
        print(f"   {jobs_str}\n")
