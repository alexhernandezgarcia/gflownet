"""
Utilities to install and use external libraries in a Python script.

Mainly used to install and checkout external code from a Git repository and prepend it
to ``sys.path`` with :py:func:`~gflownet.utils.libs.require_external_library`.
. This is useful when working with external libraries that are not
available on PyPI or that are not installed in the current environment.
"""

import os
import re
import sys
from pathlib import Path

from git import Repo

ROOT = Path(__file__).resolve().parent.parent.parent
"""Path to the root of the project."""
REPOS = ROOT / "external" / "repos"
"""Path to the directory where external repositories are cloned."""


def is_drac(from_env=True):
    """
    Check if the current environment is a DRAC cluster.

    Parameters
    ----------
    from_env : bool, optional
        Whether to check the environment variables to determine if we are on a DRAC
        cluster, by default True. Otherwise, we check for the presence of a slurm
        configuration file and read the cluster name from it.

    Returns
    -------
    bool
        Whether the current environment is a DRAC cluster.
    """
    # Easy way to check if we are on a drac cluster
    if from_env:
        return bool(os.environ.get("CC_CLUSTER"))

    # more robust to check any slurm cluster
    # not used for now but keeping for future use if needed
    slurm_conf_path = Path("/etc/slurm/slurm.conf")
    if not slurm_conf_path.exists():
        return False
    conf = slurm_conf_path.read_text()
    matches = re.findall(r"ClusterName=(\w+)", conf)
    if not matches:
        return False
    cluster = matches[0]
    return cluster in {"narval", "beluga", "cedar", "graham", "arbutus", "niagara"}


def install_and_checkout(name, url, version, pull, remote="origin", verbose=False):
    """
    Clone a git repository and checkout a specific version (tag or branch) if it exists.

    Parameters
    ----------
    name : str
        The name of the repository to clone. Will be used to name the local directory.
    url : str
        The URL of the repository to clone. Can be a local path or a remote URL.
    version : str
        The version (tag or branch) to checkout.
    pull : bool
        Whether to pull from the remote after checking out the version.
    remote : str, optional
        The name of the remote to pull from, by default "origin".
    verbose : bool, optional
        Whether to print progress messages, by default False.

    Returns
    -------
    Path
        The path to the local (checked-out) repository.

    Raises
    ------
    ValueError
        If the requested version does not exist in the repository.
    """

    repo_path = REPOS / f"{name}__{version}"

    if not repo_path.exists():
        verbose and print(f"Cloning {name} from {url}...", end="", flush=True)
        Repo.clone_from(url, repo_path)
        verbose and print("Remember to handle this library's dependecies manually.")
        pull = False  # no need to pull if we just cloned
        verbose and print("done")

    repo = Repo(repo_path)

    # check version tag exists or branch exists
    if version not in [t.name for t in repo.tags]:
        if version not in [b.name for b in repo.branches]:
            raise ValueError(f"Version {version} not found in {name}")

    # check current branch / tag against requested version and checkout if need be
    if repo.head.is_detached or repo.active_branch.name != version:
        verbose and print(f"Checking out {version}")
        repo.git.checkout(version)

    # pull from branch / tag if requested
    if pull:
        verbose and print(f"Pulling {name}...", end="", flush=True)
        repo.git.pull(remote, version)
        verbose and print("done")

    return repo_path


def require_external_library(
    name,
    url,
    version,
    sub_path=None,
    pull=True,
    fail="raise",
    remote="origin",
    verbose=False,
):
    """
    Clone & checkout external code from a Git repository and prepend it to ``sys.path``.

    If ``pull``, ``fail`` or ``remote`` are set to ``None``, their default values will
    be used.

    Examples
    --------

    >>> conf = {
        "name": "ocp",
        "url": "https://github.com/RolnickLab/ocp",
        "version": "finetuned-and-notag"
    }
    >>> require_external_library(**conf, verbose=True)
    Cloning ocp from https://github.com/RolnickLab/ocp...done
    Remember to handle this library's dependecies manually.
    Checking out finetuned-and-notag
    Prepending .../gflownet/external/repos/ocp__finetuned-and-notag to sys.path

    Parameters
    ----------
    name : str
        The name of the repository to clone. Will be used to name the local directory.
    url : str
        The URL of the repository to clone. Can be a local path or a remote URL.
    version : str
        The version (tag or branch) to checkout.
    sub_path: str, optional
        The sub-path to prepend to sys.path after checking out the library. For instance
        a ``src/`` folder containing the package to use. By default ``None``.
    pull : bool
        Whether to pull from the remote after checking out the version.
    fail : str, optional
        Whether to raise an exception if the library cannot be installed or checked-out.
        Can be either ``"pass"`` or ``"raise"``, by default ``"raise"``.
    remote : str, optional
        The name of the remote to pull from, by default ``"origin"``.
    verbose : bool, optional
        Whether to print progress messages, by default ``False``.

    Raises
    ------
    Exception
        If the library cannot be installed or checked-out and ``fail`` is set to
        ``"raise"``.
    ValueError
        If the requested ``sub_path`` does not exist in the checked-out repository.
    AssertionError
        If the requested ``sub_path`` is not a string or a ``pathlib.Path``.
    """
    if fail is None:
        fail = "raise"
    if remote is None:
        remote = "origin"
    if pull is None:
        pull = True

    if pull and is_drac():
        print(
            "💥 Warning: `pull` is `True` but you are running on a DRAC cluster. "
            "Overriding to `False` because the cluster is not connected to "
            + "the internet."
        )
        pull = False

    try:
        repo_path = install_and_checkout(
            name, url, version, pull, remote=remote, verbose=verbose
        )
    except Exception as e:
        if fail != "pass":
            raise e

    if sub_path:
        assert isinstance(
            sub_path, (str, Path)
        ), "`sub_path` must be a string or a `pathlib.Path`"
        repo_path = repo_path / sub_path
        if not repo_path.exists():
            raise ValueError(f"Could not find {sub_path} in {repo_path}")

    repo_path = str(repo_path)
    verbose and print(f"Prepending {repo_path} to sys.path")
    sys.path.insert(0, repo_path)
