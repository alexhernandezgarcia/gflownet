import sys
from pathlib import Path

from git import Repo

ROOT = Path(__file__).resolve().parent.parent.parent
REPOS = ROOT / "external" / "repos"


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


def require_external_library(name, url, version, pull, remote="origin", verbose=False):
    """
    Install and checkout an external library from a git repository and prepend it to
    sys.path.

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
    """
    repo_path = str(
        install_and_checkout(name, url, version, pull, remote=remote, verbose=verbose)
    )
    verbose and print(f"Prepending {repo_path} to sys.path")
    sys.path.insert(0, repo_path)
