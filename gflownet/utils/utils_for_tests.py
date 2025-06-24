import contextlib
import os
import tempfile
from pathlib import Path

from hydra import compose, initialize


def find_root():
    """
    Find the root of the repository by looking for the .git folder and the gflownet
    folder in the parent directories.

    Returns
    -------
    Path
        The root of the repository as a pathlib.Path object.

    Raises
    ------
    RuntimeError
        If the root of the repository could not be found.
    """
    path = Path(__file__).resolve()
    while not (
        (path / ".git").exists()
        and (path / "gflownet").exists()
        and (path / "config").exists()
        and path != path.parent
    ):
        path = path.parent
    if path == path.parent:
        raise RuntimeError("Could not find root of the repository")
    return path


REPO_ROOT = find_root()


def load_base_test_config(overrides=[]):
    """
    Load the base test configuration in config/tests.yaml.
    Simulate command-line args with overrides.

    Examples
    --------
    >>> load_base_test_config(["env=grid", "env.buffer.test=None"])

    Parameters
    ----------
    overrides : list[str], optional
        A list of overrides for the configuration, by default [].

    Returns
    -------
    OmegaConf
        The configuration as an OmegaConf object.
    """
    with initialize(
        version_base="1.1",
        config_path=os.path.relpath(
            str(REPO_ROOT / "config"), start=str(Path(__file__).parent)
        ),
        job_name="xxx",
    ):
        config = compose(config_name="tests", overrides=overrides)
    return config


@contextlib.contextmanager
def ch_tmpdir(disable=False):
    """
    Change to a temporary directory and change back to the original directory when the
    context manager exits.

    Parameters
    ----------
    disable : bool, optional
        Whether to disable the context manager, by default False.

    Yields
    ------
    str
        The path of the temporary directory (if not disabled) or the original directory.
    """
    d = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdirname:
        if not disable:
            os.chdir(tmpdirname)
        try:
            yield tmpdirname if not disable else d

        finally:
            os.chdir(d)
