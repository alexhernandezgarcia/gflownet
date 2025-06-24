import sys

import pytest
from utils_for_tests import REPO_ROOT, ch_tmpdir, load_base_test_config

sys.path.append(str(REPO_ROOT))

from gflownet.utils.common import gflownet_from_config


@pytest.fixture
def config_for_tests(request):
    """
    Load the base test configuration in config/tests.yaml.

    Simulate Hydra command-line args with overrides.

    Examples
    --------
    ```python
    @pytest.mark.parametrize(
        "config_for_tests,something_else",
        [
            (["env=grid", "env.buffer.test=None"], 1),
            (["env=ctorus"], 2),
        ],
        indirect=["config_for_tests"],
    )
    def test_my_func(config_for_tests, something_else):
        ...
    ```

    Parameters
    ----------
    request : FixtureRequest
        The request object from pytest. Its `param` attribute is used to pass the
        argument to the fixture.

    Returns
    -------
    OmegaConf
        The configuration as an OmegaConf object with the Hydra overrides applied.
    """
    overrides = []
    if hasattr(request, "param") and request.param is not None:
        overrides = request.param
        if not isinstance(overrides, list):
            overrides = [overrides]
        assert isinstance(overrides, list), "Overrides must be a list."
        assert all(
            isinstance(ov, str) for ov in overrides
        ), "Overrides must be a list of string."

    config = load_base_test_config(overrides=overrides)
    return config


@pytest.fixture
def gflownet_for_tests(config_for_tests):
    """
    Create a GFlowNet object from the configuration for tests.

    This is a generator so the code after `yield` is executed at the end of the test
    which uses this fixture (akin to a `finally` block in a `try` statement or a
    unittest `tearDown` method).

    By default, the execution is moved to a temporary directory to avoid polluting the
    current directory with files written by the GFlowNetAgent.

    Set the `disable` parameter to `True` to avoid moving the execution to a temporary
    directory (for example, when developing tests and wanting to inspect the files).

    Parameters
    ----------
    config_for_tests : OmegaConf
        The configuration for the GFlowNetAgent to be created.

    Yields
    ------
    GFlowNetAgent
        The loaded GFlowNetAgent object.
    """
    # Move execution to a temporary directory if disable is not True
    with ch_tmpdir(disable=False) as tmpdir:
        print(f"Current GFlowNetAgent execution directory: {tmpdir}")
        gfn = gflownet_from_config(config_for_tests)
        yield gfn

    # Any teardown (=post-test) code goes here
    pass
