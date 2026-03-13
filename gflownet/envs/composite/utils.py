from typing import Iterable

from gflownet.envs.base import GFlowNetEnv
from gflownet.envs.composite.setfix import SetFix
from gflownet.envs.composite.setflex import SetFlex


def make_set(
    is_flexible: bool,
    subenvs: Iterable[GFlowNetEnv] = None,
    max_elements: int = None,
    envs_unique: Iterable[GFlowNetEnv] = None,
    do_random_subenvs: bool = False,
    **kwargs,
):
    """
    Factory method to create SetFix or SetFlex classes depending on the input
    is_flexible.

    This method mimics conditional inheritance.

    Parameters
    ----------
    is_flexible : bool
        If True, return a SetFlex environment. If False, return a SetFix environment.
    """
    # If is_flexible is False, subenvs must be defined.
    if not is_flexible and subenvs is None:
        raise ValueError("subenvs must be defined to use the SetFix")
    # If is_flexible is True, then max_elements must be defined.
    if is_flexible and max_elements is None:
        raise ValueError("max_elements must be defined to use the SetFlex")

    if is_flexible:
        return SetFlex(
            max_elements=max_elements,
            envs_unique=envs_unique,
            subenvs=subenvs,
            do_random_subenvs=do_random_subenvs,
            **kwargs,
        )
    else:
        return SetFix(subenvs=subenvs, **kwargs)
