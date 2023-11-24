from typing import Optional

from omegaconf import DictConfig, OmegaConf


def parse_policy_config(config: DictConfig, kind: str) -> Optional[DictConfig]:
    """
    Helper for parsing forward/backward policy_config from the given global config.

    Parameters
    ----------
    config : DictConfig
        Global hydra config.

    kind : str
        Type of config, either 'forward' or 'backward'.
    """
    assert kind in ["forward", "backward"]

    policy_config = OmegaConf.create(config.policy)
    policy_config["config"] = config.policy.shared or {}
    policy_config["config"].update(config.policy[kind] or {})

    # Preserve backward compatibility: if neither shared nor forward/backward
    # configs were given, return None instead of an empty config.
    if len(policy_config["config"]) == 0:
        policy_config["config"] = None

    del policy_config.forward
    del policy_config.backward
    del policy_config.shared

    return policy_config
