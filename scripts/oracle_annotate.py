"""
Annotates a data set with an oracle
"""
import hydra
import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf
from oracle import Oracle


@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig):
    # Make cfg root the specific config of this script
    cfg = cfg.oracle_annotate
    print(OmegaConf.to_yaml(cfg))
    if cfg.env_id == "aptamers":
        oracle = Oracle(
            oracle=cfg.oracle,
        )
    elif cfg.env_id == "grid":
        oracle = None
    else:
        raise NotImplementedError
    # Data set
    df_input = pd.read_csv(cfg.input_csv, index_col=0)
    # Query oracles
    energies = oracle.score(df_input.samples.values)
    # Build output CSV
    if isinstance(cfg.oracle, (list, ListConfig)):
        oracles = [el.replace("nupack ", "") for el in cfg.oracle]
    if cfg.output_csv:
        if isinstance(energies, dict):
            energies.update(
                {"samples": df_input.samples.values, "energies": energies[oracles[0]]}
            )
            df = pd.DataFrame(energies)
        else:
            df = pd.DataFrame(
                {"samples": df_input.samples.values, "energies": energies}
            )
        df.to_csv(cfg.output_csv)


if __name__ == "__main__":
    main()
