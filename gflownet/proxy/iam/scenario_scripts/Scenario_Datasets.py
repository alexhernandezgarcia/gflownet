import os
from copy import deepcopy as cdc

import gdown
import pandas as pd
import torch
from torch.utils.data import Dataset

DATA_CONFIG = {
    "subsidies_df": {
        "gdrive_id": "1d_xJc1xaG3fYkwXa5rzGTdfpLwIgjAfd",
        "local_path": "scenario_data/subsidies_df.parquet",
    },
    "variables_df": {
        "gdrive_id": "1mvFLaJ6ig89jQaiPqgV4mnVZg1CxOatR",
        "local_path": "scenario_data/variables_df.parquet",
    },
    "keys_df": {
        "gdrive_id": "1ADcM9cIek1fIoNO4ceOgWgzCDFPPKMWC",
        "local_path": "scenario_data/keys_df.parquet",
    },
}


def download_file_from_gdrive(file_id, output_path, filename):
    """
    Download a file from Google Drive using gdown.

    Args:
        file_id: Google Drive file ID
        output_path: Directory to save the file
        filename: Name of the file
    """
    os.makedirs(output_path, exist_ok=True)
    file_full_path = os.path.join(output_path, filename)

    print(f"Downloading {filename}...")
    try:
        gdown.download(
            f"https://drive.google.com/uc?id={file_id}", file_full_path, quiet=False
        )
        print(f"✓ Successfully downloaded {filename}")
        return file_full_path
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        raise


def ensure_data_files_exist(data_config):
    """
    Check if data files exist locally. If not, download from Google Drive.

    Args:
        data_config: Dictionary with file metadata
    """
    for file_key, config in data_config.items():
        local_path = config["local_path"]

        if not os.path.exists(local_path):
            print(f"{local_path} not found. Attempting to download...")
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            download_file_from_gdrive(
                config["gdrive_id"],
                os.path.dirname(local_path),
                os.path.basename(local_path),
            )
        else:
            print(f"✓ {local_path} found locally")


class witch_proc_data(Dataset):
    def __init__(
        self,
        subsidies_parquet="scenario_data/subsidies_df.parquet",
        variables_parquet="scenario_data/variables_df.parquet",
        keys_parquet="scenario_data/keys_df.parquet",
        with_cuda=False,
        scaling_type="original",
        precomputed_scaling_params=None,
        drop_columns=None,
        auto_download=True,
        data_config=None,
    ):
        # Use provided data_config or fall back to default
        if data_config is None:
            data_config = DATA_CONFIG.copy()

        # Override default paths with provided input paths
        data_config["subsidies_df"]["local_path"] = subsidies_parquet
        data_config["variables_df"]["local_path"] = variables_parquet
        data_config["keys_df"]["local_path"] = keys_parquet

        if auto_download:
            ensure_data_files_exist(data_config)

        self.subsidies_parquet = subsidies_parquet
        self.variables_parquet = variables_parquet
        self.keys_parquet = keys_parquet
        self.with_cuda = with_cuda
        self.scaling_type = scaling_type
        if scaling_type not in ["original", "normalization", "maxscale", "maxmin"]:
            raise ValueError(
                "Scaling type must be either original, normalization, or maxscale \nUnknown scaling type: {}".format(
                    scaling_type
                )
            )
        self.precomputed_scaling_params = precomputed_scaling_params

        self.subsidies_df = pd.read_parquet(self.subsidies_parquet)
        self.variables_df = pd.read_parquet(self.variables_parquet)
        self.keys_df = pd.read_parquet(self.keys_parquet)
        self.keys_df["year"] = self.keys_df["year"].astype(int)

        if drop_columns is not None:
            self.subsidies_df = self.subsidies_df.drop(
                columns=drop_columns, errors="ignore"
            )
            self.variables_df = self.variables_df.drop(
                columns=drop_columns, errors="ignore"
            )

        # Replace all NaNs with 0
        self.variables_df = self.variables_df.fillna(0)

        if self.precomputed_scaling_params is None:
            use_computed = False
            self.precomputed_scaling_params = {}
        else:
            use_computed = True

        for col in self.subsidies_df.columns:
            if not use_computed:
                if scaling_type == "normalization":
                    temp_mean = self.subsidies_df[col].mean()
                    temp_std = self.subsidies_df[col].std()
                    if temp_std == 0:
                        print(col)
                    self.precomputed_scaling_params[col] = {
                        "mean": temp_mean,
                        "std": temp_std,
                    }
                elif scaling_type == "maxscale":
                    temp_max = self.subsidies_df[col].max()
                    self.precomputed_scaling_params[col] = {"max": temp_max}
                elif scaling_type == "maxmin":
                    temp_max = self.subsidies_df[col].max()
                    temp_min = self.subsidies_df[col].min()
                    self.precomputed_scaling_params[col] = {
                        "max": temp_max,
                        "min": temp_min,
                    }

            if scaling_type == "normalization":
                self.subsidies_df[col] = (
                    self.subsidies_df[col]
                    - self.precomputed_scaling_params[col]["mean"]
                ) / self.precomputed_scaling_params[col]["std"]
            elif scaling_type == "maxscale":
                self.subsidies_df[col] = (
                    self.subsidies_df[col] / self.precomputed_scaling_params[col]["max"]
                )
            elif scaling_type == "maxmin":
                self.subsidies_df[col] = (
                    self.subsidies_df[col] - self.precomputed_scaling_params[col]["min"]
                ) / (
                    self.precomputed_scaling_params[col]["max"]
                    - self.precomputed_scaling_params[col]["min"]
                )

        for col in self.variables_df.columns:
            if not use_computed:
                if scaling_type == "normalization":
                    temp_mean = self.variables_df[col].mean()
                    temp_std = self.variables_df[col].std()
                    if temp_std == 0:
                        print(col)
                    self.precomputed_scaling_params[col] = {
                        "mean": temp_mean,
                        "std": temp_std,
                    }
                elif scaling_type == "maxscale":
                    temp_max = self.variables_df[col].max()
                    self.precomputed_scaling_params[col] = {"max": temp_max}
                elif scaling_type == "maxmin":
                    temp_max = self.variables_df[col].max()
                    temp_min = self.variables_df[col].min()
                    self.precomputed_scaling_params[col] = {
                        "max": temp_max,
                        "min": temp_min,
                    }

            if scaling_type == "normalization":
                self.variables_df[col] = (
                    self.variables_df[col]
                    - self.precomputed_scaling_params[col]["mean"]
                ) / self.precomputed_scaling_params[col]["std"]
            elif scaling_type == "maxscale":
                self.variables_df[col] = (
                    self.variables_df[col] / self.precomputed_scaling_params[col]["max"]
                )
            elif scaling_type == "maxmin":
                self.variables_df[col] = (
                    self.variables_df[col] - self.precomputed_scaling_params[col]["min"]
                ) / (
                    self.precomputed_scaling_params[col]["max"]
                    - self.precomputed_scaling_params[col]["min"]
                )

        self.variables_names = cdc(self.variables_df.columns)
        self.subsidies_names = cdc(self.subsidies_df.columns)

        self.variables_df = torch.tensor(self.variables_df.values, dtype=torch.float32)
        self.subsidies_df = torch.tensor(self.subsidies_df.values, dtype=torch.float32)

        if with_cuda:
            self.variables_df = self.variables_df.to("cuda")
            self.subsidies_df = self.subsidies_df.to("cuda")

        self.index_map = {
            (
                self.keys_df.loc[i, "gdx"],
                self.keys_df.loc[i, "year"],
                self.keys_df.loc[i, "n"],
            ): i
            for i in range(len(self.keys_df))
        }
        max_year = self.keys_df["year"].max()
        valid_indices = self.keys_df[self.keys_df["year"] != max_year].index

        self.variables_next_df = torch.zeros_like(self.variables_df)
        for idx in valid_indices:
            row = self.keys_df.loc[idx]
            next_idx = self.index_map.get((row.gdx, row.year + 5, row.n))
            self.variables_next_df[idx] = self.variables_df[next_idx]

    def __len__(self):
        # Exclude the last year so that variables_next always exists
        max_year = self.keys_df["year"].max()
        return self.keys_df.loc[self.keys_df["year"] != max_year].shape[0]

    def __getitem__(self, idx):
        variables_current = self.variables_df[idx]
        subsidies_current = self.subsidies_df[idx]

        variables_next = self.variables_next_df[idx]

        return variables_current, subsidies_current, variables_next
