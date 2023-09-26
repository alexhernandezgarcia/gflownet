import pandas as pd
from pathlib import Path

# method = "gfn-ff"
# method = "torchani"
method = "xtb"

base_path = Path('/home/mila/a/alexandra.volokhova/projects/gflownet/results/conformer/metrics/')
df = pd.read_csv(base_path / f"gfn_conformers_v2_{method}_metrics.csv", index_col="smiles")
dftop = pd.read_csv(base_path / f"gfn_conformers_v2_{method}_top_k_metrics.csv", index_col="smiles") 
rd = pd.read_csv(
    base_path / "rdkit_samples_target_smiles_first_batch_2023-09-24_21-15-42_metrics.csv",
    index_col="smiles",
)
rdc = pd.read_csv(
    base_path / "rdkit_cluster_samples_target_smiles_first_batch_2023-09-24_21-16-23_metrics.csv",
    index_col="smiles",
)
rd = rd.loc[df.index]
rdc = rdc.loc[df.index]
dftop = dftop.loc[df.index]
df[f"{method}_cov"] = df["cov"]
df[f"{method}_mat"] = df["mat"]
df[f"{method}_tk_cov"] = dftop["cov"]
df[f"{method}_tk_mat"] = dftop["mat"]  
df["rdkit_cov"] = rd["cov"]
df["rdkit_mat"] = rd["mat"]
df["rdkit_cl_cov"] = rdc["cov"]
df["rdkit_cl_mat"] = rdc["mat"]
df = df.sort_values("n_tas")
df.to_csv('./merged_metrics_{}.csv'.format(method))
df = df[
    [
        f"{method}_cov",
        f"{method}_mat",
        f"{method}_tk_cov",
        f"{method}_tk_mat",
        "rdkit_cov",
        "rdkit_mat",
        "rdkit_cl_cov",
        "rdkit_cl_mat",
    ]
]
print("Mean")
print(df.mean())
print("Median")
print(df.median())
print("Var")
print(df.var())