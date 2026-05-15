# CODE FROM https://github.com/koziarskilab/RGFN/blob/main/rgfn/gfns/reaction_gfn/proxies/seh_proxy.py
import gzip
import pickle  # nosec
from pathlib import Path
from typing import Dict, List

import gin
import requests  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.nn import NNConv, Set2Set

from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionState,
    ReactionStateEarlyTerminal,
)
from rgfn.gfns.reaction_gfn.policies.graph_transformer import (
    _chunks,
    mol2graph,
    mols2batch,
)
from rgfn.shared.proxies.cached_proxy import CachedProxyBase

NUM_ATOMIC_NUMBERS = 56  # Number of atoms used in the molecules (i.e. up to Ba)


class MPNNet(nn.Module):
    def __init__(
        self,
        num_feat=14,
        num_vec=3,
        dim=64,
        num_out_per_mol=1,
        num_out_per_stem=105,
        num_out_per_bond=1,
        num_conv_steps=12,
    ):
        super().__init__()
        self.lin0 = nn.Linear(num_feat + num_vec, dim)
        self.num_ops = num_out_per_stem
        self.num_opm = num_out_per_mol
        self.num_conv_steps = num_conv_steps
        self.dropout_rate = 0

        self.act = nn.LeakyReLU()

        net = nn.Sequential(nn.Linear(4, 128), self.act, nn.Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, net, aggr="mean")
        self.gru = nn.GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin3 = nn.Linear(dim * 2, num_out_per_mol)
        self.bond2out = nn.Sequential(
            nn.Linear(dim * 2, dim),
            self.act,
            nn.Linear(dim, dim),
            self.act,
            nn.Linear(dim, num_out_per_bond),
        )

    def forward(self, data, do_dropout=False):
        out = self.act(self.lin0(data.x))
        h = out.unsqueeze(0)
        h = F.dropout(h, training=do_dropout, p=self.dropout_rate)

        for i in range(self.num_conv_steps):
            m = self.act(self.conv(out, data.edge_index, data.edge_attr))
            m = F.dropout(m, training=do_dropout, p=self.dropout_rate)
            out, h = self.gru(m.unsqueeze(0).contiguous(), h.contiguous())
            h = F.dropout(h, training=do_dropout, p=self.dropout_rate)
            out = out.squeeze(0)

        global_out = self.set2set(out, data.batch)
        global_out = F.dropout(global_out, training=do_dropout, p=self.dropout_rate)
        per_mol_out = self.lin3(global_out)  # per mol scalar outputs
        return per_mol_out


def request():
    return requests.get(
        "https://github.com/GFNOrg/gflownet/raw/master/mols/data/pretrained_proxy/best_params.pkl.gz",
        stream=True,
        timeout=30,
    )


def download(location):
    f = request()
    location.parent.mkdir(exist_ok=True)
    with open(location, "wb") as fd:
        for chunk in f.iter_content(chunk_size=128):
            fd.write(chunk)


def load_weights(cache, location):
    if not cache:
        return pickle.load(gzip.open(request().raw))  # nosec

    try:
        gz = gzip.open(location)
    except gzip.BadGzipFile:
        download(location)
        gz = gzip.open(location)
    except FileNotFoundError:
        download(location)
        gz = gzip.open(location)
    return pickle.load(gz)  # nosec


def load_original_model(
    cache=True, location=Path(__file__).parent / "cache" / "bengio2021flow_proxy.pkl.gz"
):
    num_feat = 14 + 1 + NUM_ATOMIC_NUMBERS
    mpnn = MPNNet(
        num_feat=num_feat,
        num_vec=0,
        dim=64,
        num_out_per_mol=1,
        num_out_per_stem=105,
        num_conv_steps=12,
    )

    params = load_weights(cache, location)
    param_map = {
        "lin0.weight": params[0],
        "lin0.bias": params[1],
        "conv.bias": params[3],
        "conv.nn.0.weight": params[4],
        "conv.nn.0.bias": params[5],
        "conv.nn.2.weight": params[6],
        "conv.nn.2.bias": params[7],
        "conv.lin.weight": params[2],
        "gru.weight_ih_l0": params[8],
        "gru.weight_hh_l0": params[9],
        "gru.bias_ih_l0": params[10],
        "gru.bias_hh_l0": params[11],
        "set2set.lstm.weight_ih_l0": params[16],
        "set2set.lstm.weight_hh_l0": params[17],
        "set2set.lstm.bias_ih_l0": params[18],
        "set2set.lstm.bias_hh_l0": params[19],
        "lin3.weight": params[20],
        "lin3.bias": params[21],
    }
    for k, v in param_map.items():
        mpnn.get_parameter(k).data = torch.tensor(v)
    return mpnn


class SEHProxyWrapper:
    def __init__(self, batch_size: int = 64, **kwargs):
        super().__init__(**kwargs)

        self.batch_size = batch_size
        self.model = load_original_model()
        self.model.eval()
        self.device = "cpu"

    @torch.no_grad()
    def compute_scores(self, smiles: List[str]) -> List[float]:
        outputs = []

        for chunk in _chunks(smiles, self.batch_size):
            graphs = []
            for s in chunk:
                mol = Chem.MolFromSmiles(s)
                graph = mol2graph(mol)
                graphs.append(graph)
            batch = mols2batch(graphs).to(self.device)
            output = self.model(batch)
            outputs.append(output)

        output = (torch.cat(outputs)).clip(1e-4, 100).reshape((-1,)).cpu().detach().numpy().tolist()
        return output


@gin.configurable()
class SehMoleculeProxy(CachedProxyBase[ReactionState]):
    def __init__(self, batch_size: int = 128):
        super().__init__()
        self.device = "cpu"
        self.model = SEHProxyWrapper(batch_size=batch_size)
        self.cache = {ReactionStateEarlyTerminal(None): 0.0}

    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True

    def _compute_proxy_output(
        self, states: List[ReactionState]
    ) -> List[Dict[str, float]] | List[float]:
        return self.model.compute_scores([state.molecule.smiles for state in states])

    def set_device(self, device: str, recursive: bool = True):
        self.device = device
        self.model.device = device
        self.model.model.to(device)