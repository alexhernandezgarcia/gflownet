from typing import Callable, List, Optional, Tuple

import torch
import torch_geometric.data as gd
from torch_geometric.data import Data
from rdkit.Chem import Mol as RDMol

from gflownet.proxy.base import Proxy
from gflownet.proxy.bengio2021flow import load_original_model, mol2graph
from torch import Tensor

# CODE FROM: https://github.com/recursionpharma/gflownet/blob/trunk/src/gflownet/tasks/seh_frag.py 


class SEHTask(Proxy):
    """Sets up a task where the reward is computed using a proxy for the binding energy of a molecule to
    Soluble Epoxide Hydrolases.

    The proxy is pretrained, and obtained from the original GFlowNet paper, see `gflownet.models.bengio2021flow`.

    This setup essentially reproduces the results of the Trajectory Balance paper when using the TB
    objective, or of the original paper when using Flow Matching.
    """

    def __init__(
        self,
        wrap_model: Optional[Callable[[torch.nn.Module], torch.nn.Module]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._wrap_model = wrap_model if wrap_model is not None else (lambda x: x)
        self.models = self._load_task_models()

    def _load_task_models(self):
        model = load_original_model()
        model.to(self.device)
        model = self._wrap_model(model)
        return {"seh": model}


    def compute_reward_from_graph(self, graphs: List[Data]) -> Tensor:
        batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        # batch.to(self.models["seh"].device if hasattr(self.models["seh"], "device") else get_worker_device())
        batch.to(self.device)
        preds = self.models["seh"](batch).reshape((-1,)).data.cpu() / 8
        preds[preds.isnan()] = 0
        return preds.clip(1e-4, 100).reshape((-1,))

    def compute_obj_properties(self, mols: List[RDMol]) -> Tuple[Tensor, Tensor]:
        graphs = [mol2graph(i) for i in mols]
        is_valid = torch.tensor([i is not None for i in graphs]).bool()
        if not is_valid.any():
            return torch.zeros((0, 1)), is_valid

        preds = self.compute_reward_from_graph(graphs).reshape((-1, 1))
        assert len(preds) == is_valid.sum()
        return preds, is_valid
    
    def __call__(self, mols: List[RDMol]) -> Tuple[Tensor, Tensor]:
        # output of the model 
        return self.compute_obj_properties(mols)
