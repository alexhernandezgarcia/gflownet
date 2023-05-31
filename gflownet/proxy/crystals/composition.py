import pickle
from pathlib import Path

import numpy as np
import torch
from torchtyping import TensorType

from gflownet.proxy.base import Proxy

PROTONS_NUMBER_COUNTS = None


def _read_protons_number_counts():
    global PROTONS_NUMBER_COUNTS
    if PROTONS_NUMBER_COUNTS is None:
        with open(
            Path(__file__).parent / "number_of_protons_counts.pkl", "rb"
        ) as handle:
            PROTONS_NUMBER_COUNTS = pickle.load(handle)
    return PROTONS_NUMBER_COUNTS


class CompositionMPFrequency(Proxy):
    """
    A class that calculates the frequency of the number of protons in the composition. 
    The frequency is based on stable crystals from Materials Project (MP)

    Args:
        normalize (bool, optional): Indicates whether to normalize the frequency counts. Defaults to True.
        **kwargs: Additional keyword arguments to be passed to the parent class.

    Attributes:
        counts_dict (dict): A dictionary containing the counts of the number of protons in MP.
        normalize (bool): Indicates whether to normalize the frequency counts.
        counts (torch.Tensor): A tensor containing the frequency counts for each number of protons.
        atomic_numbers (torch.Tensor): A tensor containing the atomic numbers of the elements.
        norm (float): Normalization factor for the frequency counts.
    """
    def __init__(self, normalize: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.counts_dict = _read_protons_number_counts()
        self.normalize = normalize

    def _get_max_protons_state(self, env):
        """
        Calculates the state of the environment with the maximum number of protons.
        """
        # set all the required elements to the min required number
        max_protons_state = env.source.copy()
        for elem in env.required_elements:
            max_protons_state[env.elem2idx[elem]] = max(1, env.min_atom_i)
        
        # satisfy requirement for min different elements by 
        # adding the heviest available atoms (with largest number of protons)
        n_diff_elem = len(np.nonzero(max_protons_state)[0])
        for idx, counts in reversed(list(enumerate(max_protons_state))):
            if n_diff_elem >= env.min_diff_elem:
                break
            if counts == 0:
                max_protons_state[idx] = max(1, env.min_atom_i)
                n_diff_elem += 1

        # add more heavy atoms untill reaching max_atoms or max_diff_elem or max_atom_i
        for idx in range(len(max_protons_state)):
            if (
                max_protons_state[-1 - idx] == 0
                and env.get_n_elements(max_protons_state) < env.max_diff_elem
            ) or max_protons_state[-1 - idx] != 0:
                while (
                    env.get_n_atoms(max_protons_state) < env.max_atoms
                    and max_protons_state[-1 - idx] < env.max_atom_i
                ):
                    max_protons_state[-1 - idx] += 1
        return max_protons_state

    def get_max_n_protons(self, env):
        """
        Calculates the maximum reachiable number of protons in a composition in the given enviroment.
        """
        max_protons_state = self._get_max_protons_state(env)
        max_protons_number = 0
        for idx, count in enumerate(max_protons_state):
            max_protons_number += env.idx2elem[idx] * count
        return max_protons_number

    def setup(self, env):
        """
        Sets up the necessary attributes for frequency calculation.
        """
        mpn = self.get_max_n_protons(env)
        # index in self.counts corresponds to the number of protons in the composition
        # (nth position is n protons)
        self.counts = torch.zeros(mpn + 1, device=self.device, dtype=torch.int16)
        for idx in range(mpn + 1):
            if idx in self.counts_dict.keys():
                self.counts[idx] = self.counts_dict[idx]

        self.atomic_numbers = torch.tensor(
            env.elements, device=self.device, dtype=torch.int16
        )
        if self.normalize:
            self.norm = -1.0 * torch.sum(self.counts)
        else:
            self.norm = -1.0

    def __call__(self, states: TensorType["batch", "state"]) -> TensorType["batch"]:
        """
        Calculates the frequency of the number of protons in the given states.
        """
        return self.counts[torch.sum(states * self.atomic_numbers, dim=1)] / self.norm
