import numpy as np
import numpy.typing as npt
import torch
from nupack import *

from gflownet.proxy.base import Proxy


class Aptamer(Proxy):
    """
    DNA Aptamer oracles
    """

    def __init__(self, oracle_id, norm, cost, **kwargs):
        super().__init__(**kwargs)
        self.type = oracle_id
        self.norm = norm
        self.cost = cost

    def setup(self, env, norm=True):
        self.max_seq_length = env.max_seq_length

    def _length(self, x):
        if self.norm:
            return -1.0 * np.sum(x, axis=1) / self.max_seq_length
        else:
            return -1.0 * np.sum(x, axis=1)

    def __call__(self, sequences):
        if self.type == "length":
            return self._length(sequences)
        elif self.type == "pairs":
            self.function = self._func_pairs
            return self._nupack(sequences)
        elif self.type == "energy":
            self.function = self._func_energy
            return self._nupack(sequences)
        else:
            raise NotImplementedError

    def _nupack(self, sequences):
        """
        args:
            inputs: list of arrays in desired format interpretable by oracle
        returns:
            array of scores
        function:
            creates the complex set and calls the desired nupack function
        """
        temperature = 310.0  # Kelvin
        ionicStrength = 1.0  # molar
        strandList = []
        comps = []
        i = -1
        for sequence in sequences:
            i += 1
            strandList.append(Strand(sequence, name="strand{}".format(i)))
            comps.append(Complex([strandList[-1]], name="comp{}".format(i)))

        set = ComplexSet(
            strands=strandList, complexes=SetSpec(max_size=1, include=comps)
        )
        model1 = Model(material="dna", celsius=temperature - 273, sodium=ionicStrength)
        results = complex_analysis(set, model=model1, compute=["mfe"])

        energy = self.function(sequences, results, comps)

        return torch.tensor(energy, device=self.device, dtype=self.float)

    def _func_energy(self, sequences, results, comps):
        energies = np.zeros(len(sequences))
        for i in range(len(energies)):
            energies[i] = results[comps[i]].mfe[0].energy
        return energies

    def _func_pairs(self, sequences, results, comps):
        ssStrings = np.zeros(len(sequences), dtype=object)
        for i in range(len(ssStrings)):
            ssStrings[i] = str(results[comps[i]].mfe[0].structure)
        nPairs = np.asarray([ssString.count("(") for ssString in ssStrings]).astype(int)
        return -nPairs
