from gflownet.proxy.base import Proxy
import numpy as np
import numpy.typing as npt
from nupack import *
import torch

class Aptamers(Proxy):
    """
    DNA Aptamer oracles
    """

    def __init__(self, oracle_id, norm, cost, **kwargs):
        super().__init__(**kwargs)
        self.type = oracle_id
        self.norm = norm
        self.cost = cost
        if self.type == "length":
            self.__call__ = self._length
        elif self.type == "pairs":
            # self.__call__ = self._nupack
            self.function = self._func_pairs
        elif self.type == "energy":
            # self.__call__ = self._nupack
            self.function = self._func_energy
        else:
            raise NotImplementedError
        # self.inverse_lookup = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    def setup(self, max_seq_length, norm=True):
        self.max_seq_length = max_seq_length
    
    def _length(self, x):
        if self.norm:
            return -1.0 * np.sum(x, axis=1) / self.max_seq_length
        else:
            return -1.0 * np.sum(x, axis=1)
    
    # def numbers2letters(self, state):
    #     return "".join([self.inverse_lookup[el] for el in state])
        
    def __call__(self, sequences):
        """
        args:
            inputs: list of arrays in desired format interpretable by oracle
        returns:
            array of scores
        function:
            creates the complex set and calls the desired nupack function
        """
        # x = x.tolist()
        # sequences = [self.numbers2letters(seq) for seq in x]
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

        return torch.tensor(energy, device = self.device, dtype = self.float)

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
