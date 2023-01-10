from src.gflownet.proxy.base import Proxy
import numpy as np
from nupack import *


class Nupack(Proxy):
    def __init__(self, function):
        super().__init__()
        if function == "energy":
            self.function = self._func_energy
        elif function == "pairs":
            self.function = self._func_pairs
        else:
            raise NotImplementedError
        # TODO: add pins, motif and energyweighting nupack functions?
        # TODO: dictionary implementation in previous code -- do we need to add that here?

    def __call__(self, sequences):
        """
        args:
            inputs: list of strings like "ACCTG"
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

        return energy

    def _func_energy(self, sequences, results, comps):
        energies = np.zeros(len(sequences))
        for i in range(len(energies)):
            energies[i] = -results[comps[i]].mfe[0].energy
        return energies

    def _func_pairs(self, sequences, results, comps):
        ssStrings = np.zeros(len(sequences), dtype=object)
        for i in range(len(ssStrings)):
            ssStrings[i] = str(results[comps[i]].mfe[0].structure)
        nPairs = np.asarray([ssString.count("(") for ssString in ssStrings]).astype(int)
        return -nPairs
