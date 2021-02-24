'''import statements'''
import numpy as np
import scipy.io

'''
This script computes a binding score for a given sequence or set of sequences

> Inputs: DNA sequence in letter format
> Outputs: Sequence binding scores

To-Do:
==> automate MD sampling
==> Potts Hamiltonian is giving unexpectedly small energies
'''

class oracle():
    def __init__(self,params):
        '''
        initialize the oracle
        :param params:
        '''
        self.params = params


    def score(self, oracleSequences):
        '''
        assign correct scores to selected sequences
        :param oracleSequences: sequences to be scored
        :return: computed scores
        '''
        return self.PottsEnergy(oracleSequences)


    def PottsEnergy(self, oracleSequences):
        '''
        test oracle - learn one of our 40-mer Potts Hamiltonians
        :param oracleSequences: sequences to be scored
        :return:
        '''

        # DNA Potts model
        ''''''
        coupling_dict = scipy.io.loadmat('40_level_scored.mat')

        N = coupling_dict['h'].shape[1] # length of DNA chain

        assert N == len(oracleSequences[0]), "Hamiltonian and proposed sequences are different sizes!"

        h = coupling_dict['h']
        J = coupling_dict['J']

        energies = []
        ''''''
        for sequence in oracleSequences:
            energy = 0

            # potts hamiltonian
            '''
            l = 0
            for i in range(N - 1):

                for j in range(i + 1, N):
                    energy += J[sequence[i], sequence[j], l]  # site-specific couplings
                    l += 1

            for i in range(len(sequence)):
                energy += h[sequence[i], i]  # site-specific base energy
            '''
            # simpler hamiltonian
            energy = np.sum(sequence)

            energies.append(energy)

        return np.asarray(energies)