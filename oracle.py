'''import statements'''
import numpy as np
import scipy.io
import random
from seqfold import dg, fold
from nupack import *
from utils import *

'''
This script computes a binding score for a given sequence or set of sequences

> Inputs: DNA sequence in letter format
> Outputs: Sequence binding scores

To-Do:
==> linear expansion
==> inner product
==> Potts model
==> seqfold


params
'dataset seed' - self explanatory
'dict size' - number of possible states per sequence element - e.g., for ATGC 'dict size' = 4
'variable sample length', 'min sample length', 'max sample length' - for determining the length and variability of sample sequences
'init dataset length' - number of samples for initial (random) dataset
'dataset' - name of dataset to be saved
'''


class oracle():
    def __init__(self,params):
        '''
        initialize the oracle
        :param params:
        '''
        self.params = params
        np.random.seed(self.params['dataset seed'])
        self.seqLen = self.params['max sample length']

        if self.params['test mode']:
            self.linFactors = -np.ones(self.seqLen) # Uber-simple function, for testing purposes
        else:
            self.linFactors = np.random.randn(self.seqLen)  # coefficients for linear toy energy

        hamiltonian = np.random.randn(self.seqLen,self.seqLen) # energy function
        self.hamiltonian = np.tril(hamiltonian) + np.tril(hamiltonian, -1).T # random symmetric matrix

        pham = np.zeros((self.seqLen,self.seqLen,self.params['dict size'],self.params['dict size']))
        for i in range(pham.shape[0]):
            for j in range(i, pham.shape[1]):
                for k in range(pham.shape[2]):
                    for l in range(k, pham.shape[3]):
                        num = np.random.randn(1)
                        pham[i, j, k, l] = num
                        pham[i, j, l, k] = num
                        pham[j, i, k, l] = num
                        pham[j, i, l, k] = num
        self.pottsJ = pham # multilevel spin Hamiltonian (Potts Hamiltonian) - coupling term
        self.pottsH = np.random.randn(self.seqLen,self.params['dict size']) # Potts Hamiltonian - onsite term


    def initializeDataset(self,save = True, returnData = False, customSize=None):
        '''
        generate an initial toy dataset with a given number of samples
        need an extra factor to speed it up (duplicate filtering is very slow)
        :param numSamples:
        :return:
        '''
        data = {}
        np.random.seed(self.params['dataset seed'])
        if customSize is None:
            datasetLength = self.params['init dataset length']
        else:
            datasetLength = customSize

        if self.params['variable sample length']:
            samples = []
            while len(samples) < datasetLength:
                for i in range(self.params['min sample length'], self.params['max sample length'] + 1):
                    samples.extend(np.random.randint(0 + 1, self.params['dict size'] + 1, size=(int(100 * self.params['dict size'] * i), i)))

                samples = self.numpy_fillna(np.asarray(samples)).astype(int) # pad sequences up to maximum length
                samples = filterDuplicateSamples(samples) # this will naturally proportionally punish shorter sequences
                if len(samples) < datasetLength:
                    samples = samples.tolist()
            np.random.shuffle(samples) # shuffle so that sequences with different lengths are randomly distributed
            samples = samples[:datasetLength] # after shuffle, reduce dataset to desired size, with properly weighted samples
        else: # fixed sample size
            samples = np.random.randint(1, self.params['dict size'] + 1,size=(datasetLength, self.params['max sample length']))
            samples = filterDuplicateSamples(samples)
            while len(samples) < datasetLength:
                samples = np.concatenate((samples,np.random.randint(1, self.params['dict size'] + 1, size=(datasetLength, self.params['max sample length']))),0)
                samples = filterDuplicateSamples(samples)

        data['samples'] = samples
        data['scores'] = self.score(data['samples'])

        if save:
            np.save('datasets/' + self.params['dataset'], data)
        if returnData:
            return data

    def score(self, queries):
        '''
        assign correct scores to selected sequences
        :param queries: sequences to be scored
        :return: computed scores
        '''
        if isinstance(queries,list):
            queries = np.asarray(queries) # convert queries to array

        blockSize = int(1e4)
        if len(queries) > blockSize: # score in blocks of maximum 10000
            scores = []
            for i in range(int(len(queries) // blockSize)):
                queryBlock = queries[i * blockSize:(i+1)*blockSize]
                scores.extend(self.getScore(queryBlock))

            return np.asarray(scores)
        else:
            return self.getScore(queries)


    def getScore(self,queries):
        if self.params['dataset'] == 'linear':
            return self.linearToy(queries)
        elif self.params['dataset'] == 'potts':
            return self.PottsEnergy(queries)
        elif self.params['dataset'] == 'inner product':
            return self.toyHamiltonian(queries)
        elif self.params['dataset'] == 'seqfold':
            return self.seqfoldScore(queries)
        elif self.params['dataset'] == 'nupack':
            return self.nupackScore(queries)


    def linearToy(self,queries):
        '''
        return the energy of a toy model for the given set of queries
        sites are completely uncorrelated
        :param queries:
        :return:
        '''
        energies = queries @ self.linFactors # simple matmul - padding entries (zeros) have zero contribution

        return energies


    def toyHamiltonian(self,queries):
        '''
        return the energy of a toy model for the given set of queries
        sites may be correlated if they have a strong coupling (off diagonal term in the Hamiltonian)
        :param queries:
        :return:
        '''

        energies = np.zeros(len(queries))
        for i in range(len(queries)):
            energies[i] = queries[i] @ self.hamiltonian @ queries[i].transpose() # compute energy for each sample via inner product with the Hamiltonian

        return energies


    def PottsEnergy(self, queries):
        '''
        test oracle - randomly generated Potts Multilevel Spin Hamiltonian
        each pair of sites is correlated depending on the occupation of each site
        :param queries: sequences to be scored
        :return:
        '''

        # DNA Potts model - OLD
        #coupling_dict = scipy.io.loadmat('40_level_scored.mat')
        #N = coupling_dict['h'].shape[1] # length of DNA chain
        #assert N == len(queries[0]), "Hamiltonian and proposed sequences are different sizes!"
        #h = coupling_dict['h']
        #J = coupling_dict['J']

        energies = np.zeros(len(queries))
        for k in range(len(queries)):
            nnz = np.count_nonzero(queries[k])
            # potts hamiltonian
            for ii in range(nnz): # ignore padding terms
                energies[k] += self.pottsH[ii, queries[k,ii] - 1] # add onsite term and account for indexing (e.g. 1-4 -> 0-3)

                for jj in range(ii,nnz): # this is duplicated on lower triangle so we only need to do it from i-L
                    energies[k] += 2 * self.pottsJ[ii, jj, queries[k,ii] - 1, queries[k,jj] - 1]  # site-specific couplings

        return energies


    def seqfoldScore(self,queries, returnSS = False):
        '''
        get the secondary structure for a given sequence
        using seqfold here - identical features are available using nupack, though results are sometimes different
        :param sequence:
        :return:
        '''
        temperature = 37.0  # celcius
        sequences = self.numbers2letters(queries)

        energies = np.zeros(len(sequences))
        strings = []
        pairLists = []
        i = -1
        for sequence in sequences:
            i += 1
            en = dg(sequence, temp = temperature) # get predicted minimum energy of folded structure
            if np.isfinite(en):
                if en > 1500: # no idea why it does this but sometimes it adds 1600 - we will upgrade this to nupack in the future
                    energies[i] = en - 1600
                else:
                    energies[i] = en
            else:
                energies[i] = 5 # np.nan # set infinities as being very unlikely

            if returnSS:
                structs = fold(sequence)  # identify structural features
                # print(round(sum(s.e for s in structs), 2)) # predicted energy of the final structure

                desc = ["."] * len(sequence)
                pairList = []
                for s in structs:
                    pairList.append(s.ij[0])
                    if len(s.ij) == 1:
                        i, j = s.ij[0]
                        desc[i] = "("
                        desc[j] = ")"

                ssString = "".join(desc) # secondary structure string
                strings.append(ssString)
                pairList = np.asarray(pairList) + 1 # list of paired bases
                pairLists.append(pairList)

        if returnSS:
            return energies, strings, pairLists
        else:
            return energies

    def numbers2letters(self, sequences):  # Tranforming letters to numbers (1234 --> ATGC)
        '''
        Converts numerical values to ATGC-format
        :param sequences: numerical DNA sequences to be converted
        :return: DNA sequences in ATGC format
        '''
        if type(sequences) != np.ndarray:
            sequences = np.asarray(sequences)

        my_seq = ["" for x in range(len(sequences))]
        row = 0
        for j in range(len(sequences)):
            seq = sequences[j, :]
            assert type(seq) != str, 'Function inputs must be a list of equal length strings'
            for i in range(len(sequences[0])):
                na = seq[i]
                if na == 1:
                    my_seq[row] += 'A'
                elif na == 2:
                    my_seq[row] += 'T'
                elif na == 3:
                    my_seq[row] += 'C'
                elif na == 4:
                    my_seq[row] += 'G'
            row += 1
        return my_seq


    def numpy_fillna(self, data):
        '''
        function to pad uneven-length vectors up to the max with zeros
        :param data:
        :return:
        '''
        # Get lengths of each row of data
        lens = np.array([len(i) for i in data])

        # Mask of valid places in each row
        mask = np.arange(lens.max()) < lens[:, None]

        # Setup output array and put elements from data into masked positions
        out = np.zeros(mask.shape, dtype=data.dtype)
        out[mask] = np.concatenate(data)
        return out



    def nupackScore(self,queries,returnSS=False,parallel=True):

        #use nupack instead of seqfold - more stable and higher quality predictions in general
        #returns the energy of the most probable structure only
        #:param queries:
        #:param returnSS:
        #:return:

        temperature = 310.0  # Kelvin
        ionicStrength = 1.0 # molar
        sequences = self.numbers2letters(queries)

        energies = np.zeros(len(sequences))
        strings = []
        if parallel:
            # parallel evaluation - fast
            strandList = []
            comps = []
            i = -1
            for sequence in sequences:
                i += 1
                strandList.append(Strand(sequence, name='strand{}'.format(i)))
                comps.append(Complex([strandList[-1]], name='comp{}'.format(i)))

            set = ComplexSet(strands=strandList, complexes=SetSpec(max_size=1, include=comps))
            model1 = Model(material='dna', celsius=temperature - 273, sodium=ionicStrength)
            results = complex_analysis(set, model=model1, compute=['mfe'])
            for i in range(len(energies)):
                energies[i] = results[comps[i]].mfe[0].energy

                if returnSS:
                    strings.append(str(results[comps[i]].mfe[0].structure))

        else:
            i = -1
            for sequence in sequences:
                i += 1
                A = Strand(sequence, name='A')
                comp = Complex([A], name='AA')
                set1 = ComplexSet(strands=[A], complexes=SetSpec(max_size=1, include=[comp]))
                model1 = Model(material='dna', celsius=temperature - 273, sodium=ionicStrength)
                results = complex_analysis(set1, model=model1, compute=['mfe'])
                cout = results[comp]

                energies[i] = cout.mfe[0].energy
                if returnSS:
                    strings.append(cout.mfe[0].structure)

        if returnSS:
            return energies, strings
        else:
            return energies




''' # little script to test and have a look at the data

params = {}
params['dataset seed'] = 0
params['max sample length'] = 20
params['min sample length'] = 10
params['dataset'] = 'linear toy' # 'linear', 'potts', 'inner product', 'seqfold'
params['dict size'] = 4
params['init dataset length'] = 100000
params['variable sample length'] = True

oracle = oracle(params)
dataset = oracle.initializeDataset(save=False,returnData=True)

import time

t0 = time.time()
oracle.params['dataset'] = 'linear'
linScore = oracle.score(dataset['samples'])
tf = time.time()
print("Linear took {} seconds for {} samples".format(int(tf-t0), params['init dataset length']))

t0 = time.time()
oracle.params['dataset'] = 'inner product'
innerScore = oracle.score(dataset['samples'])
tf = time.time()
print("Inner product took {} seconds for {} samples".format(int(tf-t0), params['init dataset length']))

t0 = time.time()
oracle.params['dataset'] = 'potts'
pottsScore = oracle.score(dataset['samples'])
tf = time.time()
print("Potts took {} seconds for {} samples".format(int(tf-t0), params['init dataset length']))

t0 = time.time()
oracle.params['dataset'] = 'seqfold'
seqfoldScore = oracle.score(dataset['samples'])
tf = time.time()
print("Seqfold took {} seconds for {} samples".format(int(tf-t0), params['init dataset length']))

t0 = time.time()
oracle.params['dataset'] = 'nupack'
nupackScore = oracle.score(dataset['samples'])
tf = time.time()
print("Nupack took {} seconds for {} samples".format(int(tf-t0), params['init dataset length']))


import matplotlib.pyplot as plt

plt.clf()
plt.hist((linScore - np.mean(linScore))/np.sqrt(np.var(linScore)),alpha=0.5,density=True,bins=100,label='Linear')
plt.hist((innerScore - np.mean(innerScore))/np.sqrt(np.var(innerScore)),alpha=0.5,density=True,bins=100,label='Inner')
plt.hist((pottsScore - np.mean(pottsScore))/np.sqrt(np.var(pottsScore)),alpha=0.5,density=True,bins=100,label='Potts')
plt.hist((seqfoldScore - np.mean(seqfoldScore))/np.sqrt(np.var(seqfoldScore)),alpha=0.5,density=True,bins=100,label='Seqfold')
plt.hist((nupackScore - np.mean(nupackScore))/np.sqrt(np.var(nupackScore)),alpha=0.5,density=True,bins=100,label='nupack')
plt.legend()

'''