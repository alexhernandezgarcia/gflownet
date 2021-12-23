'''import statements'''
import numpy as np
import scipy.io
import random
from seqfold import dg, fold
from utils import *
import sys
try: # we don't always install these on every platform
    from nupack import *
except:
    pass
try:
    from bbdob.utils import idx2one_hot
    from bbdob import OneMax, TwoMin, FourPeaks, DeceptiveTrap, NKLandscape, WModel
except:
    pass


'''
This script computes a binding score for a given sequence or set of sequences

> Inputs: numpy integer arrays - different oracles with different requirements
> Outputs: oracle outputs - usually numbers

config
'dataset seed' - self explanatory
'dict size' - number of possible states per sequence element - e.g., for ATGC 'dict size' = 4
'variable sample length', 'min sample length', 'max sample length' - for determining the length and variability of sample sequences
'init dataset length' - number of samples for initial (random) dataset
'dataset' - name of dataset to be saved
'''


class Oracle():
    def __init__(self, config):
        '''
        initialize the oracle
        :param config:
        '''
        self.config = config
        self.seqLen = self.config.dataset.max_length

        self.initRands()


    def initRands(self):
        '''
        initialize random numbers for custom-made toy functions
        :return:
        '''
        np.random.seed(self.config.seeds.toy_oracle)

        # set these to be always positive to play nice with gFlowNet sampling
        if True:#self.config.test_mode:
            self.linFactors = -np.ones(self.seqLen) # Uber-simple function, for testing purposes - actually nearly functionally identical to one-max, I believe
        else:
            self.linFactors = np.abs(np.random.randn(self.seqLen))  # coefficients for linear toy energy

        hamiltonian = np.random.randn(self.seqLen,self.seqLen) # energy function
        self.hamiltonian = np.tril(hamiltonian) + np.tril(hamiltonian, -1).T # random symmetric matrix

        pham = np.zeros((self.seqLen,self.seqLen,self.config.dataset.dict_size,self.config.dataset.dict_size))
        for i in range(pham.shape[0]):
            for j in range(i, pham.shape[1]):
                for k in range(pham.shape[2]):
                    for l in range(k, pham.shape[3]):
                        num =  - np.random.uniform(0,1)
                        pham[i, j, k, l] = num
                        pham[i, j, l, k] = num
                        pham[j, i, k, l] = num
                        pham[j, i, l, k] = num
        self.pottsJ = pham # multilevel spin Hamiltonian (Potts Hamiltonian) - coupling term
        self.pottsH = np.random.randn(self.seqLen,self.config.dataset.dict_size) # Potts Hamiltonian - onsite term

        # W-model parameters
        # first get the binary dimension size
        aa = np.arange(self.config.dataset.dict_size)
        if self.config.dataset.variable_length:
            aa = np.clip(aa, 1, self.config.dataset.dict_size) #  merge padding with class 1
        x0 = np.binary_repr(aa[-1])
        dimension = int(len(x0) * self.config.dataset.max_length)

        mu = np.random.randint(1, dimension + 1)
        v = np.random.randint(1, dimension + 1)
        m = np.random.randint(1, dimension)
        n = np.random.randint(1, dimension)
        gamma = np.random.randint(0, int(n * (n - 1 ) / 2))
        self.mu, self.v, self.m, self.n, self.gamma = [mu, v, m, n, gamma]


    def initializeDataset(self,save = True, returnData = False, customSize=None):
        '''
        generate an initial toy dataset with a given number of samples
        need an extra factor to speed it up (duplicate filtering is very slow)
        :param numSamples:
        :return:
        '''
        data = {}
        np.random.seed(self.config.seeds.dataset)
        if customSize is None:
            datasetLength = self.config.dataset.init_length
        else:
            datasetLength = customSize

        if self.config.dataset.variable_length:
            samples = []
            while len(samples) < datasetLength:
                for i in range(self.config.dataset.min_length, self.config.dataset.max_length + 1):
                    samples.extend(np.random.randint(0 + 1, self.config.dataset.dict_size + 1, size=(int(10 * self.config.dataset.dict_size * i), i)))

                samples = self.numpy_fillna(np.asarray(samples, dtype = object)) # pad sequences up to maximum length
                samples = filterDuplicateSamples(samples) # this will naturally proportionally punish shorter sequences
                if len(samples) < datasetLength:
                    samples = samples.tolist()
            np.random.shuffle(samples) # shuffle so that sequences with different lengths are randomly distributed
            samples = samples[:datasetLength] # after shuffle, reduce dataset to desired size, with properly weighted samples
        else: # fixed sample size
            samples = np.random.randint(1, self.config.dataset.dict_size + 1,size=(datasetLength, self.config.dataset.max_length))
            samples = filterDuplicateSamples(samples)
            while len(samples) < datasetLength:
                samples = np.concatenate((samples,np.random.randint(1, self.config.dataset.dict_size + 1, size=(datasetLength, self.config.dataset.max_length))),0)
                samples = filterDuplicateSamples(samples)

        data['samples'] = samples
        data['scores'] = self.score(data['samples'])

        if save:
            np.save('datasets/' + self.config.dataset.oracle, data)
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
            scores_list = []
            scores_dict = {}
            for i in range(int(len(queries) // blockSize)):
                queryBlock = queries[i * blockSize:(i+1)*blockSize]
                scores_block = self.getScore(queryBlock)
                if isinstance(scores_block, dict):
                    for k, v in scores_block.items():
                        if k in scores_dict:
                            scores_dict[k].extend(list(v))
                        else:
                            scores_dict.update({k: list(v)})
                else:
                    scores_list.extend(self.getScore(queryBlock))
            if len(scores_list) > 0:
                return np.asarray(scores_list)
            else:
                return {k: np.asarray(v) for k, v in scores_dict.items()}
        else:
            return self.getScore(queries)


    def getScore(self,queries):
        if self.config.dataset.oracle == 'linear':
            return self.linearToy(queries)
        elif self.config.dataset.oracle == 'potts':
            return self.PottsEnergy(queries)
        elif self.config.dataset.oracle == 'inner product':
            return self.toyHamiltonian(queries)
        elif self.config.dataset.oracle == 'seqfold':
            return self.seqfoldScore(queries)
        elif self.config.dataset.oracle == 'nupack energy':
            return self.nupackScore(queries, returnFunc = 'energy')
        elif self.config.dataset.oracle == 'nupack pins':
            return -self.nupackScore(queries, returnFunc = 'pins')
        elif self.config.dataset.oracle == 'nupack pairs':
            return -self.nupackScore(queries, returnFunc = 'pairs')
        elif (self.config.dataset.oracle == 'onemax') or (self.config.dataset.oracle == 'twomin') or (self.config.dataset.oracle == 'fourpeaks')\
                or (self.config.dataset.oracle == 'deceptivetrap') or (self.config.dataset.oracle == 'nklandscape') or (self.config.dataset.oracle == 'wmodel'):
            return self.BB_DOB_functions(queries)
        elif isinstance(self.config.dataset.oracle, list) and all(["nupack " in el for el in self.config.dataset.oracle]):
            return self.nupackScore(queries, returnFunc=[el.replace("nupack ", "") for el in self.config.dataset.oracle])
        else:
            raise NotImplementedError("Unknown orackle type")


    def BB_DOB_functions(self, queries):
        '''
        BB-DOB OneMax benchmark
        :param queries:
        :return:
        '''
        if self.config.dataset.variable_length:
            queries = np.clip(queries, 1, self.config.dataset.dict_size) #  merge padding with class 1

        x0 = [np.binary_repr((queries[i][j] - 1).astype('uint8'),width=2) for i in range(len(queries)) for j in range(self.config.dataset.max_length)] # convert to binary
        x0 = np.asarray(x0).astype(str).reshape(len(queries), self.config.dataset.max_length) # reshape to proper size
        x0= [''.join(x0[i]) for i in range(len(x0))] # concatenate to binary strings
        x1 = np.zeros((len(queries),len(x0[0])),int) # initialize array
        for i in range(len(x0)): # finally, as an array (took me long enough)
            x1[i] = np.asarray(list(x0[i])).astype(int)

        dimension = x1.shape[1]

        x1 = idx2one_hot(x1, 2) # convert to BB_DOB one_hot format

        objective = self.getObjective(dimension)

        evals, info = objective(x1)

        return evals


    def getObjective(self, dimension):
        if self.config.dataset.oracle == 'onemax': # very limited in our DNA one-hot encoding
            objective = OneMax(dimension)
        elif self.config.dataset.oracle == 'twomin':
            objective = TwoMin(dimension)
        elif self.config.dataset.oracle == 'fourpeaks': # very limited in our DNA one-hot encoding
            objective = FourPeaks(dimension, t=3)
        elif self.config.dataset.oracle == 'deceptivetrap':
            objective = DeceptiveTrap(dimension, minimize=True)
        elif self.config.dataset.oracle == 'nklandscape':
            objective = NKLandscape(dimension, minimize=True)
        elif self.config.dataset.oracle == 'wmodel':
            objective = WModel(dimension, mu=self.mu, v=self.v, m = self.m, n = self.n, gamma = self.gamma, minimize=True)
        else:
            printRecord(self.config.dataset.oracle + ' is not a valid dataset!')
            sys.exit()

        return objective

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
        out = np.zeros(mask.shape, dtype=object)
        out[mask] = np.concatenate(data)
        return out


    def nupackScore(self, queries, returnFunc='energy'):
        # Nupack requires Linux OS.
        #use nupack instead of seqfold - more stable and higher quality predictions in general
        #returns the energy of the most probable structure only
        #:param queries:
        #:param returnFunct 'energy' 'pins' 'pairs'
        #:return:

        temperature = 310.0  # Kelvin
        ionicStrength = 1.0 # molar
        sequences = self.numbers2letters(queries)

        energies = np.zeros(len(sequences))
        strings = []
        nPins = np.zeros(len(sequences)).astype(int)
        nPairs = 0
        ssStrings = np.zeros(len(sequences),dtype=object)

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
            ssStrings[i] = str(results[comps[i]].mfe[0].structure)

        dict_return = {}
        if 'pins' in returnFunc:
            for i in range(len(ssStrings)):
                indA = 0  # hairpin completion index
                for j in range(len(sequences[i])):
                    if ssStrings[i][j] == '(':
                        indA += 1
                    elif ssStrings[i][j] == ')':
                        indA -= 1
                        if indA == 0:  # if we come to the end of a distinct hairpin
                            nPins[i] += 1
            dict_return.update({"pins": nPins})
        if 'pairs' in returnFunc:
            nPairs = np.asarray([ssString.count('(') for ssString in ssStrings]).astype(int)
            dict_return.update({"pairs": nPairs})
        if 'energy' in returnFunc:
            dict_return.update({"energy": energies})

        if isinstance(returnFunc, list):
            if len(returnFunc) > 1:
                return dict_return
            else:
                return dict_return[returnFunc[0]]
        else:
            return dict_return[returnFunc]

