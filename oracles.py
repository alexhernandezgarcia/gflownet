import numpy as np
import scipy.io
import random
from seqfold import dg, fold
try:
    from nupack import *
except:
    pass


def linearToy(queries):
    '''
    return the energy of a toy model for the given set of queries
    sites are completely uncorrelated
    :param queries:
    :return:
    '''
    linFactors = -np.ones(queries.shape[1]) # ultra-simple and always positive
    #linFactors = np.random.random(queries.shape[1]) # coefficients for linear toy energy
    energies = queries @ linFactors # simple matmul - padding entries (zeros) have zero contribution
    return energies


def toyHamiltonian(queries):
    '''
    return the energy of a toy model for the given set of queries
    sites may be correlated if they have a strong coupling (off diagonal term in the Hamiltonian)
    :param queries:
    :return:
    '''
    hamiltonian = np.random.random((queries.shape[1], queries.shape[1])) # energy function
    hamiltonian = np.tril(hamiltonian) + np.tril(hamiltonian, -1).T # random symmetric matrix

    energies = np.zeros(len(queries))
    for i in range(len(queries)):
        energies[i] = queries[i] @ hamiltonian @ queries[i].transpose() # compute energy for each sample via inner product with the Hamiltonian

    return energies


def PottsEnergy(queries, nalphabet=4):
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

    pham = np.zeros((queries.shape[1], queries.shape[1], nalphabet, nalphabet))
    for i in range(pham.shape[0]):
        for j in range(i, pham.shape[1]):
            for k in range(pham.shape[2]):
                for l in range(k, pham.shape[3]):
                    num = - np.random.uniform(0, 1)
                    pham[i, j, k, l] = num
                    pham[i, j, l, k] = num
                    pham[j, i, k, l] = num
                    pham[j, i, l, k] = num
    pottsJ = pham # multilevel spin Hamiltonian (Potts Hamiltonian) - coupling term
    pottsH = np.random.random((queries.shape[1], nalphabet)) # Potts Hamiltonian - onsite term
    energies = np.zeros(len(queries))
    for k in range(len(queries)):
        nnz = np.count_nonzero(queries[k])
        # potts hamiltonian
        for ii in range(nnz): # ignore padding terms
            energies[k] += pottsH[ii, queries[k,ii] - 1] # add onsite term and account for indexing (e.g. 1-4 -> 0-3)

            for jj in range(ii,nnz): # this is duplicated on lower triangle so we only need to do it from i-L
                energies[k] += 2 * pottsJ[ii, jj, queries[k,ii] - 1, queries[k,jj] - 1]  # site-specific couplings

    return energies


def seqfoldScore(queries, returnSS = False):
    '''
    get the secondary structure for a given sequence
    using seqfold here - identical features are available using nupack, though results are sometimes different
    :param sequence:
    :return:
    '''
    temperature = 37.0  # celcius
    sequences = numbers2letters(queries)

    energies = np.zeros(len(sequences))
    strings = []
    pairLists = []
    i = -1
    for sequence in sequences:
        i += 1
        if len(sequence) == 1:
            en = np.inf
        else:
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


def nupackScore(queries, returnFunc = 'energy'):
    # Nupack requires Linux OS.
    # use nupack instead of seqfold - more stable and higher quality predictions in general
    # returns the energy of the most probable structure only
    #:param queries:
    #:param returnFunct 'energy' 'hairpins' 'pairs'
    #:return:

    temperature = 310.0  # Kelvin
    ionicStrength = 1.0  # molar
    sequences = numbers2letters(queries)

    energies = np.zeros(len(sequences))
    nPins = np.zeros(len(sequences)).astype(int)
    nPairs = 0
    ssStrings = np.zeros(len(sequences), dtype=object)

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

    if returnFunc == 'hairpins':
        for i in range(len(ssStrings)):
            indA = 0  # hairpin completion index
            for j in range(len(sequences[i])):
                if ssStrings[i][j] == '(':
                    indA += 1
                elif ssStrings[i][j] == ')':
                    indA -= 1
                    if indA == 0:  # if we come to the end of a distinct hairpin
                        nPins[i] += 1
    if returnFunc == 'pairs':
        nPairs = np.asarray([ssString.count('(') for ssString in ssStrings]).astype(int)

    if returnFunc == 'energy':
        return energies
    elif returnFunc == 'hairpins':
        return nPins
    elif returnFunc == 'pairs':
        return nPairs


def numbers2letters(sequences):  # Tranforming letters to numbers (1234 --> ATGC)
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

