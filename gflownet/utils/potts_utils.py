import scipy.io as sco
import numpy as np


def load_potts_model(num_of_elements):
    J = np.zeros(shape=(4, 4, num_of_elements))
    for ii in range(num_of_elements):
        nextJJ = "./potts/JJ_" + str(ii + 1) + ".mat"
        Jdict = sco.loadmat(nextJJ)
        J[:, :, ii] = Jdict["JJ_out"]

    hdict = sco.loadmat("./potts/h_out.mat")
    h = hdict["h"]
    return J, h


def potts_energy(J, h, seq):
    N = np.size(seq)
    seq = seq - 1
    for ii in range(N):
        energy = 0
        l = 0
        for i in range(N - 1):
            for j in range(i + 1, N):
                energy = energy + J[seq[i], seq[j], l]
                l = l + 1

    for i in range(N):
        energy = energy + h[seq[i], i]

    energy = energy * -1
    return energy
