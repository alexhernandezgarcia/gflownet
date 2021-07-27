import multiprocessing as mp
import numpy as np
import os


class parent():
    def __init__(self):
        self.i = np.arange(10)
        self.j = np.arange(2,12)

    def parallelEval(self):
        cpus = int(np.amin((len(self.i), os.cpu_count() - 1)))  # np.min((os.cpu_count()-2,params['runs']))
        pool = mp.Pool(cpus)
        for i in range(int(np.ceil(len(self.i) / cpus))):
            trainingOutput = [pool.apply_async(coolFunction, args=[self.i[j], self.j[j]]) for j in range(cpus)]

        return trainingOutput


class parent2():
    def __init__(self):
        self.i = np.arange(10)
        self.j = np.arange(2,12)

    def parallelEval(self):
        cpus = int(np.amin((len(self.i), os.cpu_count() - 1)))  # np.min((os.cpu_count()-2,params['runs']))
        pool = mp.Pool(cpus)
        for i in range(int(np.ceil(len(self.i) / cpus))):
            trainingOutput = [pool.apply_async(coolFunction, args=[self.i[j], self.j[j]]) for j in range(cpus)]

        return trainingOutput



def coolFunction(i,j):

    function = functionClass()
    function.addition(i,j)
    output = function.sum

    return output

class functionClass():
    def __init__(self):
        pass

    def addition(self, i, j):
        self.sum = i+j


if __name__ == '__main__':
    parentClass = parent2()
    output = parentClass.parallelEval()
