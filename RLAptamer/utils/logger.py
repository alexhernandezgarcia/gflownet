# Modified from (C) Wei YANG 2017
from __future__ import absolute_import
import numpy as np

__all__ = ['Logger', 'LoggerMonitor', 'savefig']

class Logger(object):
    '''Save training process to log file with simple plot function.'''

    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                if self.numbers['Valid mean iu']:
                    self.resume_epoch=np.argmax(self.numbers['Valid mean iu'])
                    self.resume_jacc=max(self.numbers['Valid mean iu'])
                    self.last_epoch = int(float(self.numbers['Epoch'][-1]))
                else:
                    self.last_epoch = 0
                    self.resume_jacc = 0
                    self.resume_epoch = 0
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'a')

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.16f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()


    def close(self):
        if self.file is not None:
            self.file.close()


class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''

    def __init__(self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)


if __name__ == '__main__':
    # # Example
    # logger = Logger('test.txt')
    # logger.set_names(['Train loss', 'Valid loss','Test loss'])

    # length = 100
    # t = np.arange(length)
    # train_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # valid_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # test_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1

    # for i in range(0, length):
    #     logger.append([train_loss[i], valid_loss[i], test_loss[i]])
    # logger.plot()

    # Example: logger monitor
    paths = {
        'resadvnet20': '/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet20/log.txt',
        'resadvnet32': '/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet32/log.txt',
        'resadvnet44': '/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet44/log.txt',
    }

    field = ['Valid Acc.']

    # monitor = LoggerMonitor(paths)
    # monitor.plot(names=field)
    # savefig('test.eps')