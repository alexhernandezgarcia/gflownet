import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import data
from torch import optim
import numpy as np
import os
import random
import matplotlib.pyplot as plt

params = {}
params['dataset size'] = 1000 # times seq len is real dataset size
params['random seed'] = 0
params['training batch'] = 1
params['dict size'] = 4
params['seq len'] = 10 # max from 1 to 10
params['hidden dim'] = 24
params['epochs'] = 10
params['embed dim'] = 4

# useful
def convertString(strings):
    outputs = []
    for string in strings:
        featureMap = np.zeros((params['dict size'],len(string)))
        for j in range(len(string)):
            for k in range(params['dict size']):
                if int(string[j]) == k:
                    featureMap[k,j] = 1
        outputs.append(featureMap)

    return outputs

def split(word):
    return [char for char in word]

def collator(data):
    '''
    We should build a custom collate_fn rather than using default collate_fn,
    as the size of every sentence is different and merging sequences (including padding)
    is not supported in default.
    Args:
        data: list of tuple (training sequence, label)
    Return:
        padded_seq - Padded Sequence, tensor of shape (batch_size, padded_length)
        length - Original length of each sequence(without padding), tensor of shape(batch_size)
        label - tensor of shape (batch_size)
    '''

    #sorting is important for usage pack padded sequence (used in model). It should be in decreasing order.
    data.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, label = zip(*data)
    length = [len(seq) for seq in sequences]
    padded_seq = torch.zeros(len(sequences), max(length)).long()
    for i, seq in enumerate(sequences):
        end = length[i]
        padded_seq[i,:end] = seq

    return padded_seq, torch.from_numpy(np.array(length)), torch.from_numpy(np.array(label))


# generate dataset
torch.random.manual_seed(params['random seed'])
samples = []
for i in range(1,params['seq len']):
    samples.extend(torch.randint(0,params['dict size'],size=(params['dataset size'], i))) # initialize sequences from length 15-40

random.shuffle(samples)
scores = np.zeros(len(samples))
for i in range(len(samples)):
    scores[i] = torch.sum(samples[i] == 0) / len(samples[i]) # proportion of each sample which is zeros
    scores[i] += torch.log(torch.sum(samples[i] == 1)+1)
    scores[i] += torch.sin(torch.sum(samples[i] == 2)/5 * torch.sum(samples[i] == 3)/10)
# create dataloader
trainSize = int(len(samples) * 0.8)
testSize = len(samples) - trainSize
trainData = []
testData = []
for i in range(trainSize):
    trainData.append([samples[i],scores[i]])
for i in range(testSize):
    testData.append([samples[i],scores[i]])

tr = data.DataLoader(trainData, batch_size=params['training batch'], shuffle=True,num_workers=0,pin_memory=True)#,collate_fn = collator)
te = data.DataLoader(testData, batch_size=params['training batch'], shuffle=True,num_workers=0,pin_memory=True)#,collate_fn = collator)

# define and initialize model
class lstm_1(nn.Module):
    def __init__(self,params):
        super(lstm_1,self).__init__()

        self.embedding = nn.Embedding(params['dict size'], embedding_dim = params['embed dim'])
        self.encoder = nn.LSTM(input_size=params['embed dim'],hidden_size=params['hidden dim'],num_layers=params['lstm layers'])
        self.decoder = nn.Linear((params['hidden dim']), 1)

    def forward(self,x):
        x = x.permute(1,0) # weird input shape requirement
        embeds = self.embedding(x)
        y = self.encoder(embeds)[0]
        return self.decoder(y[-1,:,:])


model = lstm_1(params)
optimizer = optim.AdamW(model.parameters(), amsgrad=True)

# train and test
tr_hist = []
te_hist = []
for epoch in range(params['epochs']):
    err_tr = []
    model.train(True)
    for i, trainData in enumerate(tr):
        inputs = trainData[0]
        scores = trainData[1]
        out = model(inputs)
        loss = F.smooth_l1_loss(out[:,0],scores.float())
        err_tr.append(loss.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    err_te = []
    model.train(False)
    for i, trainData in enumerate(te):
        inputs = trainData[0]
        scores = trainData[1]
        out = model(inputs)
        loss = F.smooth_l1_loss(out[:,0],scores.float())
        err_te.append(loss.data)

    te_hist.append(torch.mean(torch.stack(err_tr)))
    tr_hist.append(torch.mean(torch.stack(err_te)))
    print('Epoch {} train error {:.5f} test error {:.5f}'.format(epoch, tr_hist[-1], te_hist[-1]))

# inference