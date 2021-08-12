import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import data
from torch import optim
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import time
from oracle import oracle
from utils import *

params = {}
params['init dataset length'] = 100000 # times seq len is real dataset size
params['dict size'] = 4
params['variable sample length'] = True
params['min sample length'], params['max sample length'] = [10, 20]
params['dataset'] = 'seqfold' # linear, inner product, potts, seqfold, nupack

# model params
params['model'] = 'transformer' # 'mlp', 'lstm', 'transformer'
params['hidden dim'] = 128 # filters in fc layers
params['layers'] = 2
params['embed dim'] = 128 # embedding dimension for transformer and lstm
params['heads'] = 2 # transformer heads
params['epochs'] = 200
params['training batch'] = 100
params['GPU'] = True

params['dataset seed'] = 0


assert params['max sample length'] >= params['min sample length']

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

def numpy_fillna(data):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape, dtype=data.dtype)
    out[mask] = np.concatenate(data)
    return out

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

# define and initialize model
class lstm(nn.Module):
    def __init__(self,params):
        super(lstm,self).__init__()

        self.embedding = nn.Embedding(params['dict size'] + 1, embedding_dim = params['embed dim'])
        self.encoder = nn.LSTM(input_size=params['embed dim'],hidden_size=params['hidden dim'],num_layers=params['layers'])
        self.decoder = nn.Linear((params['hidden dim']), 1)

    def forward(self,x):
        x = x.permute(1,0) # weird input shape requirement
        embeds = self.embedding(x)
        y = self.encoder(embeds)[0]
        return self.decoder(y[-1,:,:])

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class transformer(nn.Module):
    def __init__(self,params):
        super(transformer,self).__init__()

        self.positionalEncoder = PositionalEncoding(params['embed dim'],max_len = params['max sample length'], dropout=0)
        self.embedding = nn.Embedding(params['dict size'] + 1, embedding_dim = params['embed dim'])
        encoder_layer = nn.TransformerEncoderLayer(params['embed dim'], nhead = params['heads'],dim_feedforward=params['hidden dim'], dropout=0, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = params['layers'])
        self.decoder1 = nn.Linear(params['embed dim'] * params['max sample length'], params['hidden dim'])
        self.decoder2 = nn.Linear(params['hidden dim'], 1)

    def forward(self,x):
        x_key_padding_mask = (x==0).clone().detach() # zero out the attention of empty sequence elements
        embed = self.embedding(x.transpose(1,0)) # [seq, batch]
        posEmbed = self.positionalEncoder(embed)
        encode = self.encoder(posEmbed,src_key_padding_mask=x_key_padding_mask)
        forFC = encode.permute(1,0,2).reshape(x.shape[0], int(params['embed dim']*params['max sample length']))
        y = F.gelu(self.decoder1(forFC))
        y = self.decoder2(y)
        return y

class mlp(nn.Module):
    def __init__(self, params):
        super(mlp, self).__init__()

        # build input and output layers
        self.initial_layer = nn.Linear(params['max sample length'], params['hidden dim']) # layer which takes in our sequence
        self.output_layer = nn.Linear(params['hidden dim'], 1)

        # build hidden layers
        self.lin_layers = []

        for i in range(params['layers']):
            self.lin_layers.append(nn.Linear(params['hidden dim'], params['hidden dim']))

        # initialize module lists
        self.lin_layers = nn.ModuleList(self.lin_layers)

    def forward(self, x):
        x = F.gelu(self.initial_layer(x.float())) # apply linear transformation and nonlinear activation
        for i in range(len(self.lin_layers)):
            x = F.gelu(self.lin_layers[i](x))

        x = self.output_layer(x) # linear transformation to output
        return x


# generate dataset
oracle = oracle(params)
dataset = oracle.initializeDataset(save=False,returnData=True)
samples = dataset['samples']
scores = dataset['scores']
datStd = np.sqrt(np.var(scores))
datMean = np.mean(scores)
scores = (scores - datMean) / datStd # standardize inputs

trainSize = int(len(samples) * 0.8)
testSize = len(samples) - trainSize
trainData = []
testData = []
for i in range(trainSize):
    trainData.append([samples[i],scores[i]])
for i in range(testSize):
    testData.append([samples[i],scores[i]])

tr = data.DataLoader(trainData, batch_size=params['training batch'], shuffle=True,num_workers=0,pin_memory=True)#,collate_fn = collator)
te = data.DataLoader(testData, batch_size=params['training batch'], shuffle=False,num_workers=0,pin_memory=True)#,collate_fn = collator)

torch.random.manual_seed(0)
if params['model'] == 'mlp':
    model = mlp(params)
elif params['model'] == 'transformer':
    model = transformer(params)
elif params['model'] == 'lstm':
    model = lstm(params)

if params['GPU']:
    model = model.cuda()

optimizer = optim.AdamW(model.parameters(), amsgrad=True)

# train and test
tr_hist = []
te_hist = []
for epoch in range(params['epochs']):
    t0 = time.time()
    err_tr = []
    model.train(True)
    for i, trainData in enumerate(tr):
        inputs = trainData[0]
        scores = trainData[1]
        if params['GPU']:
            inputs = inputs.cuda()
            scores = scores.cuda()
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
        if params['GPU']:
            inputs = inputs.cuda()
            scores = scores.cuda()
        out = model(inputs)
        loss = F.smooth_l1_loss(out[:,0],scores.float())
        err_te.append(loss.data)

    te_hist.append(torch.mean(torch.stack(err_tr)))
    tr_hist.append(torch.mean(torch.stack(err_te)))
    tf = time.time()
    print('Epoch {} train error {:.5f} test error {:.5f} in {} seconds'.format(epoch, tr_hist[-1], te_hist[-1], int(tf-t0)))



# inference
plt.semilogy(tr_hist,'.-')
plt.plot(te_hist,'.-')
nParams = get_n_params(model)
print('{} parameters in model'.format(nParams))
'''

checkpoint = torch.load('amodel')

if list(checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
    for i in list(checkpoint['model_state_dict']):
        checkpoint['model_state_dict'][i[7:]] = checkpoint['model_state_dict'].pop(i)

model.load_state_dict(checkpoint['model_state_dict'])
testSeq = torch.ones(20).unsqueeze(0).int()
out = model(testSeq)
'''