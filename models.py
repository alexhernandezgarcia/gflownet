'''Import statements'''
import torch
import torch.nn.functional as F
from torch.utils import data
from torch import nn, optim, cuda, backends
import math
from sklearn.utils import shuffle

import os
import sys
from utils import *

'''
This script contains models for fitting DNA sequence data

> Inputs: list of DNA sequences in letter format
> Outputs: predicted binding scores, prediction uncertainty 

To-do's
==> upgrade to twin net
==> add noisey augmentation and/or few-shot dimension reduction
==> add positional embedding

Problems
==> we need to think about whether or not to shuffle test set between runs, or indeed what to use in the test set at all - right now we shuffle
'''


class modelNet():
    def __init__(self, params, ensembleIndex):
        self.params = params
        self.ensembleIndex = ensembleIndex
        self.params.history = min(20, self.params.proxy_max_epochs) # length of past to check
        self.initModel()
        torch.random.manual_seed(int(params.model_seed + ensembleIndex))


    def initModel(self):
        '''
        Initialize model and optimizer
        :return:
        '''
        if self.params.proxy_model_type == 'transformer': # switch to variable-length sequence model
            self.model = transformer(self.params)
        elif self.params.proxy_model_type == 'mlp':
            self.model = MLP(self.params)
        else:
            print(self.params.proxy_model_type + ' is not one of the available models')

        if self.params.GPU:
            self.model = self.model.cuda()
        self.optimizer = optim.AdamW(self.model.parameters(), amsgrad=True)
        datasetBuilder = buildDataset(self.params)
        self.mean, self.std = datasetBuilder.getStandardization()


    def save(self, best):
        if best == 0:
            torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, 'ckpts/'+getModelName(self.ensembleIndex)+'_final')
        elif best == 1:
            torch.save({'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict()}, 'ckpts/'+getModelName(self.ensembleIndex))


    def load(self,ensembleIndex):
        '''
        Check if a checkpoint exists for this model - if so, load it
        :return:
        '''
        dirName = getModelName(ensembleIndex)
        if os.path.exists('ckpts/' + dirName):  # reload model
            checkpoint = torch.load('ckpts/' + dirName)

            if list(checkpoint['model_state_dict'])[0][0:6] == 'module':  # when we use dataparallel it breaks the state_dict - fix it by removing word 'module' from in front of everything
                for i in list(checkpoint['model_state_dict']):
                    checkpoint['model_state_dict'][i[7:]] = checkpoint['model_state_dict'].pop(i)

            self.model.load_state_dict(checkpoint['model_state_dict'])

            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #prev_epoch = checkpoint['epoch']

            if self.params.GPU:
                self.model.cuda()  # move net to GPU
                for state in self.optimizer.state.values():  # move optimizer to GPU
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

            self.model.eval()
            #printRecord('Reloaded model: ', dirName)
        else:
            pass
            #printRecord('New model: ', dirName)


    def converge(self, returnHist = False):
        '''
        train model until test loss converges
        :return:
        '''
        [self.err_tr_hist, self.err_te_hist] = [[], []] # initialize error records

        tr, te, self.datasetSize = getDataloaders(self.params, self.ensembleIndex)

        #printRecord(f"Dataset size is: {bcolors.OKCYAN}%d{bcolors.ENDC}" %self.datasetSize)

        self.converged = 0 # convergence flag
        self.epochs = 0

        while (self.converged != 1):
            if (self.epochs % 10 == 0) and self.params.debug:
                printRecord("Model {} epoch {}".format(self.ensembleIndex, self.epochs))

            if self.epochs > 0: #  this allows us to keep the previous model if it is better than any produced on this run
                self.train_net(tr)
            else:
                self.err_tr_hist.append(0)

            self.test(te) # baseline from any prior training
            if self.err_te_hist[-1] == np.min(self.err_te_hist): # if this is the best test loss we've seen
                self.save(best=1)
            # after training at least 10 epochs, check convergence
            if self.epochs >= self.params.history:
                self.checkConvergence()

            self.epochs += 1

        if returnHist:
            return self.err_te_hist


    def train_net(self, tr):
        '''
        perform one epoch of training
        :param tr: training set dataloader
        :return: n/a
        '''
        err_tr = []
        self.model.train(True)
        for i, trainData in enumerate(tr):
            loss = self.getLoss(trainData)
            err_tr.append(loss.data)  # record the loss

            self.optimizer.zero_grad()  # run the optimizer
            loss.backward()
            self.optimizer.step()

        self.err_tr_hist.append(torch.mean(torch.stack(err_tr)).cpu().detach().numpy())


    def test(self, te):
        '''
        get the loss over the test dataset
        :param te: test set dataloader
        :return: n/a
        '''
        err_te = []
        self.model.train(False)
        with torch.no_grad():  # we won't need gradients! no training just testing
            for i, testData in enumerate(te):
                loss = self.getLoss(testData)
                err_te.append(loss.data)  # record the loss

        self.err_te_hist.append(torch.mean(torch.stack(err_te)).cpu().detach().numpy())


    def getLoss(self, train_data):
        """
        get the regression loss on a batch of datapoints
        :param train_data: sequences and scores
        :return: model loss over the batch
        """
        inputs = train_data[0]
        targets = train_data[1]
        if self.params.GPU:
            inputs = inputs.cuda()
            targets = targets.cuda()

        output = self.model(inputs.float())
        targets = (targets - self.mean)/self.std # standardize the targets during training
        return F.smooth_l1_loss(output[:,0], targets.float())


    def checkConvergence(self):
        """
        check if we are converged
        condition: test loss has increased or levelled out over the last several epochs
        :return: convergence flag
        """
        # check if test loss is increasing for at least several consecutive epochs
        eps = 1e-4 # relative measure for constancy

        if all(np.asarray(self.err_te_hist[-self.params.history+1:])  > self.err_te_hist[-self.params.history]): #
            self.converged = 1
            printRecord(bcolors.WARNING + "Model converged after {} epochs - test loss increasing, at {:.4f}".format(self.epochs + 1, min(self.err_te_hist)) + bcolors.ENDC)

        # check if test loss is unchanging
        if abs(self.err_te_hist[-self.params.history] - np.average(self.err_te_hist[-self.params.history:]))/self.err_te_hist[-self.params.history] < eps:
            self.converged = 1
            printRecord(bcolors.WARNING + "Model converged after {} epochs - hit test loss convergence criterion at {:.4f}".format(self.epochs + 1, min(self.err_te_hist)) + bcolors.ENDC)

        if self.epochs >= self.params.proxy_max_epochs:
            self.converged = 1
            printRecord(bcolors.WARNING + "Model converged after {} epochs- epoch limit was hit with test loss {:.4f}".format(self.epochs + 1, min(self.err_te_hist)) + bcolors.ENDC)


        #if self.converged == 1:
        #    printRecord(f'{bcolors.OKCYAN}Model training converged{bcolors.ENDC} after {bcolors.OKBLUE}%d{bcolors.ENDC}' %self.epochs + f" epochs and with a final test loss of {bcolors.OKGREEN}%.3f{bcolors.ENDC}" % np.amin(np.asarray(self.err_te_hist)))


    def evaluate(self, Data, output="Average"):
        '''
        evaluate the model
        output types - if "Average" return the average of ensemble predictions
            - if 'Variance' return the variance of ensemble predictions
        # future upgrade - isolate epistemic uncertainty from intrinsic randomness
        :param Data: input data
        :return: model scores
        '''
        if self.params.GPU:
            Data = torch.Tensor(Data).cuda().float()
        else:
            Data = torch.Tensor(Data).float()

        self.model.train(False)
        with torch.no_grad():  # we won't need gradients! no training just testing
            out = self.model(Data).cpu().detach().numpy()
            if output == 'Average':
                return np.average(out,axis=1) * self.std + self.mean
            elif output == 'Variance':
                return np.var(out * self.std + self.mean,axis=1)
            elif output == 'Both':
                return np.average(out,axis=1) * self.std + self.mean, np.var(out * self.std,axis=1)


    def loadEnsemble(self,models):
        '''
        load up a model ensemble
        :return:
        '''
        self.model = modelEnsemble(models)
        if self.params.GPU:
            self.model = self.model.cuda()


class modelEnsemble(nn.Module): # just for evaluation of a pre-trained ensemble
    def __init__(self,models):
        super(modelEnsemble, self).__init__()
        self.models = models
        self.models = nn.ModuleList(self.models)

    def forward(self, x):
        output = []
        for i in range(len(self.models)): # get the prediction from each model
            output.append(self.models[i](x.clone()))

        output = torch.cat(output,dim=1) #
        return output # return mean and variance of the ensemble predictions


class buildDataset():
    '''
    build dataset object
    '''
    def __init__(self, params):
        dataset = np.load('datasets/' + params.dataset+'.npy', allow_pickle=True)
        dataset = dataset.item()
        self.samples = dataset['samples']
        self.targets = dataset['scores']

        self.samples, self.targets = shuffle(self.samples, self.targets, random_state=params.init_dataset_seed)

    def reshuffle(self, seed=None):
        self.samples, self.targets = shuffle(self.samples, self.targets, random_state=seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

    def getFullDataset(self):
        return self.samples, self.targets

    def getStandardization(self):
        return np.mean(self.targets), np.sqrt(np.var(self.targets))


def getDataloaders(params, ensembleIndex): # get the dataloaders, to load the dataset in batches
    '''
    creat dataloader objects from the dataset
    :param params:
    :return:
    '''
    training_batch = params.proxy_training_batch_size
    dataset = buildDataset(params)  # get data
    if params.proxy_shuffle_dataset:
        dataset.reshuffle(seed=ensembleIndex)
    train_size = int(0.8 * len(dataset))  # split data into training and test sets

    test_size = len(dataset) - train_size

    # construct dataloaders for inputs and targets
    train_dataset = []
    test_dataset = []

    for i in range(test_size, test_size + train_size): # take the training data from the end - we will get the newly appended datapoints this way without ever seeing the test set
        train_dataset.append(dataset[i])
    for i in range(test_size): # test data is drawn from oldest datapoints
        test_dataset.append(dataset[i])

    tr = data.DataLoader(train_dataset, batch_size=training_batch, shuffle=True, num_workers= 0, pin_memory=False)  # build dataloaders
    te = data.DataLoader(test_dataset, batch_size=training_batch, shuffle=False, num_workers= 0, pin_memory=False) # num_workers must be zero or multiprocessing will not work (can't spawn multiprocessing within multiprocessing)

    return tr, te, dataset.__len__()


def getDataSize(params):
    dataset = np.load('datasets/' + params.dataset + '.npy', allow_pickle=True)
    dataset = dataset.item()
    samples = dataset['samples']

    return len(samples[0])


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

        self.embedDim = params.proxy_model_width
        self.hiddenDim = params.proxy_model_width
        self.layers = params.proxy_model_layers
        self.maxLen = params.max_sample_length
        self.dictLen = params.dict_size
        self.tasks = params.sample_tasks
        self.heads = min([4, max([1,self.embedDim//self.dictLen])])

        self.positionalEncoder = PositionalEncoding(self.embedDim, max_len = self.maxLen)
        self.embedding = nn.Embedding(self.dictLen + 1, embedding_dim = self.embedDim)
        encoder_layer = nn.TransformerEncoderLayer(self.embedDim, nhead = self.heads,dim_feedforward=self.hiddenDim, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = self.layers)
        self.decoder1 = nn.Linear(int(self.embedDim * self.maxLen), self.hiddenDim)

        self.output_layers = []
        for i in range(self.tasks):
            self.output_layers.append(nn.Linear(self.filters, 1))
        self.output_layers = nn.ModuleList(self.output_layers)


    def forward(self,x):
        x_key_padding_mask = (x==0).clone().detach() # zero out the attention of empty sequence elements
        x = self.embedding(x.transpose(1,0).int()) # [seq, batch]
        x = self.positionalEncoder(x)
        x = self.encoder(x,src_key_padding_mask=x_key_padding_mask)
        x = x.permute(1,0,2).reshape(x_key_padding_mask.shape[0], int(self.embedDim*self.maxLen))
        x = F.gelu(self.decoder1(x))

        y = torch.zeros(self.tasks)
        for i in range(self.tasks):
            y = self.output_layers[i](x) # each task has its own head        return x

        return y

class LSTM(nn.Module):
    '''
    may not work currently - possible issues with unequal length batching
    '''
    def __init__(self,params):
        super(LSTM,self).__init__()
        # initialize constants and layers

        self.embedding = nn.Embedding(2, embedding_dim = params.embedding_dim)
        self.encoder = nn.LSTM(input_size=params.embedding_dim,hidden_size=params.proxy_model_width,num_layers=params.proxy_model_layers)
        self.decoder = nn.Linear((params.proxy_model_width), 1)

    def forward(self, x):
        x = x.permute(1,0) # weird input shape requirement
        embeds = self.embedding(x.int())
        y = self.encoder(embeds)[0]
        return self.decoder(y[-1,:,:])


class MLP(nn.Module):
    def __init__(self,params):
        super(MLP,self).__init__()
        # initialize constants and layers

        if True:
            act_func = 'gelu'

        self.inputLength = params.max_sample_length
        self.tasks = params.sample_tasks
        self.layers = params.proxy_model_layers
        self.filters = params.proxy_model_width
        self.classes = int(params.dict_size + 1)
        self.init_layer_depth = int(self.inputLength * self.classes)

        # build input and output layers
        self.initial_layer = nn.Linear(int(self.inputLength * self.classes), self.filters) # layer which takes in our sequence in one-hot encoding
        self.activation1 = Activation(act_func,self.filters,params)

        self.output_layers = []
        for i in range(self.tasks):
            self.output_layers.append(nn.Linear(self.filters, 1))
        self.output_layers = nn.ModuleList(self.output_layers)

        # build hidden layers
        self.lin_layers = []
        self.activations = []
        self.norms = []

        for i in range(self.layers):
            self.lin_layers.append(nn.Linear(self.filters,self.filters))
            self.activations.append(Activation(act_func, self.filters))
            #self.norms.append(nn.BatchNorm1d(self.filters))

        # initialize module lists
        self.lin_layers = nn.ModuleList(self.lin_layers)
        self.activations = nn.ModuleList(self.activations)
        #self.norms = nn.ModuleList(self.norms)


    def forward(self, x):
        x = F.one_hot(x.long(),num_classes=self.classes)
        x = x.reshape(x.shape[0], self.init_layer_depth).float()
        x = self.activation1(self.initial_layer(x)) # apply linear transformation and nonlinear activation
        for i in range(self.layers):
            x = self.lin_layers[i](x)
            x = self.activations[i](x)
            #x = self.norms[i](x)

        y = torch.zeros(self.tasks)
        for i in range(self.tasks):
            y = self.output_layers[i](x) # each task has its own head

        return y


class kernelActivation(nn.Module): # a better (pytorch-friendly) implementation of activation as a linear combination of basis functions
    def __init__(self, n_basis, span, channels, *args, **kwargs):
        super(kernelActivation, self).__init__(*args, **kwargs)

        self.channels, self.n_basis = channels, n_basis
        # define the space of basis functions
        self.register_buffer('dict', torch.linspace(-span, span, n_basis)) # positive and negative values for Dirichlet Kernel
        gamma = 1/(6*(self.dict[-1]-self.dict[-2])**2) # optimum gaussian spacing parameter should be equal to 1/(6*spacing^2) according to KAFnet paper
        self.register_buffer('gamma',torch.ones(1) * gamma) #

        #self.register_buffer('dict', torch.linspace(0, n_basis-1, n_basis)) # positive values for ReLU kernel

        # define module to learn parameters
        # 1d convolutions allow for grouping of terms, unlike nn.linear which is always fully-connected.
        # #This way should be fast and efficient, and play nice with pytorch optim
        self.linear = nn.Conv1d(channels * n_basis, channels, kernel_size=(1,1), groups=int(channels), bias=False)

        #nn.init.normal(self.linear.weight.data, std=0.1)


    def kernel(self, x):
        # x has dimention batch, features, y, x
        # must return object of dimension batch, features, y, x, basis
        x = x.unsqueeze(2)
        if len(x)==2:
            x = x.reshape(2,self.channels,1)

        return torch.exp(-self.gamma*(x - self.dict) ** 2)

    def forward(self, x):
        x = self.kernel(x).unsqueeze(-1).unsqueeze(-1) # run activation, output shape batch, features, y, x, basis
        x = x.reshape(x.shape[0],x.shape[1]*x.shape[2],x.shape[3],x.shape[4]) # concatenate basis functions with filters
        x = self.linear(x).squeeze(-1).squeeze(-1) # apply linear coefficients and sum

        #y = torch.zeros((x.shape[0], self.channels, x.shape[-2], x.shape[-1])).cuda() #initialize output
        #for i in range(self.channels):
        #    y[:,i,:,:] = self.linear[i](x[:,i,:,:,:]).squeeze(-1) # multiply coefficients channel-wise (probably slow)

        return x


class Activation(nn.Module):
    def __init__(self, activation_func, filters, *args, **kwargs):
        super().__init__()
        if activation_func == 'relu':
            self.activation = F.relu
        elif activation_func == 'gelu':
            self.activation = F.gelu
        elif activation_func == 'kernel':
            self.activation = kernelActivation(n_basis=20, span=4, channels=filters)

    def forward(self, input):
        return self.activation(input)


