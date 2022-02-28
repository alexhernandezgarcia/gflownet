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
    def __init__(self, config, ensembleIndex):
        self.config = config
        self.ensembleIndex = ensembleIndex
        self.config.history = min(20, self.config.proxy.max_epochs) # length of past to check
        torch.random.manual_seed(int(config.seeds.model + ensembleIndex))
        self.initModel()

    def initModel(self):
        '''
        Initialize model and optimizer
        :return:
        '''
        if self.config.proxy.model_type == 'transformer': # switch to variable-length sequence model
            self.model = transformer(self.config)
        elif self.config.proxy.model_type == 'mlp':
            self.model = MLP(self.config)
        elif self.config.proxy.model_type == 'transformer2': # upgraded self-attention model
            self.model = transformer2(self.config)
        else:
            print(self.config.proxy.model_type + ' is not one of the available models')

        if self.config.device == 'cuda':
            self.model = self.model.cuda()
        self.optimizer = optim.AdamW(self.model.parameters(), amsgrad=True)
        datasetBuilder = buildDataset(self.config)
        self.mean, self.std = datasetBuilder.getStandardization()
        self.dataset_samples, self.dataset_scores = datasetBuilder.getFullDataset()


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

            if self.config.device == 'cuda':
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

        tr, te, self.datasetSize = getDataloaders(self.config, self.ensembleIndex)

        #printRecord(f"Dataset size is: {bcolors.OKCYAN}%d{bcolors.ENDC}" %self.datasetSize)

        self.converged = 0 # convergence flag
        self.epochs = 0

        while (self.converged != 1):
            if self.epochs > 0: #  this allows us to keep the previous model if it is better than any produced on this run
                self.train_net(tr)
            else:
                self.err_tr_hist.append(0)

            self.test(te) # baseline from any prior training
            if self.err_te_hist[-1] == np.min(self.err_te_hist): # if this is the best test loss we've seen
                self.save(best=1)
            # after training at least 10 epochs, check convergence
            if self.epochs >= self.config.history:
                self.checkConvergence()

            if (self.epochs % 10 == 0) and self.config.debug:
                printRecord("Model {} epoch {} test loss {:.3f}".format(self.ensembleIndex, self.epochs, self.err_te_hist[-1]))

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
            proxy_loss = self.getLoss(trainData)
            err_tr.append(proxy_loss.data)  # record the loss

            self.optimizer.zero_grad()  # run the optimizer
            proxy_loss.backward()
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
        if self.config.device == 'cuda':
            inputs = inputs.cuda()
            targets = targets.cuda()

        output = self.model(inputs.float())
        targets = (targets - self.mean)/self.std # standardize the targets during training
        return F.smooth_l1_loss(output[:,0], targets.float())


    def getMinF(self):
        inputs = self.dataset_samples
        if self.config.device == 'cuda':
            inputs = torch.Tensor(inputs).cuda()

        outputs = l2r(self.model(inputs))
        self.best_f = np.percentile(outputs, self.config.al.EI_percentile)


    def checkConvergence(self):
        """
        check if we are converged
        condition: test loss has increased or levelled out over the last several epochs
        :return: convergence flag
        """
        # check if test loss is increasing for at least several consecutive epochs
        eps = 1e-4 # relative measure for constancy

        if all(np.asarray(self.err_te_hist[-self.config.history+1:])  > self.err_te_hist[-self.config.history]): #
            self.converged = 1
            printRecord(bcolors.WARNING + "Model converged after {} epochs - test loss increasing at {:.4f}".format(self.epochs + 1, min(self.err_te_hist)) + bcolors.ENDC)

        # check if test loss is unchanging
        if abs(self.err_te_hist[-self.config.history] - np.average(self.err_te_hist[-self.config.history:]))/self.err_te_hist[-self.config.history] < eps:
            self.converged = 1
            printRecord(bcolors.WARNING + "Model converged after {} epochs - hit test loss convergence criterion at {:.4f}".format(self.epochs + 1, min(self.err_te_hist)) + bcolors.ENDC)

        if self.epochs >= self.config.proxy.max_epochs:
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
        if self.config.device == 'cuda':
            Data = torch.Tensor(Data).cuda().float()
        else:
            Data = torch.Tensor(Data).float()

        if self.config.proxy.uncertainty_estimation == "ensemble":
            self.model.train(False)
            with torch.no_grad():  # we won't need gradients! no training just testing
                outputs = self.model(Data)
                mean = torch.mean(outputs,dim=1).cpu().detach().numpy()
                std = torch.std(outputs,dim=1).cpu().detach().numpy()
        elif self.config.proxy.uncertainty_estimation == "dropout":
            self.model.train(True) # need this to be true to activate dropout
            with torch.no_grad():
                outputs = torch.hstack([self.model(Data) for _ in range(self.config.proxy.dropout_samples)])
            mean = torch.mean(outputs, dim=1).cpu().detach().numpy()
            std = torch.std(outputs, dim=1).cpu().detach().numpy()
        else:
            print("No uncertainty estimator called {}".format(self.config.proxy.uncertainty_estimation))
            sys.exit()

        if output == 'Average':
            return mean * self.std + self.mean
        elif output == 'Uncertainty':
            return std * self.std
        elif output == 'Both':
            return mean * self.std + self.mean, std * self.std
        elif output == 'fancy_acquisition':
            if self.config.al.acquisition_function.lower() == 'ucb':
                mean = mean * self.std + self.mean
                std = std * self.std
                score = mean + self.config.al.UCB_kappa * std
                score = l2r(torch.Tensor(score))
                return score, mean * self.std + self.mean, std * self.std
            elif self.config.al.acquisition_function.lower() == 'ei':
                try:
                    if self.best_f == 'canoe': # I just want it to load for goodness sake
                        pass
                except:
                    self.getMinF()

                outputs = l2r(outputs)
                mean, std = torch.mean(torch.Tensor(outputs),dim=1), torch.std(torch.Tensor(outputs),dim=1)
                u = torch.tensor((mean - self.best_f) / (std + 1e-4))
                u = -u  # we are minimizing # MK double-check on this
                normal = torch.distributions.Normal(torch.zeros_like(u), torch.ones_like(u))
                ucdf = normal.cdf(u)
                updf = torch.exp(normal.log_prob(u))
                ei = std * (updf + u * ucdf)
                return ei.cpu().detach().numpy(), mean * self.std + self.mean, std * self.std


    def raw(self, Data, output="Average"):
        '''
        evaluate the model
        output types - if "Average" return the average of ensemble predictions
            - if 'Variance' return the variance of ensemble predictions
        # future upgrade - isolate epistemic uncertainty from intrinsic randomness
        :param Data: input data
        :return: model scores
        '''
        if self.config.device == 'cuda':
            Data = torch.Tensor(Data).cuda().float()
        else:
            Data = torch.Tensor(Data).float()

        self.model.train(False)
        with torch.no_grad():  # we won't need gradients! no training just testing
            out = self.model(Data).cpu().detach().numpy()
            if output == 'Average':
                return np.average(out,axis=1)
            elif output == 'Variance':
                return np.var(out,axis=1)
            elif output == 'Both':
                return np.average(out,axis=1), np.var(out,axis=1)

    def loadEnsemble(self,models):
        '''
        load up a model ensemble
        :return:
        '''
        self.model = modelEnsemble(models)
        if self.config.device == 'cuda':
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
    def __init__(self, config):
        dataset = np.load('datasets/' + config.dataset.oracle + '.npy', allow_pickle=True)
        dataset = dataset.item()
        self.samples = dataset['samples']
        self.targets = dataset['scores']

        self.samples, self.targets = shuffle(self.samples, self.targets, random_state=config.seeds.dataset)

    def reshuffle(self, seed=None):
        self.samples, self.targets = shuffle(self.samples, self.targets, random_state=seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]

    def returnScores(self):
        return self.targets

    def getFullDataset(self):
        return self.samples, self.targets

    def getStandardization(self):
        return np.mean(self.targets), np.sqrt(np.var(self.targets))


def getDataloaders(config, ensembleIndex): # get the dataloaders, to load the dataset in batches
    '''
    creat dataloader objects from the dataset
    :param config:
    :return:
    '''
    training_batch = config.proxy.mbsize
    dataset = buildDataset(config)  # get data
    if config.proxy.shuffle_dataset:
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


def getDataSize(config):
    dataset = np.load('datasets/' + config.dataset.oracle + '.npy', allow_pickle=True)
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
    def __init__(self,config):
        super(transformer,self).__init__()

        self.embedDim = config.proxy.width
        self.hiddenDim = config.proxy.width
        self.layers = config.proxy.n_layers
        self.maxLen = config.dataset.max_length
        self.dictLen = config.dataset.dict_size
        self.classes = int(config.dataset.dict_size + 1)
        self.heads = min([4, max([1,self.embedDim//self.dictLen])])

        self.positionalEncoder = PositionalEncoding(self.embedDim, max_len = self.maxLen, dropout=0)
        self.embedding = nn.Embedding(self.dictLen + 1, embedding_dim = self.embedDim)

        factory_kwargs = {'device': None, 'dtype': None}
        #encoder_layer = nn.TransformerEncoderLayer(self.embedDim, nhead = self.heads,dim_feedforward=self.hiddenDim, activation='gelu', dropout=0)
        #self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = self.layers)
        self.decoder_layers = []
        self.encoder_linear = []
        self.self_attn_layers = []
        self.decoder_dropouts = []
        for i in range(self.layers):
            self.encoder_linear.append(nn.Linear(self.embedDim,self.embedDim))
            self.self_attn_layers.append(nn.MultiheadAttention(self.embedDim, self.heads, dropout=config.proxy.dropout, batch_first=False, **factory_kwargs))

            if i == 0:
                in_dim = self.embedDim
            else:
                in_dim = self.hiddenDim
            out_dim = self.hiddenDim
            self.decoder_layers.append(nn.Linear(in_dim, out_dim))
            self.decoder_dropouts.append(nn.Dropout(config.proxy.dropout))

        self.decoder_layers = nn.ModuleList(self.decoder_layers)
        self.decoder_dropouts = nn.ModuleList(self.decoder_dropouts)
        self.encoder_linear = nn.ModuleList(self.encoder_linear)
        self.self_attn_layers = nn.ModuleList(self.self_attn_layers)
        self.output_layer = nn.Linear(self.hiddenDim,self.classes,bias=False)

    def forward(self,x):
        x_key_padding_mask = (x==0).clone().detach() # zero out the attention of empty sequence elements
        x = self.embedding(x.transpose(1,0).int()) # [seq, batch]
        x = self.positionalEncoder(x)
        #x = self.encoder(x,src_key_padding_mask=x_key_padding_mask)
        #x = x.permute(1,0,2).reshape(x_key_padding_mask.shape[0], int(self.embedDim*self.maxLen))
        for i in range(len(self.self_attn_layers)):
            x = self.self_attn_layers[i](x,x,x,key_padding_mask=x_key_padding_mask)[0]
            x = self.encoder_linear[i](x)

        x = x.mean(dim=0) # mean aggregation
        for i in range(len(self.decoder_layers)):
            x = F.gelu(self.decoder_layers[i](x))
            x = self.decoder_dropouts[i](x)

        x = self.output_layer(x)

        return x


class transformer2(nn.Module):
    def __init__(self,config):
        super(transformer2,self).__init__()

        self.embedDim = config.proxy.width
        self.filters = config.proxy.width
        self.encoder_layers = config.proxy.n_layers
        self.decoder_layers = 1
        self.maxLen = config.dataset.max_length
        self.dictLen = config.dataset.dict_size
        self.proxy_aggregation = 'sum'
        self.proxy_attention_norm = 'layer'
        self.proxy_norm = 'layer'
        self.classes = int(config.dataset.dict_size + 1)
        self.heads = max([4, max([1,self.embedDim//self.dictLen])])
        self.relative_attention = True
        act_func = 'gelu'

        self.positionalEncoder = PositionalEncoding(self.embedDim, max_len = self.maxLen, dropout=config.proxy.dropout)
        self.embedding = nn.Embedding(self.dictLen + 1, embedding_dim = self.embedDim)

        factory_kwargs = {'device': None, 'dtype': None}
        #encoder_layer = nn.TransformerEncoderLayer(self.embedDim, nhead = self.heads,dim_feedforward=self.filters, activation='gelu', dropout=0)
        #self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = self.layers)
        self.decoder_linear = []
        self.encoder_norms1 = []
        self.encoder_norms2 = []
        self.decoder_norms = []
        self.encoder_dropouts = []
        self.decoder_dropouts = []
        self.encoder_linear1 = []
        self.encoder_linear2 = []
        self.self_attn_layers = []
        self.aggregation_mode = self.proxy_aggregation
        self.encoder_activations = []
        self.decoder_activations = []

        for i in range(self.encoder_layers):
            self.encoder_linear1.append(nn.Linear(self.embedDim,self.embedDim))
            self.encoder_linear2.append(nn.Linear(self.embedDim,self.embedDim))

            if not self.relative_attention:
                self.self_attn_layers.append(nn.MultiheadAttention(self.embedDim, self.heads, dropout=config.proxy.dropout, batch_first=False, **factory_kwargs))
            else:
                self.self_attn_layers.append(RelativeGlobalAttention(self.embedDim, self.heads, dropout=config.proxy.dropout, max_len=self.maxLen))

            self.encoder_activations.append(Activation(act_func, self.filters))

            if config.proxy.dropout != 0:
                self.encoder_dropouts.append(nn.Dropout(config.proxy.dropout))
            else:
                self.encoder_dropouts.append(nn.Identity())

            if self.proxy_attention_norm == 'layer': # work in progress
                self.encoder_norms1.append(nn.LayerNorm(self.embedDim))
                self.encoder_norms2.append(nn.LayerNorm(self.embedDim))

            else:
                self.encoder_norms1.append(nn.Identity())
                self.encoder_norms2.append(nn.Identity())


        for i in range(self.decoder_layers):
            if i == 0:
                self.decoder_linear.append(nn.Linear(self.embedDim, self.filters))
            else:
                self.decoder_linear.append(nn.Linear(self.filters, self.filters))

            self.decoder_activations.append(Activation(act_func,self.filters))
            if config.proxy.dropout != 0:
                self.decoder_dropouts.append(nn.Dropout(config.proxy.dropout))
            else:
                self.decoder_dropouts.append(nn.Identity())

            if self.proxy_norm == 'batch':
                self.decoder_norms.append(nn.BatchNorm1d(self.filters))
            elif self.proxy_norm == 'layer':
                self.decoder_norms.append(nn.LayerNorm(self.filters))
            else:
                self.decoder_norms.append(nn.Identity())

        self.decoder_linear = nn.ModuleList(self.decoder_linear)
        self.encoder_linear1 = nn.ModuleList(self.encoder_linear1)
        self.encoder_linear2 = nn.ModuleList(self.encoder_linear2)

        self.self_attn_layers = nn.ModuleList(self.self_attn_layers)
        self.encoder_norms1 = nn.ModuleList(self.encoder_norms1)
        self.encoder_norms2 = nn.ModuleList(self.encoder_norms2)
        self.decoder_norms = nn.ModuleList(self.decoder_norms)
        self.encoder_dropouts = nn.ModuleList(self.encoder_dropouts)
        self.decoder_dropouts = nn.ModuleList(self.decoder_dropouts)
        self.encoder_activations = nn.ModuleList(self.encoder_activations)
        self.decoder_activations = nn.ModuleList(self.decoder_activations)

        self.output_layer = nn.Linear(self.filters,1,bias=False)

    def forward(self,x, clip = None):
        x_key_padding_mask = (x==0).clone().detach() # zero out the attention of empty sequence elements
        x = self.embedding(x.transpose(1,0).int()) # [seq, batch]

        for i in range(self.encoder_layers):
            # Self-attention block
            residue = x.clone()
            x = self.encoder_norms1[i](x)
            if not self.relative_attention:
                x = self.self_attn_layers[i](x,x,x,key_padding_mask=x_key_padding_mask)[0]
            else:
                x = self.self_attn_layers[i](x.transpose(1,0)).transpose(1,0) # pairwise relative position encoding embedded in the self-attention block
            x = self.encoder_dropouts[i](x)
            x = x + residue

            # dense block
            residue = x.clone()
            x = self.encoder_linear1[i](x)
            x = self.encoder_norms2[i](x)
            x = self.encoder_activations[i](x)
            x = self.encoder_linear2[i](x)
            x = x + residue

        if self.aggregation_mode == 'mean':
            x = x.mean(dim=0) # mean aggregation
        elif self.aggregation_mode == 'sum':
            x = x.sum(dim=0) # sum aggregation
        elif self.aggregation_mode == 'max':
            x = x.max(dim=0) # max aggregation
        else:
            print(self.aggregation_mode + ' is not a valid aggregation mode!')

        for i in range(self.decoder_layers):
            if i != 0:
                residue = x.clone()
            x = self.decoder_linear[i](x)
            x = self.decoder_norms[i](x)
            x = self.decoder_dropouts[i](x)
            x = self.decoder_activations[i](x)
            if i != 0:
                x += residue

        x = self.output_layer(x)

        if clip is not None:
            x = torch.clip(x,max=clip)

        return x


class RelativeGlobalAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_len=1024, dropout=0.1):
        super().__init__()
        d_head, remainder = divmod(d_model, num_heads)
        if remainder:
            raise ValueError(
                "incompatible `d_model` and `num_heads`"
            )
        self.max_len = max_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(max_len, d_head))
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_len, max_len))
                .unsqueeze(0).unsqueeze(0)
        )
        # self.mask.shape = (1, 1, max_len, max_len)

    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape

        if seq_len > self.max_len:
            raise ValueError(
                "sequence length exceeds model capacity"
            )

        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape = (batch_size, num_heads, seq_len, d_head)

        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)

        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, float("-inf"))
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        return self.dropout(out)

    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel


class MLP(nn.Module):
    def __init__(self,config):
        super(MLP,self).__init__()
        # initialize constants and layers

        if True:
            act_func = 'gelu'

        self.inputLength = config.dataset.max_length
        self.tasks = config.dataset.sample_tasks
        self.layers = config.proxy.n_layers
        self.filters = config.proxy.width
        self.classes = int(config.dataset.dict_size + 1)
        self.init_layer_depth = int(self.inputLength * self.classes)

        # build input and output layers
        self.initial_layer = nn.Linear(int(self.inputLength * self.classes), self.filters) # layer which takes in our sequence in one-hot encoding
        self.activation1 = Activation(act_func,self.filters,config)

        self.output_layers = []
        for i in range(self.tasks):
            self.output_layers.append(nn.Linear(self.filters, 1))
        self.output_layers = nn.ModuleList(self.output_layers)

        # build hidden layers
        self.lin_layers = []
        self.activations = []
        self.norms = []
        self.dropouts = []

        for i in range(self.layers):
            self.lin_layers.append(nn.Linear(self.filters,self.filters))
            self.activations.append(Activation(act_func, self.filters))
            #self.norms.append(nn.BatchNorm1d(self.filters))
            self.dropouts.append(nn.Dropout(p=config.proxy.dropout))

        # initialize module lists
        self.lin_layers = nn.ModuleList(self.lin_layers)
        self.activations = nn.ModuleList(self.activations)
        #self.norms = nn.ModuleList(self.norms)
        self.dropouts = nn.ModuleList(self.dropouts)


    def forward(self, x):
        x = F.one_hot(x.long(),num_classes=self.classes)
        x = x.reshape(x.shape[0], self.init_layer_depth).float()
        x = self.activation1(self.initial_layer(x)) # apply linear transformation and nonlinear activation
        for i in range(self.layers):
            x = self.lin_layers[i](x)
            x = self.activations[i](x)
            x = self.dropouts[i](x)
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



class UCB:
    def __init__(self, model, kappa, device):
        self.kappa = kappa
        self.model = model
        self.device = device
        self.sigmoid = nn.Sigmoid()

    def __call__(self, x1, x2, **f_kwargs):
        outputs = self.model(x1, x2)
        outputs = torch.cat(outputs)
        mean = outputs.mean(dim=0)
        std = outputs.std(dim=0)
        return self.l2r(torch.tensor([[(mean + self.kappa * std)]]).to(self.device), **f_kwargs)

    def l2r(self, x, **f_kwargs):
        return self.sigmoid(x.clamp(min=f_kwargs["r_min"])) / f_kwargs["r_norm"]

class EI:
    def __init__(self, config, model, maximize=False):
        #tokenizer = pickle.load(gzip.open('tokenizer.pkl.gz', 'rb'))
        #self.model = model
        #self.device = device
        self.maximize = maximize
        self.sigmoid = nn.Sigmoid()
        self.best_f = self._get_best_f(dataset, tokenizer)

    def _get_best_f(self, dataset, tok):
        f_values = []
        for sample in dataset.pos_train:
            x = tok.process([sample]).to(self.device)
            # ys = self.model(x.swapaxes(0,1), x.lt(2), **self.f_kwargs)
            outputs = self.model(x.swapaxes(0,1), x.lt(2))
            outputs = self.sigmoid(torch.cat(outputs))
            mean, _ = outputs.mean(dim=0), outputs.std(dim=0)
            f_values.append(mean.item())
        return torch.tensor(np.percentile(f_values, args.max_percentile))

    def __call__(self, x1, x2, **f_kwargs):
        self.best_f = self.best_f.to(x1)

        outputs = self.model(x1, x2)
        outputs = torch.cat([self.l2r(outputs[i].unsqueeze(0), **f_kwargs) for i in range(len(outputs))])
        outputs = outputs.swapaxes(0, 1)
        mean, sigma = outputs.mean(dim=1).unsqueeze(-1), outputs.std(dim=1).unsqueeze(-1)
        # deal with batch evaluation and broadcasting
        view_shape = mean.shape[:-2] if mean.dim() >= x1.dim() else x1.shape[:-2]
        mean = mean.view(view_shape)
        sigma = sigma.view(view_shape)

        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = torch.distributions.Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)
        return ei.cpu().numpy()

    def l2r(self, x, **f_kwargs):
        return self.sigmoid(x.clamp(min=f_kwargs["r_min"])) / f_kwargs["r_norm"]


def l2r(x):
    r_max = 0
    r_norm = 1
    score = torch.clip(x, min=-np.inf, max=r_max).sigmoid() / r_norm
    return score.cpu().detach().numpy()