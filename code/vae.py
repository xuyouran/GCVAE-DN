from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import json
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import math
import networkx as nx

            
def create_graph(data:np.ndarray, norm=True):
    zeros = np.where(data==0)
    if norm:
        norm_data = np.tril(data, -1)
        norm_data = norm_data[np.where(norm_data!=0)]
        min_ = np.min(norm_data)
        max_ = np.max(norm_data)
        data = (data - min_)/(max_ - min_)
    data[zeros] = 0
    h, w = data.shape
    assert h == w
    G = nx.Graph() 
    G.add_nodes_from([str(i) for i in range(h)])
    G.add_weighted_edges_from([(str(i), str(i), 1.) for i in range(h)])
    where = np.where(data != 0)
    G.add_weighted_edges_from(zip(*[item.astype('str') for item in where], data[where]))
    return G


class KDD99(Dataset):
    def __init__(self, instances):
        self.instances = instances

    def __getitem__(self, index):
        ins = self.instances[index]
        return ins

    def __len__(self):
        return len(self.instances)


def mape_vectorized_v2(a, b): 
    mask = a != 0
    return (np.fabs(a.numpy() - b.numpy())/a.numpy())[mask].mean()


class GraphDiffusionConvolution(Module):
    def __init__(self, in_features, out_features, k=2, bias=True):
        super(GraphDiffusionConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.k = k
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.in_features*self.k)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def graph_diffusion_convolution_fn(self, inputs, adj, weight, bias=None, k=2):
        inputs = inputs.unsqueeze(dim=2)

        batch_size = inputs.shape[0]
        support = torch.bmm(inputs, weight.expand(batch_size, -1, -1))
        output = support.clone()
        adj_ = adj
        for i in range(k):
            output += torch.bmm(adj_.expand(batch_size, -1, -1), support)
            adj_ = torch.spmm(adj_, adj)
        if bias is not None:
            output = output + bias
        output = output.squeeze(dim=2)
        return output
    
    def forward(self, inputs, DW):
        return self.graph_diffusion_convolution_fn(inputs, DW, self.weight, self.bias, self.k)
    
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class VAE(nn.Module):
    def __init__(self, nodeNum, in_features, out_features, k=2, bias=True):
        super(VAE, self).__init__()
        self.nodeNum = nodeNum
        self.gcn1 = GraphDiffusionConvolution(in_features, out_features, k=2, bias=True)
        self.fc1 = nn.Linear(nodeNum,  60) 
        self.fc2 = nn.Linear(60,  30)
        self.fc3 = nn.Linear(30,  10)
        self.fc41 = nn.Linear(10, 5)
        self.fc42 = nn.Linear(10, 5)
        self.fc5 = nn.Linear(5, 10)
        self.fc6 = nn.Linear(10, 30)
        self.fc7 = nn.Linear(30, 60)
        self.fc8 = nn.Linear(60, nodeNum) 
        self.gcn2 = GraphDiffusionConvolution(out_features, in_features, k=2, bias=True)

    def encode(self, x, adj):
        h1 = F.relu(self.fc1(self.gcn1(x, adj)))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return self.fc41(h3), self.fc42(h3)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, adj):
        h5 = F.relu(self.fc5(z))
        h6 = F.relu(self.fc6(h5))
        h7 = F.relu(self.fc7(h6))
        return self.gcn2(self.fc8(h7), adj)

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, adj)
        return x_hat, mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    Mse = torch.nn.MSELoss(reduce=False, size_average=False)
    MSE = Mse(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    totalLoss = MSE + 0.00000001*KLD # kdd99
    # totalLoss = MSE + 0.0001*KLD # thyroid
    # totalLoss = MSE + 0.00000001*KLD # letter
    return MSE, KLD, totalLoss


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data) in enumerate(train_loader):
        data = data.to(device)
        
        recon_batch, mu, logvar = model(data, adj)
        
        mse, kld, loss = loss_function(recon_batch, data, mu, logvar)
        loss = loss.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        num = len(data) * len(data[0])
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tMSE: {:.6f}\tKLD: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item() / num, mse.sum() / num, kld.sum() / num))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
    test_loss = 0

    fin_mu = []
    fin_logvar = []
    with torch.no_grad():
        for i, (data) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data, adj)
            mse, kld, loss = loss_function(recon_batch, data, mu, logvar)
            
            loss = loss.sum().item()
            test_loss += loss

            fin_mu.extend(mu.tolist())
            fin_logvar.extend(logvar.tolist())
    test_loss /= len(test_loader.dataset)

    print('====> Test set loss: {:.4f}'.format(test_loss))
    return fin_mu, fin_logvar


def generatePara():
    
    model.eval()
    fin_mu = []
    fin_std = []
    with torch.no_grad():
        for i, (data) in enumerate(test_loader):
            data = data.to(device)
            for instance in data:
                instance = instance.unsqueeze(dim=0)
                mu, logvar = model.encode(instance, adj)
                mu_hat = model.decode(mu, adj)
                mse, kld, loss = loss_function(mu_hat, instance, mu, logvar)
                loss = mse
                std = torch.exp(0.5*logvar)
                std = std
                fin_mu.append(loss.tolist())
                fin_std.append(std.tolist())
    np.save(datadir + 'mu_loss.npy', np.array(fin_mu))
    np.save(datadir + 'std.npy', np.array(fin_std))


def get_data(model):
    return list(map(lambda x: x.data, model.parameters()))


if __name__ == "__main__":
    datadir = '../data/thyroid/'
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    train_instances = np.load(datadir + 'X_train.npy')
    train_labels = np.load(datadir + 'y_train.npy')
    test_instances = np.load(datadir + 'X_test.npy')
    test_labels = np.load(datadir + 'y_test.npy')
    nodeNum = train_instances.shape[1]

    graph_data = np.load(datadir + 'thyroid.npy')
    graph_data = graph_data + graph_data.T
    graph_data += np.eye(graph_data.shape[0])
    graph_data = np.nan_to_num(graph_data)
    G = create_graph(graph_data[:nodeNum, :nodeNum])
    adj = np.array(nx.normalized_laplacian_matrix(G).todense())
    adj = torch.FloatTensor(adj).to(device)
    adj.requires_grad = False

    train_dataset = KDD99(torch.FloatTensor(train_instances))
    test_dataset = KDD99(torch.FloatTensor(test_instances))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    print('data prepared!\n')

    model = VAE(nodeNum, 1, 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)
        
    generatePara()

