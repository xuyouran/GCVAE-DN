from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import json
from torch.distributions.normal import Normal
from scipy import stats


class anomalyScore(Dataset):
    def __init__(self, instances, labels):
        self.instances = instances
        self.labels = labels

    def __getitem__(self, index):
        ins = self.instances[index]
        lab = self.labels[index]
        return ins, lab

    def __len__(self):
        return len(self.instances)

train_instances = np.load('../data/activity/training_data.npy')
train_labels = np.load('../data/activity/traning_label.npy')
test_instances = np.load('../data/activity/testing_data.npy')
test_labels = np.load('../data/activity/testing_label.npy')
nodeNum = train_instances.shape[1]

train_dataset = anomalyScore(torch.FloatTensor(train_instances), torch.LongTensor(train_labels))
test_dataset = anomalyScore(torch.FloatTensor(test_instances), torch.LongTensor(test_labels))

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                    help='input batch size for training (default: 1024)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
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

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

print('data prepared!\n')

class Model1(nn.Module):
    def __init__(self, nodeNum):
        super(Model1, self).__init__()
        self.nodeNum = nodeNum
        self.bn = nn.BatchNorm1d(self.nodeNum)
        self.fc1 = nn.Linear(self.nodeNum, self.nodeNum // 2)
        self.fc2 = nn.Linear(self.nodeNum // 2, self.nodeNum // 4)
        self.fc3 = nn.Linear(self.nodeNum // 4, 10)
        self.fc4 = nn.Linear(10, 5)
    
    def embedding(self, x):
        h1 = torch._C._nn.elu(self.fc1(self.bn(x)))
        h2 = torch._C._nn.elu(self.fc2(h1))
        h3 = torch._C._nn.elu(self.fc3(h2))
        return torch._C._nn.elu(self.fc4(h3))
    
    def forward(self, x):
        return self.embedding(x)

class Model2(nn.Module):
    def __init__(self, nodeNum):
        super(Model2, self).__init__()
        self.nodeNum = nodeNum
        self.bn = nn.BatchNorm1d(self.nodeNum)
        self.fc1 = nn.Linear(self.nodeNum, self.nodeNum // 2)
        self.fc2 = nn.Linear(self.nodeNum // 2, 2)
    
    def embedding(self, x):
        h1 = torch._C._nn.elu(self.fc1(self.bn(x)))
        return torch._C._nn.elu(self.fc2(h1))
    
    def forward(self, x):
        return self.embedding(x)

class Score(nn.Module):
    def __init__(self, nodeNum1, nodeNum2):
        super(Score, self).__init__()
        self.nodeNum1 = nodeNum1
        self.nodeNum2 = nodeNum2
        self.embedding1 = Model1(self.nodeNum1)
        self.embedding2 = Model2(self.nodeNum2)
        self.bn = nn.BatchNorm1d(7)
        self.fc1 = nn.Linear(7,  4)
        self.fc2 = nn.Linear(4,  2)

    def classification(self, x):
        h1 = torch._C._nn.elu(self.embedding1(x[:,:self.nodeNum1]))
        h2 = torch._C._nn.elu(self.embedding2(x[:,-self.nodeNum2:]))
        h3 = torch.cat((h1, h2), dim=1)
        h4 = torch._C._nn.elu(self.fc1(self.bn(h3)))
        
        return F.softmax(self.fc2(h4))


    def forward(self, x):
        return self.classification(x)


# model = Score(121, 5).to(device) #kdd99 5-2 8epoch for result 10epoch for picture shuffle=true
# model = Score(41, 5).to(device) #letter 4-3 500epoch for result 1000epoch for picture shuffle=false
model = Score(18, 5).to(device) #thyroid 5-2 500epoch for result 500epoch for picture shuffle=false
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss = loss.sum()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        if batch_idx % args.log_interval == 0:    # print every 2000 mini-batches
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item() / len(data)))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data = data.to(device)
            outputs = model(data)
            if i == 0:
                pred = outputs
                observe = labels
                testdata = data
            else:
                pred = torch.cat((pred, outputs), 0)
                observe = torch.cat((observe, labels), 0)
                testdata = torch.cat((testdata, data), 0)
            loss = criterion(outputs, labels)
            test_loss += loss
    test_loss /= len(test_loader.dataset)
    value, pred = torch.max(pred, dim=1)
    true = len(np.where((pred - observe).numpy() == 0)[0])
    observe_anomaly = set(np.where(observe.numpy() == 0)[0])
    predict_anomaly = set(np.where(pred.numpy() == 0)[0])
    print(len(observe_anomaly), len(predict_anomaly), len(predict_anomaly & observe_anomaly))
    if len(observe_anomaly) == 0 or len(predict_anomaly) == 0 or len(predict_anomaly & observe_anomaly) == 0:
        return
    recall = len(predict_anomaly & observe_anomaly) / len(observe_anomaly)
    presicion = len(predict_anomaly & observe_anomaly) / len(predict_anomaly)
    f1 = 2*recall*presicion / (recall + presicion)
    print(f" Precision = {presicion:.3f}")
    print(f" Recall    = {recall:.3f}")
    print(f" F1-Score  = {f1:.3f}")
    print('correct:', true, 'precision:', true / len(observe))

    print('====> Test set loss: {:.4f}'.format(test_loss))

    # np.save('../data/kddcup99/pred.npy', pred.numpy())
    # np.save('../data/kddcup99/observe.npy', observe.numpy())
    # np.save('../data/kddcup99/data.npy', testdata.numpy())

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test(epoch)

