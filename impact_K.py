# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:40:57 2022

@author: ChandlerZhu
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.utils.data.sampler as sampler
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import copy
transform=torchvision.transforms.Compose(
[transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
n_epochs = 3
num_straggler = 4
batch_size_train = 100
batch_size_test = 400
learning_rate = 0.3
momentum = 0
log_interval = 10
M = 20
EPS = 0.00008
iters = 400
typeMNIST = 'balanced'
length_out = 47
miu = 1e4
S = 10
scale = 2
miu_e = 1e-4
# transform=torchvision.transforms.Compose(
# [transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
dataset_workers = []
sampler_workers = []
loader_workers = []
'''dataset_global = torchvision.datasets.MNIST('./dataset/', train=True, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ]))'''
dataset_global = torchvision.datasets.CIFAR10(
                            root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download = False,
                            )
length = len(dataset_global)
# index = dataset_global.targets.argsort()
# index = np.array(dataset_global.targets).argsort()
# index = torch.randperm(length)
index = np.array(torch.randperm(length))

for i in range(M):
    '''dataset_workers.append(torchvision.datasets.MNIST('./dataset/', train=True, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])))'''
    dataset_workers.append(torchvision.datasets.CIFAR10(
                            root='./data',
                            train=True,
                            transform=transform,
                            download = False,
                            ))
    '''dataset_workers.append(torchvision.datasets.MNIST('./dataset/', train=True, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ])))'''

    dataset_workers[i].data, dataset_workers[i].targets = dataset_workers[i].data[index], np.array(dataset_workers[i].targets)[index].tolist()
    # dataset_workers[i].data, dataset_workers[i].targets = dataset_workers[i].data[index], dataset_workers[i].targets[index]
    dataset_workers[i].data, dataset_workers[i].targets = dataset_workers[i].data[int(length / M * i) : int(length / M * (i + 1))]\
    , dataset_workers[i].targets[int(length / M * i) : int(length / M * (i + 1))]

    sampler_workers.append(sampler.BatchSampler(sampler.RandomSampler(data_source=dataset_workers[i], replacement=True), batch_size=batch_size_train, drop_last=False))
    loader_workers.append(torch.utils.data.DataLoader(dataset_workers[i],batch_sampler=sampler_workers[i], shuffle=False))

'''test_dataset = torchvision.datasets.MNIST('./dataset/', train=False, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))'''
test_dataset = torchvision.datasets.CIFAR10(
                            root='./data',
                            train=False,
                            transform=transform,
                            download = False,
                            )
'''test_dataset = torchvision.datasets.MNIST('./dataset/', train=False, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor()
                             ]))'''

#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
test_sampler = sampler.BatchSampler(sampler.RandomSampler(data_source=test_dataset, replacement=False), batch_size=batch_size_test, drop_last=False)
test_loader = torch.utils.data.DataLoader(test_dataset
  ,batch_sampler=test_sampler)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n_cls = 10
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*5*5, 384) 
        self.fc2 = nn.Linear(384, 192) 
        self.fc3 = nn.Linear(192, self.n_cls)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
criterion = nn.CrossEntropyLoss()
network1 = Net()


acc = 70

#%% ADA_Wireless_modified1
# iters = iters * 3
# iters3 = 800
iters3 = 6000
S0 = 1
S = S0
K0 = 10
K = K0
t_ADA_Wireless_modified1 = [0]*(iters3+1)
comm_ADA_Wireless_modified1 = [0]*(iters3+1)
train_losses_ADA_Wireless_modified1 = []
test_losses_ADA_Wireless_modified1 = []
network = copy.deepcopy(network1)
# optimizer = optim.Adam(network.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer = optim.SGD(network.parameters(), lr=1, momentum=momentum)
network.train()
optimizer_worker = []
for m in range(M):
    optimizer_worker.append(0)
for i in range(iters3):
    if i == 0:
        for batch_idx, (data, target) in enumerate(test_loader):
            output = network(data)
            # output = network(data)
            loss = criterion(output, target)
            break
        loss_orig = loss.item()
    if i % 10 == 0:
        network.eval()
        test_loss_ADA_Wireless_modified1 = 0
        correct_ADA_Wireless_modified1 = 0
        with torch.no_grad():
          for data, target in test_loader:
            output = network(data)
            # output = network(data)
            test_loss_ADA_Wireless_modified1 += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct_ADA_Wireless_modified1 += pred.eq(target.data.view_as(pred)).sum()
        test_loss_ADA_Wireless_modified1 /= len(test_loader.dataset)
        test_losses_ADA_Wireless_modified1.append(100. * correct_ADA_Wireless_modified1 / len(test_loader.dataset))
    time_workers1 = {}
    for k in range(K):
        time_workers1[k]=0
    for k in range(K):
        time_workers1[k] += np.random.exponential(miu_e, M)
        # time_workers1[k] += np.random.pareto(miu, M)
        time_workers1[k+1] = copy.deepcopy(time_workers1[k])
    time_workers = np.sort(time_workers1[K-1])
    selection = list(np.argsort(time_workers1[K-1]))[0:S]
    time_workers = time_workers[0:S]
    time_max = np.max(time_workers)
    # selection = list(range(M))
    noselection = list(set(list(range(M))).difference(set(selection)))
    grad_workers= {}
    for m in selection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum) 
        time_flag = time_workers1[K-1][m]
        rounds = 0
        # local_rounds = int(np.max(time_workers) / time_workers[flag])
        for aa in range(10000):
            for batch_idx, (data, target) in enumerate(loader_workers[m]):
                optimizer_worker[m].zero_grad()
                output = network_worker(data)
                # output = network_worker(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer_worker[m].step()
                if aa == 0 and batch_idx <= K-2:
                    continue
                else:
                    time_flag += np.random.exponential(miu_e, [1, 1])[0]

                # local_rounds = int(np.max(time_workers) / time_workers[m] - 1)
                if time_flag > time_max:
                    grad_workers[m]=optimizer_worker[m].param_groups[0]['params']
                    break
            if time_flag >= time_max:
                break
    for m in noselection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum) 
        time_flag = time_workers1[K-1][m]
        for batch_idx, (data, target) in enumerate(loader_workers[m]):
            optimizer_worker[m].zero_grad()
            output = network_worker(data)
            # output = network_worker(data)
            loss = criterion(output, target)
            loss.backward()
            if time_workers1[batch_idx][m] > time_max:
                break
            optimizer_worker[m].step()
            if time_workers1[batch_idx+1][m] > time_max:
                grad_workers[m]=optimizer_worker[m].param_groups[0]['params']
                selection.append(m)
                break
    temp = [0]*10
    for l in range(10):
        count = 0
        for m in selection:
            temp[l] += grad_workers[m][l]
            count += 1
        temp[l] /= count
    
    for batch_idx, (data, target) in enumerate(test_loader):
        optimizer.zero_grad()
        for l in range(10):
            optimizer.param_groups[0]['params'][l].grad = -temp[l] + optimizer.param_groups[0]['params'][l]
        optimizer.step()
        output = network(data)
        # output = network(data)
        loss = criterion(output, target)
        break
    # optimizer.param_groups[0]['lr'] *= 0.99
    print(loss.item(), "STSyn")
    print("E=", S, "K=", K)
    train_losses_ADA_Wireless_modified1.append(loss.item())
    t_ADA_Wireless_modified1[i] += time_max
    if t_ADA_Wireless_modified1[i] < t_ADA_Wireless_modified1[i-1]:
        print("error!!!")
        break
    t_ADA_Wireless_modified1[i+1] = t_ADA_Wireless_modified1[i]
    comm_ADA_Wireless_modified1[i] += M + len(selection)
    comm_ADA_Wireless_modified1[i+1] = comm_ADA_Wireless_modified1[i]    
    S_old = S
    # S = min(max(int(loss_orig / loss.item()*S0),S_old), M)
    K_old = K
    #K = min(max(int((loss_orig / loss.item())*K0),K_old), M)
    if test_losses_ADA_Wireless_modified1[-1] >= acc:
        break

#%% ADA_Wireless_modified2
# iters = iters * 3
# iters3 = 800
iters3 = 6000
S0 = 5
S = S0
K0 = 10
K = K0
t_ADA_Wireless_modified2 = [0]*(iters3+1)
comm_ADA_Wireless_modified2 = [0]*(iters3+1)
train_losses_ADA_Wireless_modified2 = []
test_losses_ADA_Wireless_modified2 = []
network = copy.deepcopy(network1)
# optimizer = optim.Adam(network.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer = optim.SGD(network.parameters(), lr=1, momentum=momentum)
network.train()
optimizer_worker = []
for m in range(M):
    optimizer_worker.append(0)
for i in range(iters3):
    if i == 0:
        for batch_idx, (data, target) in enumerate(test_loader):
            output = network(data)
            # output = network(data)
            loss = criterion(output, target)
            break
        loss_orig = loss.item()
    if i % 10 == 0:
        network.eval()
        test_loss_ADA_Wireless_modified2 = 0
        correct_ADA_Wireless_modified2 = 0
        with torch.no_grad():
          for data, target in test_loader:
            output = network(data)
            # output = network(data)
            test_loss_ADA_Wireless_modified2 += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct_ADA_Wireless_modified2 += pred.eq(target.data.view_as(pred)).sum()
        test_loss_ADA_Wireless_modified2 /= len(test_loader.dataset)
        test_losses_ADA_Wireless_modified2.append(100. * correct_ADA_Wireless_modified2 / len(test_loader.dataset))
    time_workers1 = {}
    for k in range(K):
        time_workers1[k]=0
    for k in range(K):
        time_workers1[k] += np.random.exponential(miu_e, M)
        # time_workers1[k] += np.random.pareto(miu, M)
        time_workers1[k+1] = copy.deepcopy(time_workers1[k])
    time_workers = np.sort(time_workers1[K-1])
    selection = list(np.argsort(time_workers1[K-1]))[0:S]
    time_workers = time_workers[0:S]
    time_max = np.max(time_workers)
    # selection = list(range(M))
    noselection = list(set(list(range(M))).difference(set(selection)))
    grad_workers= {}
    for m in selection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum) 
        time_flag = time_workers1[K-1][m]
        rounds = 0
        # local_rounds = int(np.max(time_workers) / time_workers[flag])
        for aa in range(10000):
            for batch_idx, (data, target) in enumerate(loader_workers[m]):
                optimizer_worker[m].zero_grad()
                output = network_worker(data)
                # output = network_worker(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer_worker[m].step()
                if aa == 0 and batch_idx <= K-2:
                    continue
                else:
                    time_flag += np.random.exponential(miu_e, [1, 1])[0]

                # local_rounds = int(np.max(time_workers) / time_workers[m] - 1)
                if time_flag > time_max:
                    grad_workers[m]=optimizer_worker[m].param_groups[0]['params']
                    break
            if time_flag >= time_max:
                break
    for m in noselection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum) 
        time_flag = time_workers1[K-1][m]
        for batch_idx, (data, target) in enumerate(loader_workers[m]):
            optimizer_worker[m].zero_grad()
            output = network_worker(data)
            # output = network_worker(data)
            loss = criterion(output, target)
            loss.backward()
            if time_workers1[batch_idx][m] > time_max:
                break
            optimizer_worker[m].step()
            if time_workers1[batch_idx+1][m] > time_max:
                grad_workers[m]=optimizer_worker[m].param_groups[0]['params']
                selection.append(m)
                break
    temp = [0]*10
    for l in range(10):
        count = 0
        for m in selection:
            temp[l] += grad_workers[m][l]
            count += 1
        temp[l] /= count
    
    for batch_idx, (data, target) in enumerate(test_loader):
        optimizer.zero_grad()
        for l in range(10):
            optimizer.param_groups[0]['params'][l].grad = -temp[l] + optimizer.param_groups[0]['params'][l]
        optimizer.step()
        output = network(data)
        # output = network(data)
        loss = criterion(output, target)
        break
    # optimizer.param_groups[0]['lr'] *= 0.99
    print(loss.item(), "STSyn")
    print("E=", S, "K=", K)
    train_losses_ADA_Wireless_modified2.append(loss.item())
    t_ADA_Wireless_modified2[i] += time_max
    if t_ADA_Wireless_modified2[i] < t_ADA_Wireless_modified2[i-1]:
        print("error!!!")
        break
    t_ADA_Wireless_modified2[i+1] = t_ADA_Wireless_modified2[i]
    comm_ADA_Wireless_modified2[i] += M + len(selection)
    comm_ADA_Wireless_modified2[i+1] = comm_ADA_Wireless_modified2[i]    
    S_old = S
    # S = min(max(int(loss_orig / loss.item()*S0),S_old), M)
    K_old = K
    #K = min(max(int((loss_orig / loss.item())*K0),K_old), M)
    if test_losses_ADA_Wireless_modified2[-1] >= acc:
        break

#%% ADA_Wireless_modified3
# iters = iters * 3
# iters3 = 800
iters3 = 6000
S0 = 10
S = S0
K0 = 10
K = K0
t_ADA_Wireless_modified3 = [0]*(iters3+1)
comm_ADA_Wireless_modified3 = [0]*(iters3+1)
train_losses_ADA_Wireless_modified3 = []
test_losses_ADA_Wireless_modified3 = []
network = copy.deepcopy(network1)
# optimizer = optim.Adam(network.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer = optim.SGD(network.parameters(), lr=1, momentum=momentum)
network.train()
optimizer_worker = []
for m in range(M):
    optimizer_worker.append(0)
for i in range(iters3):
    if i == 0:
        for batch_idx, (data, target) in enumerate(test_loader):
            output = network(data)
            # output = network(data)
            loss = criterion(output, target)
            break
        loss_orig = loss.item()
    if i % 10 == 0:
        network.eval()
        test_loss_ADA_Wireless_modified3 = 0
        correct_ADA_Wireless_modified3 = 0
        with torch.no_grad():
          for data, target in test_loader:
            output = network(data)
            # output = network(data)
            test_loss_ADA_Wireless_modified3 += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct_ADA_Wireless_modified3 += pred.eq(target.data.view_as(pred)).sum()
        test_loss_ADA_Wireless_modified3 /= len(test_loader.dataset)
        test_losses_ADA_Wireless_modified3.append(100. * correct_ADA_Wireless_modified3 / len(test_loader.dataset))
    time_workers1 = {}
    for k in range(K):
        time_workers1[k]=0
    for k in range(K):
        time_workers1[k] += np.random.exponential(miu_e, M)
        # time_workers1[k] += np.random.pareto(miu, M)
        time_workers1[k+1] = copy.deepcopy(time_workers1[k])
    time_workers = np.sort(time_workers1[K-1])
    selection = list(np.argsort(time_workers1[K-1]))[0:S]
    time_workers = time_workers[0:S]
    time_max = np.max(time_workers)
    # selection = list(range(M))
    noselection = list(set(list(range(M))).difference(set(selection)))
    grad_workers= {}
    for m in selection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum) 
        time_flag = time_workers1[K-1][m]
        rounds = 0
        # local_rounds = int(np.max(time_workers) / time_workers[flag])
        for aa in range(10000):
            for batch_idx, (data, target) in enumerate(loader_workers[m]):
                optimizer_worker[m].zero_grad()
                output = network_worker(data)
                # output = network_worker(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer_worker[m].step()
                if aa == 0 and batch_idx <= K-2:
                    continue
                else:
                    time_flag += np.random.exponential(miu_e, [1, 1])[0]

                # local_rounds = int(np.max(time_workers) / time_workers[m] - 1)
                if time_flag > time_max:
                    grad_workers[m]=optimizer_worker[m].param_groups[0]['params']
                    break
            if time_flag >= time_max:
                break
    for m in noselection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum) 
        time_flag = time_workers1[K-1][m]
        for batch_idx, (data, target) in enumerate(loader_workers[m]):
            optimizer_worker[m].zero_grad()
            output = network_worker(data)
            # output = network_worker(data)
            loss = criterion(output, target)
            loss.backward()
            if time_workers1[batch_idx][m] > time_max:
                break
            optimizer_worker[m].step()
            if time_workers1[batch_idx+1][m] > time_max:
                grad_workers[m]=optimizer_worker[m].param_groups[0]['params']
                selection.append(m)
                break
    temp = [0]*10
    for l in range(10):
        count = 0
        for m in selection:
            temp[l] += grad_workers[m][l]
            count += 1
        temp[l] /= count
    
    for batch_idx, (data, target) in enumerate(test_loader):
        optimizer.zero_grad()
        for l in range(10):
            optimizer.param_groups[0]['params'][l].grad = -temp[l] + optimizer.param_groups[0]['params'][l]
        optimizer.step()
        output = network(data)
        # output = network(data)
        loss = criterion(output, target)
        break
    # optimizer.param_groups[0]['lr'] *= 0.99
    print(loss.item(), "STSyn")
    print("E=", S, "K=", K)
    train_losses_ADA_Wireless_modified3.append(loss.item())
    t_ADA_Wireless_modified3[i] += time_max
    if t_ADA_Wireless_modified3[i] < t_ADA_Wireless_modified3[i-1]:
        print("error!!!")
        break
    t_ADA_Wireless_modified3[i+1] = t_ADA_Wireless_modified3[i]
    comm_ADA_Wireless_modified3[i] += M + len(selection)
    comm_ADA_Wireless_modified3[i+1] = comm_ADA_Wireless_modified3[i]    
    S_old = S
    # S = min(max(int(loss_orig / loss.item()*S0),S_old), M)
    K_old = K
    #K = min(max(int((loss_orig / loss.item())*K0),K_old), M)
    if test_losses_ADA_Wireless_modified3[-1] >= acc:
        break

#%% ADA_Wireless_modified4
# iters = iters * 3
# iters3 = 800
iters3 = 6000
S0 = 20
S = S0
K0 = 10
K = K0
t_ADA_Wireless_modified4 = [0]*(iters3+1)
comm_ADA_Wireless_modified4 = [0]*(iters3+1)
train_losses_ADA_Wireless_modified4 = []
test_losses_ADA_Wireless_modified4 = []
network = copy.deepcopy(network1)
# optimizer = optim.Adam(network.parameters(), lr=0.005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer = optim.SGD(network.parameters(), lr=1, momentum=momentum)
network.train()
optimizer_worker = []
for m in range(M):
    optimizer_worker.append(0)
for i in range(iters3):
    if i == 0:
        for batch_idx, (data, target) in enumerate(test_loader):
            output = network(data)
            # output = network(data)
            loss = criterion(output, target)
            break
        loss_orig = loss.item()
    if i % 10 == 0:
        network.eval()
        test_loss_ADA_Wireless_modified4 = 0
        correct_ADA_Wireless_modified4 = 0
        with torch.no_grad():
          for data, target in test_loader:
            output = network(data)
            # output = network(data)
            test_loss_ADA_Wireless_modified4 += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct_ADA_Wireless_modified4 += pred.eq(target.data.view_as(pred)).sum()
        test_loss_ADA_Wireless_modified4 /= len(test_loader.dataset)
        test_losses_ADA_Wireless_modified4.append(100. * correct_ADA_Wireless_modified4 / len(test_loader.dataset))
    time_workers1 = {}
    for k in range(K):
        time_workers1[k]=0
    for k in range(K):
        time_workers1[k] += np.random.exponential(miu_e, M)
        # time_workers1[k] += np.random.pareto(miu, M)
        time_workers1[k+1] = copy.deepcopy(time_workers1[k])
    time_workers = np.sort(time_workers1[K-1])
    selection = list(np.argsort(time_workers1[K-1]))[0:S]
    time_workers = time_workers[0:S]
    time_max = np.max(time_workers)
    # selection = list(range(M))
    noselection = list(set(list(range(M))).difference(set(selection)))
    grad_workers= {}
    for m in selection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum) 
        time_flag = time_workers1[K-1][m]
        rounds = 0
        # local_rounds = int(np.max(time_workers) / time_workers[flag])
        for aa in range(10000):
            for batch_idx, (data, target) in enumerate(loader_workers[m]):
                optimizer_worker[m].zero_grad()
                output = network_worker(data)
                # output = network_worker(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer_worker[m].step()
                if aa == 0 and batch_idx <= K-2:
                    continue
                else:
                    time_flag += np.random.exponential(miu_e, [1, 1])[0]

                # local_rounds = int(np.max(time_workers) / time_workers[m] - 1)
                if time_flag > time_max:
                    grad_workers[m]=optimizer_worker[m].param_groups[0]['params']
                    break
            if time_flag >= time_max:
                break
    for m in noselection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum) 
        time_flag = time_workers1[K-1][m]
        for batch_idx, (data, target) in enumerate(loader_workers[m]):
            optimizer_worker[m].zero_grad()
            output = network_worker(data)
            # output = network_worker(data)
            loss = criterion(output, target)
            loss.backward()
            if time_workers1[batch_idx][m] > time_max:
                break
            optimizer_worker[m].step()
            if time_workers1[batch_idx+1][m] > time_max:
                grad_workers[m]=optimizer_worker[m].param_groups[0]['params']
                selection.append(m)
                break
    temp = [0]*10
    for l in range(10):
        count = 0
        for m in selection:
            temp[l] += grad_workers[m][l]
            count += 1
        temp[l] /= count
    
    for batch_idx, (data, target) in enumerate(test_loader):
        optimizer.zero_grad()
        for l in range(10):
            optimizer.param_groups[0]['params'][l].grad = -temp[l] + optimizer.param_groups[0]['params'][l]
        optimizer.step()
        output = network(data)
        # output = network(data)
        loss = criterion(output, target)
        break
    # optimizer.param_groups[0]['lr'] *= 0.99
    print(loss.item(), "STSyn")
    print("E=", S, "K=", K)
    train_losses_ADA_Wireless_modified4.append(loss.item())
    t_ADA_Wireless_modified4[i] += time_max
    if t_ADA_Wireless_modified4[i] < t_ADA_Wireless_modified4[i-1]:
        print("error!!!")
        break
    t_ADA_Wireless_modified4[i+1] = t_ADA_Wireless_modified4[i]
    comm_ADA_Wireless_modified4[i] += M + len(selection)
    comm_ADA_Wireless_modified4[i+1] = comm_ADA_Wireless_modified4[i]    
    S_old = S
    # S = min(max(int(loss_orig / loss.item()*S0),S_old), M)
    K_old = K
    #K = min(max(int((loss_orig / loss.item())*K0),K_old), M)
    if test_losses_ADA_Wireless_modified4[-1] >= acc:
        break
#%%
figure()
plot(t_ADA_Wireless_modified1[0:len(test_losses_ADA_Wireless_modified1)*10:10], test_losses_ADA_Wireless_modified1,linestyle='-')
plot(t_ADA_Wireless_modified2[0:len(test_losses_ADA_Wireless_modified2)*10:10], test_losses_ADA_Wireless_modified2,linestyle='--')
plot(t_ADA_Wireless_modified3[0:len(test_losses_ADA_Wireless_modified3)*10:10], test_losses_ADA_Wireless_modified3,linestyle='-.')
plot(t_ADA_Wireless_modified4[0:len(test_losses_ADA_Wireless_modified4)*10:10], test_losses_ADA_Wireless_modified4,linestyle=':')
# plot(t_ADA_Wireless_modified5[0:iters5:10], test_losses_ADA_Wireless_modified5)
# plot(test_losses_SCAFFOLD)
# legend(['FED','FED_ACSA','ACSA'])
legend(['K=1, U=10','K=5, U=10','K=15, U=10','K=20, U=10'],loc="lower right")
xlabel('wall-clock time')
ylabel('test accuracy')
# legend(['DSGD','FED','FED_ACSA','ACSA','SCAFFOLD'])
#plot(train_losses_AMS)
# xlim([0, 0.5])
# ylim([46, 57])
grid('on')
savefig('time-accuracy_K_cifar_iid.pdf')

figure()
plt.plot(comm_ADA_Wireless_modified1[0:len(test_losses_ADA_Wireless_modified1)*10:10], test_losses_ADA_Wireless_modified1,linestyle='-')
plt.plot(comm_ADA_Wireless_modified2[0:len(test_losses_ADA_Wireless_modified2)*10:10], test_losses_ADA_Wireless_modified2,linestyle='--')
plt.plot(comm_ADA_Wireless_modified3[0:len(test_losses_ADA_Wireless_modified3)*10:10], test_losses_ADA_Wireless_modified3,linestyle='-.')
plot(comm_ADA_Wireless_modified4[0:len(test_losses_ADA_Wireless_modified4)*10:10], test_losses_ADA_Wireless_modified4,linestyle=':')
# plot(comm_ADA_Wireless_modified5[0:iters5:10], test_losses_ADA_Wireless_modified5)
# plot(test_losses_SCAFFOLD)
# legend(['FED','FED_ACSA','ACSA'])
legend(['K=1, U=10','K=5, U=10','K=15, U=10','K=20, U=10'],loc="lower right")
xlabel('communication')
ylabel('test accuracy')
# legend(['DSGD','FED','FED_ACSA','ACSA','SCAFFOLD'])
#plot(train_losses_AMS)
# xlim([0, 30000])
# ylim([30, 57])
plt.ticklabel_format(style='sci', axis='x',scilimits=(0,0))
grid('on')
savefig('comm-accuracy_K_cifar_iid.pdf')
