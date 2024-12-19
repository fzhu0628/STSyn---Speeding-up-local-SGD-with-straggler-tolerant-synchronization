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
miu_e = 1e4
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
index = np.array(dataset_global.targets).argsort()
# index = torch.randperm(length)
# index = np.array(torch.randperm(length))

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network1 = Net().to(device)


acc = 70

#%% ADA_Wireless_modified1
# iters = iters * 3
# iters3 = 800
iters3 = 6000
lr = 0.1
S0 = 5
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
            data, target = data.to(device), target.to(device)
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
            data, target = data.to(device), target.to(device)
            output = network(data)
            # output = network(data)
            test_loss_ADA_Wireless_modified1 += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct_ADA_Wireless_modified1 += pred.eq(target.data.view_as(pred)).sum()
        test_loss_ADA_Wireless_modified1 /= len(test_loader.dataset)
        test_losses_ADA_Wireless_modified1.append(100. * correct_ADA_Wireless_modified1 / len(test_loader.dataset))
    time_workers1 = {}
    mean_workers = np.random.uniform(miu_e, miu_e*2, M)
    
    for k in range(K):
        time_workers1[k]=0
    batch_time = np.zeros(M)
    for m in range(M):
        batch_time[m] = np.random.pareto(mean_workers[m], 1)
    for k in range(K):
        time_workers1[k] += batch_time
        # time_workers1[k] += np.random.pareto(miu, M)
        time_workers1[k+1] = copy.deepcopy(time_workers1[k])
    time_workers = np.sort(time_workers1[K-1])
    selection = list(np.argsort(time_workers1[K-1]))[0:S]
    time_workers = time_workers[0:S]
    time_max = np.max(time_workers)
    # selection = list(range(M))
    noselection = list(set(list(range(M))).difference(set(selection)))
    grad_workers= {}
    NLU = np.zeros(M)
    for m in selection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=lr, momentum=momentum) 
        time_flag = time_workers1[K-1][m]
        rounds = 0
        # local_rounds = int(np.max(time_workers) / time_workers[flag])
        for aa in range(10000):
            for batch_idx, (data, target) in enumerate(loader_workers[m]):
                data, target = data.to(device), target.to(device)
                if rounds >= K*1.5:
                    grad_workers[m]=optimizer_worker[m].param_groups[0]['params']
                    break
                rounds += 1
                optimizer_worker[m].zero_grad()
                output = network_worker(data)
                # output = network_worker(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer_worker[m].step()
                if aa == 0 and batch_idx <= K-2:
                    continue
                else:
                    time_flag += batch_time[m]
                    # time_flag += np.random.pareto(mean_workers[m], 1)

                # local_rounds = int(np.max(time_workers) / time_workers[m] - 1)
                if time_flag > time_max:
                    grad_workers[m]=optimizer_worker[m].param_groups[0]['params']
                    break
            if rounds >= K*1.5:
                break
            if time_flag >= time_max:
                break
        NLU[m] = rounds
    for m in noselection:
        rounds = 0
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=lr, momentum=momentum) 
        time_flag = time_workers1[K-1][m]
        for batch_idx, (data, target) in enumerate(loader_workers[m]):
            data, target = data.to(device), target.to(device)
            rounds += 1
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
                NLU[m] = rounds
                break
    
    selection.remove(argmax(NLU))
    
    
    temp = [0]*10
    LUSUM = sum(NLU)
    for l in range(10):
        for m in selection:
            # temp[l] += (grad_workers[m][l] - optimizer.param_groups[0]['params'][l])/NLU[m]*LUSUM/len(selection)/len(selection)
            temp[l] += (grad_workers[m][l] - optimizer.param_groups[0]['params'][l])/len(selection)
        # temp[l] /= count^2
    
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        for l in range(10):
            optimizer.param_groups[0]['params'][l].grad = -temp[l]
        optimizer.step()
        output = network(data)
        # output = network(data)
        loss = criterion(output, target)
        break
    
    # temp = [0]*10
    # NLUSUM = sum(NLU)
    # for l in range(10):
    #     count = 0
    #     for m in selection:
    #         temp[l] += grad_workers[m][l]
    #         count += 1
    #     temp[l] /= count
    
    # for batch_idx, (data, target) in enumerate(test_loader):
    #     optimizer.zero_grad()
    #     for l in range(10):
    #         optimizer.param_groups[0]['params'][l].grad = -temp[l] + optimizer.param_groups[0]['params'][l]
    #     optimizer.step()
    #     output = network(data)
    #     # output = network(data)
    #     loss = criterion(output, target)
    #     break
    # optimizer.param_groups[0]['lr'] *= 0.99
    print(loss.item(), "STSyn")
    print("E=", S, "K=", K)
    train_losses_ADA_Wireless_modified1.append(loss.item())
    t_ADA_Wireless_modified1[i] += time_max
    if t_ADA_Wireless_modified1[i] < t_ADA_Wireless_modified1[i-1]:
        print("error!!!")
        break
    t_ADA_Wireless_modified1[i+1] = t_ADA_Wireless_modified1[i]
    comm_ADA_Wireless_modified1[i] +=  len(selection) + M
    comm_ADA_Wireless_modified1[i+1] = comm_ADA_Wireless_modified1[i]    
    S_old = S
    # S = min(max(int(loss_orig / loss.item()*S0),S_old), M)
    K_old = K
    #K = min(max(int((loss_orig / loss.item())*K0),K_old), M)
    if test_losses_ADA_Wireless_modified1[-1] >= acc:
        break
 
    
#%% DSGD_Wireless2
S = 5
tau0 = 10
taul = tau0
# iters2 = 1200
iters2 = 6000
t_DSGD_Wireless2 = [0]*(iters2+1)
comm_DSGD_Wireless2 = [0]*(iters2+1)
train_losses_DSGD_Wireless2 = []
test_losses_DSGD_Wireless2 = []
network = copy.deepcopy(network1)
# optimizer = optim.Adam(network.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.1, amsgrad=True)
optimizer = optim.SGD(network.parameters(), lr=1, momentum=momentum)
network.train()
for i in range(iters2):
    if i % 10 == 0:
        network.eval()
        test_loss_DSGD_Wireless2 = 0
        correct_DSGD_Wireless2 = 0
        with torch.no_grad():
          for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss_DSGD_Wireless2 += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct_DSGD_Wireless2 += pred.eq(target.data.view_as(pred)).sum()
        test_loss_DSGD_Wireless2 /= len(test_loader.dataset)
        test_losses_DSGD_Wireless2.append(100. * correct_DSGD_Wireless2 / len(test_loader.dataset))
    grad_workers= {}
    selection = list(np.random.choice(range(M), S, replace=False))
    for m in selection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum)  
        k = 0
        for aa in range(100):
            for batch_idx, (data, target) in enumerate(loader_workers[m]):
                data, target = data.to(device), target.to(device)
                k += 1
                optimizer_worker.zero_grad()
                output = network_worker(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer_worker.step()
                # time_workers[m] += np.random.pareto(miu, [1, 1])
                if i == 0:
                    if k == tau0:
                        grad_workers[m]=optimizer_worker.param_groups[0]['params']
                        break
                else:
                    if k == taul:
                        grad_workers[m]=optimizer_worker.param_groups[0]['params']
                        break
            if (i == 0 and k == tau0) or (i > 0 and k == taul):
                break
    for m in selection:
        for l in range(10):
            grad_workers[m][l] = grad_workers[m][l] - optimizer.param_groups[0]['params'][l]
    temp = [0]*10
    for l in range(10):
        count = 0
        for m in selection:
            temp[l] += grad_workers[m][l] 
            count += 1
        temp[l] = temp[l] / count

    optimizer.zero_grad()
    norm_use = 0
    for l in range(10):
        optimizer.param_groups[0]['params'][l].grad = -temp[l] 
        norm_use += torch.norm(temp[l], p=2).item()**2
    norm_use = np.sqrt(norm_use)
    
    
    optimizer.step()
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = network(data)
        loss = criterion(output, target)
        print(loss.item(), 'FLANP', 'S=',S)
        train_losses_DSGD_Wireless2.append(loss.item())
        # taul = int(ceil(sqrt(train_losses_DSGD_Wireless2[-1] / orig_loss) * tau0))
        break
    # optimizer.param_groups[0]['lr'] *= 0.99
    #if loss.item() < EPS:
        #break
    mean_workers = np.random.uniform(miu_e, miu_e*2, M)
    time_workers = [0]*M
    for m in range(M):
        time_workers[m] = np.random.pareto(mean_workers[m], 1) * taul
    # for m in range(M):
    #     for t in range(taul):
    #         time_workers[m] += np.random.pareto(mean_workers[m], 1)
    t_DSGD_Wireless2[i] += np.sort(time_workers)[S-1]
    t_DSGD_Wireless2[i+1] = t_DSGD_Wireless2[i]*1
    comm_DSGD_Wireless2[i] += S + M
    comm_DSGD_Wireless2[i+1] = comm_DSGD_Wireless2[i]
    if norm_use <= 0.2:
        S = min(S*2, M)
    if test_losses_DSGD_Wireless2[-1] >= acc:
        break

#%% FedNova
# iters = iters * 3
# iters3 = 800
iters3 = 6000
S0 = M
S = S0
selection = range(M)
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
# NLU = np.maximum(floor(np.random.pareto(K0, M)),1)
NLU = np.maximum(floor(np.random.randint(low=8, high=13, size=(M))),1)
for i in range(iters3):
    if i == 0:
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
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
            data, target = data.to(device), target.to(device)
            output = network(data)
            # output = network(data)
            test_loss_ADA_Wireless_modified2 += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct_ADA_Wireless_modified2 += pred.eq(target.data.view_as(pred)).sum()
        test_loss_ADA_Wireless_modified2 /= len(test_loader.dataset)
        test_losses_ADA_Wireless_modified2.append(100. * correct_ADA_Wireless_modified2 / len(test_loader.dataset))
    time_workers = np.zeros(M)
    grad_workers= {}
    LU = np.zeros(M)
    mean_workers = np.random.uniform(miu_e, miu_e*2, M)
    for m in selection:
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker[m] = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum) 
        rounds = 0
        batch_time = np.random.pareto(mean_workers[m], 1)
        # local_rounds = int(np.max(time_workers) / time_workers[flag])
        for aa in range(10000):
            for batch_idx, (data, target) in enumerate(loader_workers[m]):
                data, target = data.to(device), target.to(device)
                rounds += 1
                optimizer_worker[m].zero_grad()
                output = network_worker(data)
                # output = network_worker(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer_worker[m].step()
                if rounds <= NLU[m]:
                    time_workers[m] += batch_time
                    # time_workers[m] += np.random.pareto(mean_workers[m], 1)
                else:
                    break
                # local_rounds = int(np.max(time_workers) / time_workers[m] - 1)
            if rounds >= NLU[m]:
                grad_workers[m]=optimizer_worker[m].param_groups[0]['params']
                break
    temp = [0]*10
    LUSUM = sum(NLU)
    for l in range(10):
        count = 0
        for m in selection:
            temp[l] += (grad_workers[m][l] - optimizer.param_groups[0]['params'][l])/M*LUSUM/NLU[m]/M
            count += 1
        # temp[l] /= count^2
    
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        for l in range(10):
            optimizer.param_groups[0]['params'][l].grad = -temp[l]
        optimizer.step()
        output = network(data)
        # output = network(data)
        loss = criterion(output, target)
        break
    # optimizer.param_groups[0]['lr'] *= 0.99
    print(loss.item(), "FedNova")
    print("E=", S, "K=", K)
    train_losses_ADA_Wireless_modified2.append(loss.item())
    t_ADA_Wireless_modified2[i] += np.max(time_workers)
    if t_ADA_Wireless_modified2[i] < t_ADA_Wireless_modified2[i-1]:
        print("error!!!")
        break
    t_ADA_Wireless_modified2[i+1] = t_ADA_Wireless_modified2[i]
    comm_ADA_Wireless_modified2[i] += len(selection) + M
    comm_ADA_Wireless_modified2[i+1] = comm_ADA_Wireless_modified2[i]    
    S_old = S
    # S = min(max(int(loss_orig / loss.item()*S0),S_old), M)
    K_old = K
    #K = min(max(int((loss_orig / loss.item())*K0),K_old), M)
    if test_losses_ADA_Wireless_modified2[-1] >= acc:
        break

#%% DSGD_Wireless1
S = M
tau0 = 10
taul = tau0
# iters2 = 1200
iters2 = 6000
t_DSGD_Wireless1 = [0]*(iters2+1)
comm_DSGD_Wireless1 = [0]*(iters2+1)
time_workers = [0]*M
train_losses_DSGD_Wireless1 = []
test_losses_DSGD_Wireless1 = []
network = copy.deepcopy(network1)
# optimizer = optim.Adam(network.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.1, amsgrad=True)
optimizer = optim.SGD(network.parameters(), lr=1, momentum=momentum)
network.train()
for i in range(iters2):
    if i % 10 == 0:
        network.eval()
        test_loss_DSGD_Wireless1 = 0
        correct_DSGD_Wireless1 = 0
        with torch.no_grad():
          for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss_DSGD_Wireless1 += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct_DSGD_Wireless1 += pred.eq(target.data.view_as(pred)).sum()
        test_loss_DSGD_Wireless1 /= len(test_loader.dataset)
        test_losses_DSGD_Wireless1.append(100. * correct_DSGD_Wireless1 / len(test_loader.dataset))
    grad_workers= {}
    selection = list(np.random.choice(range(M), S, replace=False))
    mean_workers = np.random.uniform(miu_e, miu_e*2, M)
    for m in selection:
        time_workers[m] = 0
        batch_time = np.random.pareto(mean_workers[m], 1)
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum)  
        k = 0
        for aa in range(100):
            for batch_idx, (data, target) in enumerate(loader_workers[m]):
                data, target = data.to(device), target.to(device)
                k += 1
                optimizer_worker.zero_grad()
                output = network_worker(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer_worker.step()
                time_workers[m] += batch_time
                # time_workers[m] += np.random.pareto(mean_workers[m], 1)
                # time_workers[m] += np.random.pareto(miu, [1, 1])
                if i == 0:
                    if k == tau0:
                        grad_workers[m]=optimizer_worker.param_groups[0]['params']
                        break
                else:
                    if k == taul:
                        grad_workers[m]=optimizer_worker.param_groups[0]['params']
                        break
            if (i == 0 and k == tau0) or (i > 0 and k == taul):
                break
    for m in selection:
        for l in range(10):
            grad_workers[m][l] = grad_workers[m][l] - optimizer.param_groups[0]['params'][l]
    temp = [0]*10
    for l in range(10):
        count = 0
        for m in selection:
            temp[l] += grad_workers[m][l] 
            count += 1
        temp[l] = temp[l] / count

    optimizer.zero_grad()
    for l in range(10):
        optimizer.param_groups[0]['params'][l].grad = -temp[l] 
    optimizer.step()
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = network(data)
        loss = criterion(output, target)
        print(loss.item(), 'PASGD', 'K=',taul)
        train_losses_DSGD_Wireless1.append(loss.item())
        # taul = int(ceil(sqrt(train_losses_DSGD_Wireless1[-1] / orig_loss) * tau0))
        break
    # optimizer.param_groups[0]['lr'] *= 0.99
    #if loss.item() < EPS:
        #break
    t_DSGD_Wireless1[i] += max(time_workers)
    t_DSGD_Wireless1[i+1] = t_DSGD_Wireless1[i]*1
    comm_DSGD_Wireless1[i] += M * 2
    comm_DSGD_Wireless1[i+1] = comm_DSGD_Wireless1[i]
    if test_losses_DSGD_Wireless1[-1] >= acc:
        break

#%% DSGD_Wireless
S = M
tau0 = 10
taul = tau0
taul_old = tau0
# iters2 = 1200
ind = 0 
interval = 0.05
iters2 = 6000
t_DSGD_Wireless = [0]*(iters2+1)
comm_DSGD_Wireless = [0]*(iters2+1)
time_workers = [0]*M
train_losses_DSGD_Wireless = []
test_losses_DSGD_Wireless = []
network = copy.deepcopy(network1)
# optimizer = optim.Adam(network.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-07, weight_decay=0.1, amsgrad=True)
optimizer = optim.SGD(network.parameters(), lr=1, momentum=momentum)
network.train()
for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = network(data)
        loss = criterion(output, target)
        print(loss.item(), 'AdaComm')
        orig_loss = loss.item()
        break
for i in range(iters2):
    if i % 10 == 0:
        network.eval()
        test_loss_DSGD_Wireless = 0
        correct_DSGD_Wireless = 0
        with torch.no_grad():
          for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss_DSGD_Wireless += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct_DSGD_Wireless += pred.eq(target.data.view_as(pred)).sum()
        test_loss_DSGD_Wireless /= len(test_loader.dataset)
        test_losses_DSGD_Wireless.append(100. * correct_DSGD_Wireless / len(test_loader.dataset))
    grad_workers= {}
    selection = list(np.random.choice(range(M), S, replace=False))
    mean_workers = np.random.uniform(miu_e, miu_e*2, M)
    for m in selection:
        time_workers[m] = 0
        batch_time = np.random.pareto(mean_workers[m], 1)
        network_worker = copy.deepcopy(network)
        network_worker.train()
        optimizer_worker = optim.SGD(network_worker.parameters(), lr=0.1, momentum=momentum) 
        k = 0
        for aa in range(100):
            for batch_idx, (data, target) in enumerate(loader_workers[m]):
                data, target = data.to(device), target.to(device)
                k += 1
                optimizer_worker.zero_grad()
                output = network_worker(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer_worker.step()
                time_workers[m] += batch_time
                # time_workers[m] += np.random.pareto(mean_workers[m], 1)
                # time_workers[m] += np.random.pareto(miu, [1, 1])
                if i == 0:
                    if k == tau0:
                        grad_workers[m]=optimizer_worker.param_groups[0]['params']
                        break
                else:
                    if k == taul:
                        grad_workers[m]=optimizer_worker.param_groups[0]['params']
                        break
            if (i == 0 and k == tau0) or (i > 0 and k == taul):
                break
    for m in selection:
        for l in range(10):
            grad_workers[m][l] = grad_workers[m][l] - optimizer.param_groups[0]['params'][l]
    temp = [0]*10
    for l in range(10):
        count = 0
        for m in selection:
            temp[l] += grad_workers[m][l] 
            count += 1
        temp[l] = temp[l] / count

    optimizer.zero_grad()
    for l in range(10):
        optimizer.param_groups[0]['params'][l].grad = -temp[l] 
    optimizer.step()
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = network(data)
        loss = criterion(output, target)
        print(loss.item(), 'AdaComm', 'K=',taul)
        train_losses_DSGD_Wireless.append(loss.item())
        break
    
    if t_DSGD_Wireless[i] > ind * interval :
        taul = min(int(ceil(sqrt(train_losses_DSGD_Wireless[-1] / orig_loss) * tau0)), taul_old)
        ind += 1
    taul_old = taul
    # optimizer.param_groups[0]['lr'] *= 0.99
    #if loss.item() < EPS:
        #break
    t_DSGD_Wireless[i] += max(time_workers)
    t_DSGD_Wireless[i+1] = t_DSGD_Wireless[i]*1
    comm_DSGD_Wireless[i] += M * 2
    comm_DSGD_Wireless[i+1] = comm_DSGD_Wireless[i]
    if test_losses_DSGD_Wireless[-1] >= acc:
        break


#%%
for a in range(len(test_losses_DSGD_Wireless)):
    test_losses_DSGD_Wireless[a] = np.array(test_losses_DSGD_Wireless[a].cpu())
    
for a in range(len(test_losses_DSGD_Wireless1)):
    test_losses_DSGD_Wireless1[a] = np.array(test_losses_DSGD_Wireless1[a].cpu())

for a in range(len(test_losses_DSGD_Wireless2)):
    test_losses_DSGD_Wireless2[a] = np.array(test_losses_DSGD_Wireless2[a].cpu())    

for a in range(len(test_losses_ADA_Wireless_modified2)):
    test_losses_ADA_Wireless_modified2[a] = np.array(test_losses_ADA_Wireless_modified2[a].cpu())

for a in range(len(test_losses_ADA_Wireless_modified1)):
    test_losses_ADA_Wireless_modified1[a] = np.array(test_losses_ADA_Wireless_modified1[a].cpu())

#%%
figure()
plot(t_DSGD_Wireless[0:len(test_losses_DSGD_Wireless)*10:10], test_losses_DSGD_Wireless,linestyle='--')
plot(t_DSGD_Wireless1[0:len(test_losses_DSGD_Wireless1)*10:10], test_losses_DSGD_Wireless1,linestyle='-.')
plot(t_DSGD_Wireless2[0:len(test_losses_DSGD_Wireless2)*10:10], test_losses_DSGD_Wireless2,linestyle='-',color='black')
# plot(t_ADA_Wireless[0:iters3:10], test_losses_ADA_Wireless)
#plot(t_ADA_Wireless_modified[0:iters4:10], test_losses_ADA_Wireless_modified)
plot(t_ADA_Wireless_modified2[0:len(test_losses_ADA_Wireless_modified2)*10:10], test_losses_ADA_Wireless_modified2,linestyle='-',color='green')

plot(t_ADA_Wireless_modified1[0:len(test_losses_ADA_Wireless_modified1)*10:10], test_losses_ADA_Wireless_modified1,linestyle='-',color='red')
# plot(test_losses_SCAFFOLD)
# legend(['FED','FED_ACSA','ACSA'])
legend(['AdaComm','PASGD','FLANP','FedNova','STSyn'])
xlabel('wall-clock time')
ylabel('test accuracy')
# legend(['DSGD','FED','FED_ACSA','ACSA','SCAFFOLD'])
#plot(train_losses_AMS)
# xlim([0, 7])
# ylim([35, 58])
grid('on')
savefig('time-accuracy_state_of_the_art_unfixedtime_ada_noniid_cifar_averaging_contention.pdf')

figure()
plot(comm_DSGD_Wireless[0:len(test_losses_DSGD_Wireless)*10:10], test_losses_DSGD_Wireless,linestyle='--')
plot(comm_DSGD_Wireless1[0:len(test_losses_DSGD_Wireless1)*10:10], test_losses_DSGD_Wireless1,linestyle='-.')
plot(comm_DSGD_Wireless2[0:len(test_losses_DSGD_Wireless2)*10:10], test_losses_DSGD_Wireless2,linestyle='-',color='black')
# plot(comm_DSGD_Wireless1[0:iters4:10], test_losses_DSGD_Wireless1)
# plot(comm_ADA_Wireless[0:iters3:10], test_losses_ADA_Wireless)
#plot(comm_ADA_Wireless_modified[0:iters4:10], test_losses_ADA_Wireless_modified)
plot(comm_ADA_Wireless_modified2[0:len(test_losses_ADA_Wireless_modified2)*10:10], test_losses_ADA_Wireless_modified2,linestyle='-',color='green')
plot(comm_ADA_Wireless_modified1[0:len(test_losses_ADA_Wireless_modified1)*10:10], test_losses_ADA_Wireless_modified1,linestyle='-',color='red')

# plot(test_losses_SCAFFOLD)
# legend(['FED','FED_ACSA','ACSA'])
legend(['AdaComm','PASGD','FLANP','FedNova','STSyn'])
xlabel('communication cost')
ylabel('test accuracy')
# legend(['DSGD','FED','FED_ACSA','ACSA','SCAFFOLD'])
#plot(train_losses_AMS )

# xlim([0, 270000])
# ylim([30, 60])
grid('on')
savefig('comm-accuracy_state_of_the_art_unfixedtime_ada_noniid_cifar_averaging_contention.pdf')

# figure()
# plot(t_DSGD, comm_DSGD)
# plot(t_DSGD_Wireless, comm_DSGD_Wireless)
# plot(t_ADA_Wireless, comm_ADA_Wireless)
# # plot(t_FED, train_losses_FED)
# # plot(t_FED_ACSA, train_losses_FED_ACSA)
# # plot(t_ACSA, train_losses_ACSA)
# # # plot(train_losses_SCAFFOLD)
# # #plot(train_losses_AMS)
# legend(['AdaSync','AdaComm','AdaLU'])
# # legend(['DSGD','FED','FED_ACSA','ACSA'])
# xlabel('wall-clock time')
# ylabel('communication')
# # # legend(['DSGD','FED','FED_ACSA','ACSA','SCAFFOLD'])
# grid('on')

# figure()
# plot(comm_DSGD, train_losses_DSGD)
# plot(comm_DSGD_Wireless, train_losses_DSGD_Wireless)
# plot(comm_ADA_Wireless, train_losses_ADA_Wireless)
# # plot(t_FED, train_losses_FED)
# # plot(t_FED_ACSA, train_losses_FED_ACSA)
# # plot(t_ACSA, train_losses_ACSA)
# # # plot(train_losses_SCAFFOLD)
# # #plot(train_losses_AMS)
# legend(['AdaSync','AdaComm','AdaLU'])
# # legend(['DSGD','FED','FED_ACSA','ACSA'])
# xlabel('communication')
# ylabel('training loss')
# # # legend(['DSGD','FED','FED_ACSA','ACSA','SCAFFOLD'])
# grid('on')

