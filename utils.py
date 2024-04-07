import numpy as np
import pandas as pd

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split

import ot

from torch.autograd import grad
from torch.autograd import Variable

class Energy(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)

    def score(self, x, sigma=None):
        x = x.requires_grad_()
        logq = -self.net(x).sum()
        return torch.autograd.grad(logq, x, create_graph=True)[0]

class MLP(nn.Module):
    def __init__(self, input_dim=8, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, 1, bias=True)
        self.act = lambda x: torch.tanh(x)
    
    def forward(self, x_input):
        x = self.fc1(x_input)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x
    
class MLP2(nn.Module):
    def __init__(self, input_dim=8, hidden_num=100):
        super().__init__()
        self.fc1 = nn.Linear(input_dim+1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = lambda x: torch.tanh(x)
    
    def forward(self, x_input, t):
        inputs = torch.cat([x_input, t], dim=1)
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x
    
class RectifiedFlow():
    def __init__(self, model=None, num_steps=1000):
        self.model = model
        self.N = num_steps
  
    def get_train_tuple(self, z0=None, z1=None):
        t = torch.rand((z1.shape[0], 1)).cuda()
        z_t =  t * z1 + (1.-t) * z0 # + sqrt_t*z
        # z_t = np.sin(t * np.pi/2) * z1 + np.cos(t * np.pi/2) * z0
        target = z1 - z0 # + (1-2*t)/(2*sqrt_t)
        return z_t, t, target

def initial_classfier(d, o):
    return nn.Sequential(
        nn.Linear(d, 128), nn.ReLU(),
        nn.Dropout(p=0.25),
        nn.BatchNorm1d(num_features=128),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Dropout(p=0.25),
        nn.Linear(128, o)
    ).cuda()

def langevin_dynamics(score_fn, x, eps=0.1, n_steps=1000):
    for i in range(n_steps):
        x = x + eps / 2.0 * score_fn(x).detach()
        x = x + torch.randn_like(x) * np.sqrt(eps)
    return x

def cal_wasserstein_loss(s_feature, t_feature, reg=0.1, cuda=True):
    M = ot.dist(s_feature, t_feature)
    size = t_feature.shape[0]
    a, b = torch.ones((size,)) / size, torch.ones((size,)) / size
    if cuda:
        a= a.cuda()
        b= b.cuda()
    # emd2, sinkhorn2
    wasserstein_loss = ot.emd2(a, b, M, reg)
    return wasserstein_loss

def cal_entropy(s_pred):
    h = F.softmax(s_pred, dim=1) * F.log_softmax(s_pred, dim=1)
    return -1.0 * h.mean()

def get_energy(model, tmp, label, w=1):
    s_x = Variable(tmp, requires_grad = True).cuda()
    s_y = label.cuda()
    
    s_pred = model.forward(s_x)
    ce_loss = F.cross_entropy(s_pred, s_y.long())
    nm_loss = cal_entropy(s_pred)
    loss = w * ce_loss  + (1-w) * nm_loss
    y_x = grad(loss, s_x, grad_outputs=torch.ones_like(loss), create_graph=False, retain_graph=False)[0]
    return y_x

def eval_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for step, t_data in enumerate(test_loader):
            x, y = t_data
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            s_pred = model.forward(x)
            pred = s_pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.long().view_as(pred)).sum().item()
            total += len(y)
    model.train()
    return round(correct / total, 4)

def train_model(model, train_loader, val_loader, epochs, lr=0.001, cuda=True):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    best_val = 0
    for epoch in range(epochs):
        for step, src_data in enumerate(train_loader):
            if len(src_data) == 2:
                s_x, s_y = src_data
            if cuda:
                s_x = s_x.cuda()
                s_y = s_y.cuda()
            s_pred = model.forward(s_x)
            ce_loss = F.cross_entropy(s_pred,  s_y.long())
            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()
        if val_loader is not None and (epoch+1) % 2 == 0:
            acc_val = eval_model(model, val_loader)
            if best_val < acc_val:
                best_val = acc_val
                best_model = deepcopy(model)
    if val_loader is not None:
        return best_model
    return model

def get_pseudo_dataset(model, data, confidence_q=0.1):
    model.eval()
    new_data, new_label  = [], []
    with torch.no_grad():
        pred = model.forward(data)
        new_data.append(data)
        new_label.append(pred)
        new_data = torch.cat(new_data, dim=0)
        new_label = torch.cat(new_label, dim=0)
        confidence = np.array(torch.Tensor.cpu(new_label.amax(dim=1) - new_label.amin(dim=1)))
        alpha = np.quantile(confidence, confidence_q)
        conf_index = np.array(np.argwhere(confidence >= alpha)[:,0])
        pseudo_y = new_label.argmax(dim=1)
    model.train()
    return new_data[conf_index], pseudo_y[conf_index]