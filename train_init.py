import argparse
import os

import torch.nn as nn
import pandas as pd
from utils import *

# Credit to https://github.com/Ending2015a/toy_gradlogp 
def dsm_loss(energy_model, x, x_s, w):
    # sigma = 1
    sigma = (0.5 * torch.abs(x - x_s).median(0)[0] + 0.5 * torch.abs(x - x_s).min(0)[0])
    x = x.requires_grad_()
    v = torch.randn_like(x) * sigma
    x_ = x + v
    s = energy_model.score(x_)
    loss = w * (torch.norm(s + v / (sigma ** 2), dim=-1) ** 2)
    loss = loss.mean() / 2.0
    return loss

def weighted_target_dataset(args, classifier, target_loader):
    new_x, new_y, new_label  = [], [], []
    with torch.no_grad():
        for step, data in enumerate(target_loader):
            x, y = data
            x = x.cuda()
            y = y.cuda()
            pred = classifier.forward(x)
            new_x.append(x)
            new_y.append(y)
            new_label.append(pred)
        new_x = torch.cat(new_x, dim=0)
        new_y = torch.cat(new_y, dim=0)
        new_label = torch.cat(new_label, dim=0)
        confidence = np.array(torch.Tensor.cpu(new_label.amax(dim=1) - new_label.amin(dim=1)))
        weight = confidence.mean()/confidence
    weight = torch.tensor(1 + weight).log()
    weight = torch.clamp(weight, min=0.1, max=10)

    w_dataset = TensorDataset(new_x, new_y, weight)
    target_loader_w = DataLoader(dataset=w_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    return target_loader_w

def train_score_network(args, energy_model, optimizer, source_loader, target_loader_w):
    for e in range(args.epoch2):
        loss_value = np.array([0])
        for step, (src_data, tgt_data) in enumerate(zip(source_loader, target_loader_w)):
            s_x, s_y = src_data
            t_x, t_y, t_w = tgt_data
            s_x = s_x.cuda()
            t_x = t_x.cuda()
            t_w = t_w.cuda()
            
            t_pred = classifier.forward(t_x)
            t_pred = t_pred.argmax(dim=1, keepdim=True).float()
            loss = dsm_loss(energy_model, t_x, s_x, t_w)
            loss_value = loss_value + np.array([loss.item()])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        if e % 100 == 0:
            print("Stage 1: ", e, loss_value)
    
    return energy_model
            
def train_rectified_flow(args, rectified_flow, optimizer, source_loader, target_loader_w):
    cross_entropy = nn.CrossEntropyLoss(reduction='none')
    for e in range(args.epoch3):
        loss_value = np.array([0])
        for step, (src_data, tgt_data) in enumerate(zip(source_loader, target_loader_w)):
            s_x, s_y = src_data
            t_x, t_y, t_w = tgt_data
            s_x = s_x.cuda()
            t_x = t_x.cuda()
            t_w = t_w.cuda()

            z_t, t, target = rectified_flow.get_train_tuple(z0=s_x, z1=t_x)
            pred = rectified_flow.model(z_t, t)
            
            loss = ((target - pred)).view(pred.shape[0], -1).abs().pow(2).sum(dim=1)
            # norm = torch.norm(target, dim=1) # **2
            # minv = norm.min()
            # loss = (t_w*(minv/norm)*loss).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_value = loss_value + np.array([loss.item()])
            
        if e % 100 == 0:
            print("Stage 2: ", e, loss_value)
    
    return rectified_flow


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code for GGF')
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--dimension', type=int, default=8, help="dimension of the features")
    parser.add_argument('--class_num', type=int, default=2, help="number of classes")
    parser.add_argument('--task', type=str, default='portraits')
    parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=1024, help="batch size")
    parser.add_argument('--save_path', type=str, default='save/', help="modules path")
    parser.add_argument('--epoch1', type=int, default=2000, help="epoch for training initial classifier")
    parser.add_argument('--epoch2', type=int, default=50000, help="epoch for training score network")
    parser.add_argument('--epoch3', type=int, default=50000, help="epoch for training rectified flow")
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    batch_size = args.batch_size
    dimension = args.dimension
    datapkl = pd.read_pickle('dataset/%s.pkl' % args.task)
    z_all = datapkl['data']
    y_all = datapkl['label']
    
    s_data = torch.from_numpy(z_all[0])
    s_label = torch.from_numpy(y_all[0])
    t_data = torch.from_numpy(z_all[-1])
    t_label = torch.from_numpy(y_all[-1])
    s_dataset = TensorDataset(s_data, s_label)
    t_dataset = TensorDataset(t_data, t_label)
    source_loader = DataLoader(dataset=s_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = DataLoader(dataset=t_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    source_loader_test = DataLoader(dataset=s_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    target_loader_test = DataLoader(dataset=t_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # init modules
    energy_model = Energy(MLP(dimension, 1000)).cuda()
    classifier = initial_classfier(dimension, args.class_num)
    mlp = MLP2(8, hidden_num=100).cuda()
    rectified_flow = RectifiedFlow(model=mlp, num_steps=100)
    
    classifier = train_model(classifier, source_loader, source_loader, args.epoch1)
    print("Initual Accuracy: ", eval_model(classifier, target_loader_test))
    
    optimizer_score = optim.SGD(energy_model.parameters(), lr=args.lr, momentum=0.9)
    optimizer_rectified = optim.SGD(list(rectified_flow.model.parameters()), lr=args.lr, momentum=0.9)
    

    target_loader_w = weighted_target_dataset(args, classifier, target_loader_test)
    
    energy_model = train_score_network(args, energy_model, optimizer_score, source_loader, target_loader_w)
    rectified_flow = train_rectified_flow(args, rectified_flow, optimizer_rectified, source_loader, target_loader_w)
    
    torch.save(energy_model.state_dict(), args.save_path + 'score_network/score_%s.pt' % args.task)
    torch.save(classifier.state_dict(), args.save_path + 'initial_classifier/classifier_%s.pt' % args.task)
    torch.save(rectified_flow.model.state_dict(), args.save_path + 'rectified_flow/rectified_%s.pt' % args.task)

    print("Finished!")
    