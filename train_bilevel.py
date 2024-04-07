import argparse
import os

import torch.nn as nn
import pandas as pd
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Code for GGF')
    parser.add_argument('--gpu_id', type=str, default='0', help="device id to run")
    parser.add_argument('--dimension', type=int, default=8, help="dimension of the features")
    parser.add_argument('--class_num', type=int, default=2, help="number of classes")
    parser.add_argument('--task', type=str, default='portraits')
    parser.add_argument('--batch_size', type=int, default=1024, help="batch size")
    
    parser.add_argument('--alpha', type=int, default=10, help="alpha")
    parser.add_argument('--iterations', type=int, default=120, help="iterations = alpha * T")
    parser.add_argument('--lamb', type=float, default=0, help="the weight of two classifier-based energy functions")
    parser.add_argument('--eta1', type=float, default=0.03, help="step size of distribution-based energy functions")
    parser.add_argument('--eta2', type=float, default=0.08, help="step size of classifier-based energy functions")
    parser.add_argument('--eta3', type=float, default=0.01, help="step size of sample-based energy functions")
    parser.add_argument('--confidence', type=float, default=0.05, help="confidence threshold, 0 means finetuning with preserved learning")
    parser.add_argument('--save_path', type=str, default='save/', help="modules path")
    
    parser.add_argument('--epoch', type=int, default=100, help="epoch for bilevel optimization")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate for bilevel optimization")
    parser.add_argument('--update_epoch', type=int, default=2, help="epoch for classifier updating")
    parser.add_argument('--update_lr', type=float, default=0.0001, help="learning rate for classifier updating")
    parser.add_argument('--frequency', type=int, default=5, help="Update alpha every how many epochs")
    
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
    
    # load modules
    energy_model.load_state_dict(torch.load(args.save_path + 'score_network/score_%s.pt' % args.task))
    classifier.load_state_dict(torch.load(args.save_path + 'initial_classifier/classifier_%s.pt' % args.task))
    rectified_flow.model.load_state_dict(torch.load(args.save_path + 'rectified_flow/rectified_%s.pt' % args.task))
    

    best_acc = 0
    best_model = None
    
    # init temp variables
    tmp = deepcopy(s_data).repeat((5,1)).cuda()
    tmp_labels = deepcopy(s_label).repeat(5).cuda()
    tmp_clf = deepcopy(classifier)

    print("Initial accuracy: ", eval_model(classifier, target_loader_test))
    
    
    eta1 = torch.ones(args.class_num).cuda() * args.eta1
    eta2 = torch.ones(args.class_num).cuda() * args.eta2
    eta3 = torch.ones(args.class_num).cuda() * args.eta3
    
    eta1.requires_grad = True
    eta2.requires_grad = True
    eta3.requires_grad = True
    
    # optimizer for hyperparameter eta
    optimizer = optim.SGD([
        {"params":eta1, "lr":args.lr},
        {"params":eta2, "lr":args.lr},
        {"params":eta3, "lr":args.lr},
    ], weight_decay=1e-3)
    
    domain_number = round(args.iterations/args.alpha)
    sum_num = domain_number
    step_num_per_domain = args.alpha
    
    for e in range(args.epoch):
        loss_value = 0
        for step, data in enumerate(zip(source_loader, target_loader)):
            src_data, tgt_data = data
            s_x, s_y = src_data
            t_x, t_y = tgt_data
            s_x = s_x.cuda()
            t_x = t_x.cuda()
            s_y = s_y.cuda()

            tmp_clf = deepcopy(classifier)
            clf_optimizer = optim.SGD([
                {"params":tmp_clf.parameters(), "lr":args.update_lr},
            ], weight_decay=1e-3)

            tmp = deepcopy(s_x.detach())

            w_before = cal_wasserstein_loss(s_x, t_x)
            for num in range(round(sum_num)):
                for _ in range(step_num_per_domain):
                    y_x = get_energy(tmp_clf, s_x, s_y, w=1 - args.lamb)
                    eta1_tmp = torch.take(eta1, s_y.detach())
                    eta2_tmp = torch.take(eta2, s_y.detach())
                    eta3_tmp = torch.take(eta3, s_y.detach())

                    s_x = s_x + eta1_tmp[:, None] / 2.0 * (energy_model.score(s_x).detach() - y_x.sign().detach())
                    s_x = s_x + torch.randn_like(s_x) * torch.sqrt(eta1_tmp)[:, None]

                    t = torch.ones((s_x.shape[0], 1)) * (num*step_num_per_domain+_) / (sum_num*step_num_per_domain)
                    v_t = rectified_flow.model(s_x.detach().to(torch.float32), t.cuda())
                    s_x = s_x + v_t.detach() * eta3_tmp[:, None]
                    s_x = s_x + eta2_tmp[:, None] * (- y_x.sign().detach())


                for _ in range(args.update_epoch):
                    pred_t = tmp_clf(s_x.detach())
                    ce_loss = F.cross_entropy(pred_t,  s_y.detach().long())
                    clf_optimizer.zero_grad()
                    ce_loss.backward()
                    clf_optimizer.step()

            w_after = cal_wasserstein_loss(s_x, t_x)

            loss = w_after / w_before

            loss_value = loss_value + loss.item()
            optimizer.zero_grad()
            loss.backward()        
            total_grad_norm = nn.utils.clip_grad_norm_(energy_model.parameters(), 100)
            optimizer.step()
        # if e % 10 == 0:
        # print(e, loss_value, eta1, eta2, eta3)
        
        if e % args.frequency == 0:
            eta1_d = eta1.detach()
            eta2_d = eta2.detach()
            eta3_d = eta3.detach()
            
            tmp_model = deepcopy(classifier)
            tmp = deepcopy(s_data).cuda()
            # print("before_dis", cal_wasserstein_loss(tmp, t_data.cuda()).item())
            optimizer_inner = optim.SGD([
                    {"params":tmp_model.parameters(), "lr":args.update_lr},
            ], weight_decay=1e-3)
            
            s_y_tmp = deepcopy(s_y)
            dis = cal_wasserstein_loss(tmp, t_data.cuda()).item()
            count = 0
            tmp_sum = sum_num
            sum_num = 0
            while count < 1:
                tmps = []
                for num in range(step_num_per_domain):
                    y_x = get_energy(tmp_model, tmp, s_label.cuda(), w=1 - args.lamb)
                    eta1_d_tmp = torch.take(eta1_d, s_label.cuda().detach())
                    eta2_d_tmp = torch.take(eta2_d, s_label.cuda().detach())
                    eta3_d_tmp = torch.take(eta3_d, s_label.cuda().detach())

                    tmp = tmp + eta1_d_tmp[:, None] / 2.0 * (energy_model.score(tmp).detach())
                    tmp = tmp + torch.randn_like(tmp) * torch.sqrt(eta1_d_tmp)[:, None]
                    if count < sum_num:
                        t = torch.ones((tmp.shape[0], 1)) * (count*step_num_per_domain+num) / (tmp_sum*step_num_per_domain)
                        v_t = rectified_flow.model(tmp.detach().to(torch.float32), t.cuda()) 
                        tmp = tmp + v_t.detach() * eta3_d_tmp[:, None]
                    tmp = tmp + eta2_d_tmp[:, None] * - y_x.sign().detach()
                    tmps.append(tmp)
                    
                tmp_dis = cal_wasserstein_loss(tmp, t_data.cuda()).item()
                if tmp_dis > dis:
                    count += 1
                else:
                    count = 0
                sum_num += 1
                dis = tmp_dis
                
                acc = eval_model(tmp_model, target_loader_test)
                # print("#%d Intermediate Domains, Wasserstein Distance: %.3f, Accuracy: %.3f" % (sum_num, tmp_dis, acc))

                for _ in range(args.update_epoch):
                    pred_t = tmp_model(torch.cat(tmps))
                    ce_loss = F.cross_entropy(pred_t,  s_label.repeat(step_num_per_domain).cuda().long())
                    optimizer_inner.zero_grad()
                    ce_loss.backward()
                    optimizer_inner.step()
            step_num_per_domain = max(round(step_num_per_domain*sum_num/domain_number), 2)
            
            if acc > best_acc:
                best_acc = acc
            print("Epoch: %d, Alpha: %d, Best Accuracy: %.3f, Current Accuracy: %.3f, Wasserstein distance: %.3f" % (e, step_num_per_domain, best_acc, acc, tmp_dis))
                
    print("Finished!")