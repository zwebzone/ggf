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
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=1024, help="batch size")
    
    parser.add_argument('--alpha', type=int, default=4, help="alpha")
    parser.add_argument('--iterations', type=int, default=120, help="iterations = alpha * T")
    parser.add_argument('--lamb', type=float, default=0, help="the weight of two classifier-based energy functions")
    parser.add_argument('--eta1', type=float, default=0.03, help="step size of distribution-based energy functions")
    parser.add_argument('--eta2', type=float, default=0.08, help="step size of classifier-based energy functions")
    parser.add_argument('--eta3', type=float, default=0.01, help="step size of sample-based energy functions")
    parser.add_argument('--confidence', type=float, default=0.05, help="confidence threshold, 0 means finetuning with preserved learning")
    parser.add_argument('--save_path', type=str, default='save/', help="modules path")
    
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
    

    best_value = 0
    best_model = None
    
    # init temp variables
    tmp = deepcopy(s_data).repeat((5,1)).cuda()
    tmp_labels = deepcopy(s_label).repeat(5).cuda()
    tmp_clf = deepcopy(classifier)
    eta_tmp = torch.tensor([args.eta1]).cuda()
    
    # optimizer for self-training/fine-tuning
    optimizer_inner = optim.SGD([
        {"params":tmp_clf.parameters(), "lr":0.0001},
    ], weight_decay=1e-3)

    print("Initial accuracy: ", eval_model(classifier, target_loader_test))
    
    for epoch in range(args.iterations):
        y_x = get_energy(tmp_clf, tmp, tmp_labels.cuda(), w=1-args.lamb)
        tmp = tmp + eta_tmp / 2.0 * (energy_model.score(tmp).detach() )
        tmp = tmp + torch.randn_like(tmp) * torch.sqrt(eta_tmp)[:, None]
        
        t = torch.ones((tmp.shape[0], 1)) * epoch / args.iterations
        v_t = rectified_flow.model(tmp.detach().to(torch.float32), t.cuda()) * args.eta3
        tmp = tmp + v_t.detach() * args.eta3
        tmp = tmp - y_x.sign().detach() * args.eta2

        if epoch % args.alpha == args.alpha-1:
            if args.confidence == 0:
                tmp_pseudo = deepcopy(tmp)
                tmp_labels_pseudo = deepcopy(s_label).cuda()
            else:
                tmp_pseudo, tmp_labels_pseudo = get_pseudo_dataset(tmp_clf, tmp, confidence_q=args.confidence)
            
            for _ in range(5):
                pred_t = tmp_clf(tmp_pseudo)
                ce_loss = F.cross_entropy(pred_t, tmp_labels_pseudo)
                optimizer_inner.zero_grad()
                ce_loss.backward()
                optimizer_inner.step()

            acc = eval_model(tmp_clf, target_loader_test)
            if acc > best_value:
                best_value = acc
                best_model = deepcopy(tmp_clf)
                print("Epoch: %d, Best Accuracy: %f" % (epoch, best_value))
            
            
            # print("Epoch: %d, Current Accuracy: %f, Best Accuracy: %f" % (epoch, acc, best_value))
    
    torch.save(best_model.state_dict(), args.save_path + 'updated_classifier/classifier_%s.pt' % args.task)
    
    print("Finished!")