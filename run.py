import torch
import numpy as np
import  scipy.stats
# from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
from meta import *
from getConfig import modelArch
from data import DataProcessor, task_generator, test_task_generator, test_task_generator_backup
from models import SGC
from sklearn.metrics import auc, roc_curve
from utils import  aucPerformance



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    num_labeled_ano = 10 # each graph (auxiliary or target) has 10 sampled anomaly nodes

    results_meta_gdn = []
    results_gdn = []
    for t in range(args.num_run):
        dataset = DataProcessor(num_graph=args.num_graph, degree=2, data_name=args.data_name)
        dataset.data_loader()

        # training meta-gdn
        print("Meta-GDN training...")
        print("In %d-th run..." % (t + 1))
        [feature_list, l_list, ul_list], [target_feature, target_l_idx, target_ul_idx] = dataset.sample_anomaly(num_labeled_ano)

        config = modelArch(feature_list[0].shape[1], 1)

        maml = Meta(args, config).to(device)
        best_val_auc = 0
        for e in range(1, args.num_epochs + 1):

            # training
            maml.train()
            x_train, y_train, x_qry, y_qry = task_generator(feature_list, l_list, ul_list, bs=args.bs, device=device)
            loss = maml(x_train, y_train, x_qry, y_qry)
            torch.save(maml.state_dict(), 'temp.pkl')
            # validation
            model_meta_eval = Meta(args, config).to(device)
            model_meta_eval.load_state_dict(torch.load('temp.pkl'))
            model_meta_eval.eval()
            x_train, y_train, x_val, y_val = test_task_generator(target_feature, target_l_idx,
                                                                   target_ul_idx, args.bs,
                                                                   dataset.target_label,
                                                                   dataset.target_idx_val, device)
            auc_roc, auc_pr, ap = model_meta_eval.evaluate(x_train, y_train, x_val, y_val)
            print("%dth Epoch: Training Loss %4f, Validation, AUC-ROC %.4f, AUC-PR %.4f, AP %.4f" % (e, loss.item(), auc_roc, auc_pr, ap))

            if auc_roc > best_val_auc: # store the best model
                best_val_auc = auc_roc
                torch.save(maml.state_dict(), 'best_meta_GDN.pkl')

        print("End of training.")
        # testing
        print("Load the best performing Meta-GDN model and Evaluate")
        maml = Meta(args, config).to(device)
        maml.load_state_dict(torch.load('best_meta_GDN.pkl'))
        maml.eval()
        x_train, y_train, x_test, y_test = test_task_generator(target_feature, target_l_idx,
                                                               target_ul_idx, args.bs,
                                                               dataset.target_label,
                                                               dataset.target_idx_test, device)
        auc_roc, auc_pr, ap = maml.evaluate(x_train, y_train, x_test, y_test)
        print("Testing performance of Meta-GDN: AUC-ROC %.4f, AUC-PR %.4f, AP %.4f" % (auc_roc, auc_pr, ap))
        print("End of evaluating.")
        results_meta_gdn.append(auc_roc)

        # GDN training
        print('GDN training...')
        model = SGC(target_feature.shape[1], 1).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=args.gdn_lr, weight_decay=0)
        best_val_auc = 0
        for e in range(1, args.num_epochs_GDN + 1):

            x_train, y_train, x_test, y_test = test_task_generator_backup(target_feature, target_l_idx,
                                                                   target_ul_idx, num_labeled_ano * 2,
                                                                   dataset.target_label,
                                                                   dataset.target_idx_test, device)
            x_train, y_train = x_train.to(device), y_train.to(device)
            model.train()
            optim.zero_grad()
            y_pred = model(x_train)
            loss = dev_loss(y_train, y_pred)
            loss.backward()
            optim.step()

            # validation
            _, _, x_val, y_val = test_task_generator_backup(target_feature, target_l_idx,
                                                                   target_ul_idx, num_labeled_ano * 2,
                                                                   dataset.target_label,
                                                                   dataset.target_idx_val, device)
            model.eval()
            y_pred = model(x_val).detach().cpu().numpy()
            y_val = y_val.detach().cpu().numpy()
            auc_roc, auc_pr, ap = aucPerformance(y_val, y_pred)
            print("%dth Epoch: Training Loss %4f, Validation, AUC-ROC %.4f, AUC-PR %.4f, AP %.4f" % (e, loss.item(), auc_roc, auc_pr, ap))

            if auc_roc > best_val_auc: # store the best model
                best_val_auc = auc_roc
                torch.save(model.state_dict(), 'best_GDN.pkl')

        # testing
        model = SGC(target_feature.shape[1], 1).to(device)
        model.load_state_dict(torch.load('best_GDN.pkl'))
        model.eval()
        _, _, x_test, y_test = test_task_generator_backup(target_feature, target_l_idx,
                                                               target_ul_idx, num_labeled_ano * 2,
                                                               dataset.target_label,
                                                               dataset.target_idx_test, device)
        y_pred = model(x_test).detach().cpu().numpy()
        y_test = y_test.detach().cpu().numpy()
        auc_roc, auc_pr, auc_pr = aucPerformance(y_test, y_pred)
        print("Testing performance of GDN: AUC-ROC: %.4f, AUC-PR: %.4f, AP: %.4f" % (auc_roc, auc_pr, ap))
        results_gdn.append(auc_roc)

    print(results_gdn)
    print(results_meta_gdn)
    print("Average Testing performance of GDN: AUC-ROC: %.4f" % (sum(results_gdn)*1.0/len(results_gdn)))
    print("Average Testing performance of meta-GDN: AUC-ROC: %.4f" % (sum(results_meta_gdn) * 1.0 / len(results_meta_gdn)))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_name', help='pubmed/yelp', default='pubmed')
    argparser.add_argument('--num_epochs', type=int, help='epoch number', default=100)
    argparser.add_argument('--num_epochs_GDN', type=int, help='epoch number for GDN', default=100)
    argparser.add_argument('--gdn_lr', type=float, help='learning rate for GDN', default=0.01)
    argparser.add_argument('--bs', type=int, help='batch size', default=16)
    argparser.add_argument('--num_graph', type=int, help='meta batch size, namely task num', default=5)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.003)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.5)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=3)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=3)
    argparser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    argparser.add_argument('--num_run', type=int, help='run the experiments multiple times', default=100)

    args = argparser.parse_args()

    main()