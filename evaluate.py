import torch
import numpy as np
import  scipy.stats
# from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
from datetime import date, datetime
from meta import Meta
from getConfig import modelArch
from data import SGC_process, DataLoaderN, DataLoader
from process_task import task
from models import SGC, deviation_loss
from sklearn.metrics import auc, roc_curve
from utils import  aucPerformance

def main():

    torch.manual_seed(666)
    torch.cuda.manual_seed_all(666)
    np.random.seed(666)

    print(args)
    nb_epochs = 20
    nb_runs = 10
    nb_try = 20
    nb_batch_maml = 64
    nb_batch = 32
    lr_1 = 0.05
    lr_s = lr_1 * args.task_num
    tr = 0.6
    name = "pubmed"
    # features, labels, idx_train, y_train, idx_val, y_val, idx_labeled, train_unlabeled = SGC_process(data_name, degree=2, l_ratio=0.08, tr_ratio=0.8)
    # print(args.task_num)

    aucfile = 'results/auc_' + datetime.now().strftime("%m_%d_%H_%M") + name + '.txt'
    with open(aucfile, 'a') as f:
        f.write("settings: {labeled ratio: %f, training ratio: %f, epochs: %d, update_step: %d}\n" % (lr_1, tr, nb_epochs, args.update_step))
        for t in range(nb_try):
            taskData = task(nb_task=args.task_num, degree=2, l_ratio=lr_1, t_ratio=tr, name=name)
            taskData.loadNProcess()
            f.write("target data name:" + taskData.f_name[-1] + "\n")
            f.write("%d-th try: \n" % t)
            for i in range(nb_runs):
                # training meta-gdn
                print("Meta-GDN training...")
                print("In %d-th run..." % (i + 1))
                f.write("%d-th run\n" % i)
                feature_list, label, l_list, ul_list, idx_test = taskData.sampleAnomaly()
                config = modelArch(feature_list[0].shape[1], args.n_way)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                maml = Meta(args, config).to(device)
                # # stats of parameters to be updated
                tmp = filter(lambda x: x.requires_grad, maml.parameters())
                num = sum(map(lambda x: np.prod(x.shape), tmp))
                print(maml)
                print("Total #trainable tensors: ", num)
                batch_gen = DataLoaderN(feature_list, l_list, ul_list, b_size=16, b_size_qry=6, nb_task=args.task_num, device=device)
                maml.train()
                for e in range(1, nb_epochs + 1):
                    print("Running %d-th epoch" % e)
                    epoch_loss = 0
                    epoch_acc = 0
                    for b in range(nb_batch_maml):
                        x_train, y_train, x_qry, y_qry = batch_gen.getBatch(qry=False)
                        y_pred, loss = maml(x_train, y_train, x_qry, y_qry)
                        epoch_loss += loss
                    print("Epoch loss: %f" % epoch_loss)
                print("End of training.")
                # testing
                print("Evaluating Meta-GDN model")
                maml.eval()
                x_test, y_test = feature_list[args.task_num-1][idx_test].to(device), label[idx_test].to(device)
                auc_roc, auc_pr, ap = maml.evaluating(x_test, y_test)
                print("End of evaluating.")
                f.write("Meta-GDN auc_roc: %.5f, auc_pr: %.5f, ap: %.5f\n" % (auc_roc, auc_pr, ap))

                # g-dev training
                print('GDN training...')
                features, labels, idx_labeled, idx_unlabeled, idx_test = SGC_process(taskData.target, degree=2, l_ratio=lr_s, tr_ratio=tr)
                # print("finish loading data...")
                attr_dim = features.shape[1]
                # print("%d-th run:" % i)
                model = SGC(attr_dim, 1).to(device)
                # print(model)
                optim = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
                # loss = deviation_loss()
                data_sampler = DataLoader(features, idx_labeled, idx_unlabeled, b_size=32)
                model.float()
                model.train()
                for e in range(1, nb_epochs + 1):
                    # print('Epoch: %d' % e)
                    epoch_loss = 0
                    epoch_acc = 0
                    for b in range(nb_batch):
                        x_b, y_b = data_sampler.getBatch()
                        x_b, y_b = x_b.to(device), y_b.to(device)
                        y_pred = model(x_b)
                        loss = deviation_loss(y_b, y_pred)
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                        epoch_loss += loss.item()
                    print("epoch loss %f" % epoch_loss)
                # validation
                model.eval()
                # print(idx_val.shape)
                x_val = features[idx_test].to(device)
                # print(x_val.shape)
                y_pred = model(x_val).detach().cpu().numpy()
                y_val = labels[idx_test].detach().cpu().numpy()
                # fpr, tpr, roc_auc = dict(), dict(), dict()
                # for i in range(2):
                #     fpr[i], tpr[i], _ = roc_curve(y_val, y_pred, pos_label=1)
                #     roc_auc[i] = auc(fpr[i], tpr[i])
                auc_roc, _, auc_pr = aucPerformance(y_val, y_pred)
                print("GDN auc_roc: %.5f, auc_pr: %.5f" % (auc_roc, auc_pr))
                f.write("GDN auc_roc: %.5f, auc_pr: %.5f\n" % (auc_roc, auc_pr))

    f.close()



if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=20)
    argparser.add_argument('--n_way', type=int, help='n way', default=1)
    # argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    # argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    # argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
    # argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=5)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)

    args = argparser.parse_args()

    main()