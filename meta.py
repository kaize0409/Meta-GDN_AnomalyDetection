import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np
from learner import Learner
from copy import deepcopy
from sklearn.metrics import auc, roc_curve
from utils import aucPerformance


def dev_loss(y_true, y_prediction):
    '''
    z-score based deviation loss
    :param y_true: true anomaly labels
    :param y_prediction: predicted anomaly label
    :return: loss in training
    '''
    confidence_margin = 5.0
    ref = torch.tensor(np.random.normal(loc=0.0, scale=1.0, size=5000), dtype=torch.float32)
    dev = (y_prediction - torch.mean(ref)) / torch.std(ref)
    inlier_loss = torch.abs(dev)
    outlier_loss = confidence_margin - dev
    outlier_loss[outlier_loss < 0.] = 0
    return torch.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)

class Meta(nn.Module):

    def __init__(self, args, config):

        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.task_num = args.task_num
        self.update_step = args.update_step

        self.net = Learner(config)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)


    def forward(self, x_train, y_train, x_qry, y_qry):
        '''
        :param x_train: [nb_task, batch_size, attr_dimension]
        :param y_train: [nb_task, batch_size]
        :param x_qry: [nb_task, qry_batch_size, attr_dimension]
        :param y_qry: [nb_task, qry_batch_size]
        :return:
        '''
        nb_task, batch_size, attr_dim = len(x_train), x_train[0], x_train[1]
        losses = [0 for _ in range(self.update_step + 1)]
        results = []

        for t in range(nb_task):
            # pred_task = self.net(x_train[t], self.net.parameters(), bn_training=True)
            pred_task = self.net(x_train[t], vars=None, bn_training=True)
            loss = dev_loss(y_train[t], pred_task)
            grad = torch.autograd.grad(loss, self.net.parameters())
            # update the parameters
            adapt_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            #before the first update
            with torch.no_grad():
                # prediction = self.net(x_train[t], self.net.parameters(), bn_training=True)
                # loss_t = dev_loss(y_train[t], prediction)
                if x_qry:
                    prediction = self.net(x_qry[t], self.net.parameters(), bn_training=True)
                    loss_t = dev_loss(y_qry[t], prediction)
                else:
                    prediction = self.net(x_train[t], self.net.parameters(), bn_training=True)
                    loss_t = dev_loss(y_train[t], prediction)
                losses[0] += loss_t

                # evaluation can be done here
            # after the first update
            with torch.no_grad():
                # prediction = self.net(x_train[t], adapt_weights, bn_training=True)
                # loss_t = dev_loss(y_train[t], prediction)
                if x_qry:
                    prediction = self.net(x_qry[t], adapt_weights, bn_training=True)
                    loss_t = dev_loss(y_qry[t], prediction)
                else:
                    prediction = self.net(x_train[t], adapt_weights, bn_training=True)
                    loss_t = dev_loss(y_train[t], prediction)
                losses[1] += loss_t

                # evaluation can be done here

            # for multiple step update
            for k in range(1, self.update_step):
                # evaluate the i-th task
                prediction = self.net(x_train[t], adapt_weights, bn_training=True)
                loss = dev_loss(y_train[t], prediction)
                # compute gradients on theta'
                grad = torch.autograd.grad(loss, adapt_weights)
                # perform one-step update step i + 1
                adapt_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, adapt_weights)))
                # evaluate on the same batch of samples

                # prediction = self.net(x_train[t], adapt_weights, bn_training=True)
                # loss_last = dev_loss(y_train[t], prediction)
                if x_qry:
                    prediction = self.net(x_qry[t], adapt_weights, bn_training=True)
                    loss_last = dev_loss(y_qry[t], prediction)
                else:
                    prediction = self.net(x_train[t], adapt_weights, bn_training=True)
                    loss_last = dev_loss(y_train[t], prediction)
                losses[k + 1] += loss_last


                # evaluation can be done here
            # get all output for all tasks
            results.append(self.net(x_train[t], adapt_weights, bn_training=True))

        # finish all tasks
        loss_f = losses[-1] / nb_task
        # update parameters
        self.meta_optim.zero_grad()
        loss_f.backward()
        self.meta_optim.step()

        # evaluate
        return torch.cat(results), loss_f

    def evaluating(self, x_test, y_test):
        net = deepcopy(self.net)
        y_pred = net(x_test)
        # loss = dev_loss(y_test, y_pred)
        # print(loss)
        y_test = y_test.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        # fpr, tpr, roc_auc = dict(), dict(), dict()
        # for i in range(2):
        #     fpr[i], tpr[i], _ = roc_curve(y_test, y_pred, pos_label=1)
        #     roc_auc[i] = auc(fpr[i], tpr[i])
        auc_roc, auc_pr, ap = aucPerformance(y_test, y_pred)
        print(auc_roc)
        del net
        return auc_roc, auc_pr, ap



def main():
    pass

if __name__ == '__main__':
    main()
