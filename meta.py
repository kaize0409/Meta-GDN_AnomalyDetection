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
    return torch.mean((1 - y_true) * inlier_loss.flatten() + y_true * outlier_loss.flatten())


class Meta(nn.Module):

    def __init__(self, args, config):

        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.task_num = args.num_graph - 1
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

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
        num_task = len(x_train)
        losses = [0 for _ in range(self.update_step + 1)]
        results = []

        for t in range(num_task):
            prediction = self.net(x_train[t], vars=None, bn_training=True)
            loss = dev_loss(y_train[t], prediction)
            grad = torch.autograd.grad(loss, self.net.parameters())
            # update the parameters
            adapt_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            #before the first update
            with torch.no_grad():
                prediction_q = self.net(x_qry[t], self.net.parameters(), bn_training=True)
                loss_q = dev_loss(y_qry[t], prediction_q)
                losses[0] += loss_q

            # after the first update
            with torch.no_grad():
                prediction_q = self.net(x_qry[t], adapt_weights, bn_training=True)
                loss_q = dev_loss(y_qry[t], prediction_q)
                losses[1] += loss_q

            # for multiple step update
            for k in range(1, self.update_step):
                # evaluate the i-th task
                prediction = self.net(x_train[t], adapt_weights, bn_training=True)
                loss = dev_loss(y_train[t], prediction)
                # compute gradients on theta'
                grad = torch.autograd.grad(loss, adapt_weights)
                # perform one-step update step i + 1
                adapt_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, adapt_weights)))

                prediction_q = self.net(x_qry[t], adapt_weights, bn_training=True)
                loss_q = dev_loss(y_qry[t], prediction_q)
                losses[k+1] += loss_q

                # evaluation can be done here

        # finish all tasks
        loss_f = losses[-1] / num_task
        # update parameters
        self.meta_optim.zero_grad()
        loss_f.backward()
        self.meta_optim.step()

        # evaluate
        return loss_f


    def evaluate(self, x_train, y_train, x_test, y_test):

        prediction = self.net(x_train[0], vars=None, bn_training=True)
        loss = dev_loss(y_train[0], prediction)
        grad = torch.autograd.grad(loss, self.net.parameters())
        # update the parameters
        adapt_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

        # for multiple step update
        for k in range(1, self.update_step_test):
            # evaluate the i-th task
            prediction = self.net(x_train[0], adapt_weights, bn_training=True)
            loss = dev_loss(y_train[0], prediction)
            # compute gradients on theta'
            grad = torch.autograd.grad(loss, adapt_weights)
            # perform one-step update step i + 1
            adapt_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, adapt_weights)))

        for i in range(1, len(x_train)):
            # for multiple step update
            for k in range(self.update_step_test):
                # evaluate the i-th task
                prediction = self.net(x_train[i], adapt_weights, bn_training=True)
                loss = dev_loss(y_train[i], prediction)
                # compute gradients on theta'
                grad = torch.autograd.grad(loss, adapt_weights)
                # perform one-step update step i + 1
                adapt_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, adapt_weights)))

        y_pred = self.net(x_test, adapt_weights, bn_training=True)
        y_test = y_test.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        auc_roc, auc_pr, ap = aucPerformance(y_test, y_pred)
        return auc_roc, auc_pr, ap


    def evaluate2(self, x_train, y_train, x_test, y_test):

        for i in range(len(x_train)):
            prediction = self.net(x_train[i], vars=None, bn_training=True)
            loss = dev_loss(y_train[i], prediction)
            grad = torch.autograd.grad(loss, self.net.parameters())
            # update the parameters
            adapt_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))


            # for multiple step update
            for k in range(1, self.update_step_test):
                # evaluate the i-th task
                prediction = self.net(x_train[i], adapt_weights, bn_training=True)
                loss = dev_loss(y_train[i], prediction)
                # compute gradients on theta'
                grad = torch.autograd.grad(loss, adapt_weights)
                # perform one-step update step i + 1
                adapt_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, adapt_weights)))

        y_pred = self.net(x_test, adapt_weights, bn_training=True)
        y_test = y_test.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        auc_roc, auc_pr, ap = aucPerformance(y_test, y_pred)
        return auc_roc, auc_pr, ap

    def evaluate_backup(self, x_train, y_train, x_test, y_test):

        prediction = self.net(x_train, vars=None, bn_training=True)
        loss = dev_loss(y_train, prediction)
        grad = torch.autograd.grad(loss, self.net.parameters())
        # update the parameters
        adapt_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))


        # for multiple step update
        for k in range(1, self.update_step_test):
            # evaluate the i-th task
            prediction = self.net(x_train, adapt_weights, bn_training=True)
            loss = dev_loss(y_train, prediction)
            # compute gradients on theta'
            grad = torch.autograd.grad(loss, adapt_weights)
            # perform one-step update step i + 1
            adapt_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, adapt_weights)))


        y_pred = self.net(x_test, adapt_weights, bn_training=True)
        y_test = y_test.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        auc_roc, auc_pr, ap = aucPerformance(y_test, y_pred)
        return auc_roc, auc_pr, ap



def main():
    pass

if __name__ == '__main__':
    main()
