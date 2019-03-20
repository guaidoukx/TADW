import numpy as np
import collections
from numpy import linalg as la
import torch
import torch.nn.functional as F
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection

LABEL = {
    'Theory': 0,
    'Case_Based': 1,
    'Genetic_Algorithms': 2,
    'Neural_Networks': 3,
    'Probabilistic_Methods': 4,
    'Reinforcement_Learning': 5,
    'Rule_Learning': 6
}


class DataProcess:
    def __init__(self, _node_path, _edge_path):
        self.node_path = _node_path
        self.edge_path = _edge_path
        self.ids, self.num_nodes, self.target = self.load_nodes()
        self.T = self.load_text_features()
        self.A = self.load_edges_A()
    
    def load_nodes(self):
        """
        这里的ID都是string类型的，没有转换为int。ids 也是 idx[string]:int
        """
        ids, labels = {}, []
        with open(self.node_path, 'r') as f:
            line = f.readline()
            while line:
                line = line.split()
                ids[line[0]] = len(ids)  # turn string ids to int ids. (from 0 to number of nodes)
                labels.append(LABEL[line[-1]])  # turn labels to int
                line = f.readline()
        num_nodes = len(ids)
        labels = np.array(labels)
        return ids, num_nodes, labels
    
    def load_text_features(self):
        """
        这里要转化为np，所以将text features转换为int
        :return:
        """
        features = []
        with open(self.node_path, 'r') as f:
            line = f.readline()
            while line:
                line = line.split()
                a = line[1:-1]
                # if len(a) == self.num_nodes-2:
                features.append(list(map(int, a)))
                line = f.readline()
        features_M = np.array(features)
        # TF/IDF process
        for i in range(features_M.shape[1]):
            if np.sum(features_M[:, i]) > 0:
                features_M[:, i] = features_M[:, i] * np.log(self.num_nodes / np.sum(features_M[:, i]))
        if features_M.shape[1] > 200:  # when feature size > 200, use SVD to decrease the dimensions of T
            U, S, VT = la.svd(features_M)
            Ud = U[:, 0:200]
            Sd = S[0:200]
            features_M = np.array(Ud) * Sd.reshape(200)
        for i in range(len(features_M)):  # normalization of matrix T.
            features_M[i] = features_M[i] / np.linalg.norm(features_M[i], ord=2)
        print(features_M.T)
        return features_M.T
    
    def load_edges_A(self):
        edges_degree = collections.defaultdict(int)
        A = np.zeros((self.num_nodes, self.num_nodes))
        with open(self.edge_path, 'r') as f:
            line = f.readline()
            while line:
                line_splited = line.split()
                A[self.ids[line_splited[0]], self.ids[line_splited[1]]] = 1.0
                edges_degree[self.ids[line_splited[0]]] += 1
                line = f.readline()
        for i in range(self.num_nodes):  #
            A[i, :] = A[i, :] / (edges_degree[i] + 0.0001)  # 0.0001 -> in case degrees of some nodes will be 0
        return A
    
    def get_M(self, order=2):
        """
        the matrix version of deepwalk. To decrease the computational cost, default order=2
        """
        assert isinstance(order, int)
        if order > 1:
            A = self.A
            for i in range(2, order + 1):
                A = A + A ** i
            self.A = A / order
    
    def return_all(self):
        self.get_M()
        return self.A, self.T, self.target


class TadwModel:
    def __init__(self, A, T, k=80, lamb=0.2, lr=0.1, lower_control=10 ** -15):
        self.A = A
        self.T = T
        self.k = k
        self.lamb = lamb
        self.lr = lr
        self.lower_control = lower_control
        
        self.W = np.random.randn(self.k, self.A.shape[0])
        self.H = np.random.randn(self.k, self.T.shape[0])
        self.losses = []
        print(self.W)
    
    def update_W(self):
        HT = np.dot(self.H, self.T)
        # grad = self.lamb * self.W - np.dot(HT, self.A - np.dot(np.transpose(HT), self.W))  # when use summation
        grad = self.lamb * self.W - 2 / np.prod(self.A.shape) * np.dot(HT, self.A - np.dot(HT.T, self.W))
        self.W = self.W - grad * self.lr
        self.W[self.W < self.lower_control] = self.lower_control
    
    def update_H(self):
        inside = self.A - np.dot(np.dot(self.W.T, self.H), self.T)
        # grad = self.lamb * self.H - np.dot(np.dot(self.W, inside), self.T.T)  # when use summation
        grad = self.lamb * self.H - 2 / np.prod(self.A.shape) * np.dot(np.dot(self.W, inside), self.T.T)
        self.H = self.H - self.lr * grad
        self.H[self.H < self.lower_control] = self.lower_control
    
    def compute_loss(self, iteration):
        inside = self.A - np.dot(np.dot(self.W.T, self.H), self.T)
        # loss0 = np.sum(np.square(inside))  # when use summation
        loss0 = np.mean(np.square(inside))
        loss1 = self.lamb * np.sum(np.square(self.W)) / 2
        loss2 = self.lamb * np.sum(np.square(self.H)) / 2
        loss_sum = loss2 + loss1 + loss0
        self.losses.append([iteration, loss_sum, loss0, loss1, loss2])
    
    def get_embeddings(self):
        return np.concatenate((self.W.T, np.dot(self.H,self.T).T), axis=1)
    
    def iterations(self, n):
        for i in range(n):
            self.update_W()
            self.update_H()
            self.compute_loss(i)
            print('iteration {},\tsum loss {},\tmain loss {},\tloss1 {},\tloss2 {}'.format(*self.losses[-1]))
        return self.get_embeddings()
    
    #  ========= torch version =====分割线======用两种方式来，上面的是手动求导，下面的用pytorch===========================
    def re_initial_weights(self):
        W = torch.randn(self.k, self.A.shape[0], dtype=torch.float, requires_grad=True)
        H = torch.randn(self.k, self.T.shape[0], dtype=torch.float, requires_grad=True)
        self.Wt = W
        self.Ht = H
    
    def compute_loss_tensor(self):
        inside = torch.Tensor(self.A) - torch.mm(torch.mm(torch.transpose(self.Wt, 0, 1), self.Ht),torch.Tensor(self.T))
        # loss0 = torch.sum(inside * inside)  # when use summation
        loss0 = torch.mean(inside * inside)
        loss1 = self.lamb * torch.sum(self.Wt * self.Wt)
        loss2 = self.lamb * torch.sum(self.Ht * self.Ht)
        loss_fc = loss0 + loss1 + loss2
        self.losses.append([loss_fc, loss0, loss1, loss2])
        return loss_fc
    
    def update_W_tensor(self):
        grad = torch.autograd.grad(self.compute_loss_tensor(), self.Wt, retain_graph=True)
        self.Wt = self.Wt - self.lr * grad[0]
        self.Wt[self.Wt < self.lower_control] = self.lower_control
    
    def update_H_tensor(self):
        grad = torch.autograd.grad(self.compute_loss_tensor(), self.Ht, retain_graph=True)
        self.Ht = self.Ht - self.lr * grad[0]
        self.Ht[self.Ht < self.lower_control] = self.lower_control
    
    def get_embeddings_tensor(self):
        print(self.Wt.shape, self.Ht.shape, self.T.shape)
        torch.mm(self.Ht, torch.Tensor(self.T))
        return torch.cat([torch.transpose(self.Wt, 0, 1), torch.transpose(torch.mm(self.Ht, torch.Tensor(self.T)), 0, 1)], dim=1)
    
    def iterations_tensor(self, n):
        self.re_initial_weights()
        for i in range(n):
            self.update_W_tensor()
            self.update_H_tensor()
            self.compute_loss_tensor()
            print('iteration_tensor {},\tsum loss {},\tmain loss {},\tloss1 {},\tloss2 {}'.format(i, *self.losses[-1]))
        return self.get_embeddings_tensor().detach().numpy()


if __name__ == '__main__':
    edge_path = 'data/cora/cora.cites'
    node_path = 'data/cora/cora.content'
    data_process = DataProcess(node_path, edge_path)
    A, T, target = data_process.return_all()
    tadw = TadwModel(A, T)
    embeddings = tadw.iterations(30)
    print(embeddings.shape)
    print(target)
    # =========== classification using SVM ==============
    train_x, test_x, train_y, test_y = model_selection.train_test_split(embeddings, target, test_size=0.2, shuffle=True)
    clf = svm.LinearSVC(C=5.0)
    clf.fit(train_x, train_y)
    predict_y = clf.predict(test_x)
    print(predict_y)
    auc = metrics.accuracy_score(test_y, predict_y)
    print(auc)
