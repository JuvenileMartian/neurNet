import os
import gzip
import numpy as np
import pickle
from time import time
from matplotlib import pylab as plt
from decimal import Decimal

from models import twoLayerNetwork
from utils import showp, mnist, sigmoid, softmax, ceLoss_onehot

class Trainer:
    def __init__(self, model:twoLayerNetwork, train_imgs, train_labels, test_images, test_labels, valid_images, valid_labels):
        self.model = model
        self.train_imgs = train_imgs
        self.train_labels = train_labels
        self.test_imgs = test_images
        self.test_labels = test_labels
        self.valid_imgs = valid_images
        self.valid_labels = valid_labels
        self.N = train_imgs.shape[0]

    def calc_loss_backward(self, arr, label):
        # arr: 784
        # label: 10 * 1
        arr_predict = self.model.forward(arr)
        loss = ceLoss_onehot(arr_predict, label)

        dloss_dout = -(label-arr_predict)                           #由交叉熵求梯度
        dout_dout0 = self.model.output_arr * (1-self.model.output_arr)
        dloss_dout0 = (dloss_dout * dout_dout0).reshape(-1,1)       # 10 * 1
        dloss_dm2 = dloss_dout0  * self.model.h_arr                 # 10 * h

        dout0_dh = self.model.mat2                      # K * h
        dloss_dh = (dloss_dout0 * dout0_dh).sum(0)      # h * 1
        dh_dh0 = self.model.h_arr * (1-self.model.h_arr)          # h * 1
        dloss_dh0 = (dloss_dh * dh_dh0).reshape(-1,1)   # h * 1
        dloss_dm1 = dloss_dh0 * arr                     # h * 784

        self.model.update_gradient(dloss_dm1, dloss_dm2)
        return loss

    def step(self):
        self.model.mat1 -= self.model.lr * self.model.mat1_gd
        self.model.mat2 -= self.model.lr * self.model.mat2_gd

    def train_sgd(self, epoches=100, iterations=None):
        train_acc_rec = []
        train_loss_rec = []
        test_acc_rec = []
        test_loss_rec = []
        valid_acc_rec = []
        valid_loss_rec = []
        recorder = {'train_acc_rec':train_acc_rec, 'train_loss_rec':train_loss_rec,
                        'test_acc_rec':test_acc_rec, 'test_loss_rec':test_loss_rec,
                        'valid_acc_rec':valid_acc_rec, 'valid_loss_rec':valid_loss_rec,}
        for epoch in range(epoches):
            start = time()
            idx = np.random.permutation(self.N)                
            for iter, i in enumerate(list(idx)):
                arr, label = self.train_imgs[i], self.train_labels[i]
                self.calc_loss_backward(arr, label)
                self.step()
            train_acc, train_loss, test_acc, test_loss, valid_acc, valid_loss = self.accuracy()
            train_acc_rec.append(train_acc)
            train_loss_rec.append(train_loss)
            test_acc_rec.append(test_acc)
            test_loss_rec.append(test_loss)
            valid_acc_rec.append(test_acc)
            valid_loss_rec.append(test_loss)
            train_acc = round(train_acc, 4)
            train_loss = round(train_loss, 2)
            test_acc = round(test_acc, 4)
            test_loss = round(test_loss, 2)
            valid_acc = round(valid_acc, 4)
            valid_loss = round(valid_loss, 2)

            print(epoch+1, 'epoches | Accuracy:', train_acc, 'Loss:', train_loss,
                    '| Valid Accuracy:', valid_acc, 'Loss:', valid_loss,
                    '| Test Accuracy:', test_acc, 'Loss:', test_loss,
                    '| Epoch Time:', round(time()-start,2))
            try:
                f = open('./'+self.model.filename+'/'+self.model.filename+'.pkl', 'wb')
                pickle.dump(self.model, f)
                f.close()
                f = open('./'+self.model.filename+'/'+self.model.filename+'_recorder.pkl', 'wb')
                pickle.dump(recorder, f)
                f.close()
            except:
                pass
        return valid_loss

    def accuracy(self):
        train_acc = 0
        train_predict = self.model.forward(self.train_imgs.T)
        train_acc = np.mean([1 if self.train_labels[i,j] else 0 for (i,j) in enumerate(list(train_predict.argmax(0)))])
        train_loss = -(np.log(train_predict.T) * self.train_labels).sum()/self.N      
        test_acc = 0
        test_predict = self.model.forward(self.test_imgs.T)
        test_acc = np.mean([1 if self.test_labels[i,j] else 0 for (i,j) in enumerate(list(test_predict.argmax(0)))])
        test_loss = -(np.log(test_predict.T) * self.test_labels).sum()/test_predict.shape[1]
        valid_acc = 0
        valid_predict = self.model.forward(self.valid_imgs.T)
        valid_acc = np.mean([1 if self.valid_labels[i,j] else 0 for (i,j) in enumerate(list(valid_predict.argmax(0)))])
        valid_loss = -(np.log(valid_predict.T) * self.valid_labels).sum()/valid_predict.shape[1] 
        return train_acc, train_loss, test_acc, test_loss, valid_acc, valid_loss