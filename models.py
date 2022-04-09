import os
import gzip
import numpy as np
import pickle
from time import time
from matplotlib import pylab as plt
from decimal import Decimal

from utils import showp, mnist, sigmoid, softmax, ceLoss_onehot

INPUT_SIZE = 784  # 28*28
OUTPUT_SIZE = 10

def initOrFetchModel(hidden=300, lr=0.01, l2=0.0005):
    # 如果存在对应目录，则读取目录中的模型 （包含学习率/正则化率超参）
    # 否则创建目录，并初始化新的模型

    filename = 'hidden_'+str(hidden)+'_lr_'+'%.2E' % Decimal(lr)\
                            + '_l2_'+'%.1E' % Decimal(l2)
    targetPath = './'+filename+'/'
    
    if not os.path.exists(targetPath):
        print("Created Model with Hyperparameters: Hidden Size =", hidden, '-- lr =', lr, '-- gamma =', l2)
        os.mkdir(targetPath)
        return twoLayerNetwork(hidden, lr, l2)
    else:
        print("Loaded Existing Model with Hyperparameters: Hidden Size =", hidden, '-- lr =', lr, '-- gamma =', l2)
        f = open(targetPath + filename + '.pkl', 'rb')
        model = pickle.load(f)
        f.close()
        return model

class twoLayerNetwork:
    def __init__(self, hidden=300, lr=0.01, l2=0.0001):
        self.hidden=hidden
        self.lr = lr
        self.l2 = l2

        self.filename = 'hidden_'+str(hidden)+'_lr_'+'%.2E' % Decimal(lr)\
                            + '_l2_'+'%.1E' % Decimal(l2)
        targetPath = './'+self.filename+'/'

        self.mat1 = np.random.randn(INPUT_SIZE*self.hidden)\
                        .reshape(self.hidden, INPUT_SIZE)

        self.mat1_gd = np.zeros(INPUT_SIZE*self.hidden)\
                        .reshape(self.hidden, INPUT_SIZE)

        self.mat2 = np.random.randn(OUTPUT_SIZE*self.hidden)\
                        .reshape(OUTPUT_SIZE, self.hidden)

        self.mat2_gd = np.zeros(OUTPUT_SIZE*self.hidden)\
                        .reshape(OUTPUT_SIZE, self.hidden)

    def forward(self, arr):
        self.h_arr0 = np.dot(self.mat1, arr)
        self.h_arr = sigmoid(self.h_arr0)

        self.output_arr0 = np.dot(self.mat2, self.h_arr)
        self.output_arr = sigmoid(self.output_arr0)
        return softmax(self.output_arr)

    def update_gradient(self, dm1, dm2):
        self.mat1_gd = dm1 + self.l2 * self.mat1
        self.mat2_gd = dm2 + self.l2 * self.mat2