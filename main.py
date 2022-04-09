import os
import gzip
import numpy as np
import pickle
from time import time
from matplotlib import pylab as plt
from decimal import Decimal

from utils import showp, mnist, sigmoid, softmax, ceLoss_onehot
from models import initOrFetchModel, twoLayerNetwork
from sgd import Trainer

if __name__ == '__main__':
    st = time()
    hiddens = [5, 10, 20, 30, 40, 50]
    lrs = [0.001, 0.01, 0.05, 0.1]
    l2s = [0.00001, 0.0001, 0.0005, 0.001, 0.005]
    train_images, train_labels, test_images, test_labels = mnist()
    N = train_images.shape[0]
    train_images, valid_images = train_images[:int(N*0.8)], train_images[int(N*0.8):]
    train_labels, valid_labels = train_labels[:int(N*0.8)], train_labels[int(N*0.8):]
    best_valid_loss = float('inf')
    best_hyparam = {'hidden':None, 'lr':None, 'l2':None}

    for hidden in hiddens:
        for lr in lrs:
            for l2 in l2s:
                model = initOrFetchModel(hidden=hidden, lr=lr, l2=l2)
                trainer = Trainer(model, train_images, train_labels, test_images, test_labels, valid_images, valid_labels)
                valid_loss = trainer.train_sgd(epoches=100)
                if valid_loss < best_valid_loss:
                    print('Updating Best Hyperparameters')
                    best_valid_loss = valid_loss
                    try:
                        f = open('bestModel.pkl', 'wb')
                        pickle.dump(model, f)
                        f.close()
                    except:
                        pass
                    best_hyparam['hidden'] = hidden
                    best_hyparam['lr'] = lr
                    best_hyparam['l2'] = l2

    print(best_hyparam)
    print('Total Time:', time()-st)