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
    model = pickle.load(open('bestModel.pkl', 'rb'))
    lr = model.lr
    hidden = model.hidden
    l2 = model.l2
    filedir = 'hidden_'+str(hidden)+'_lr_'+'%.2E' % Decimal(lr)\
                            + '_l2_'+'%.1E' % Decimal(l2)
    rec_name = filedir + '_recorder.pkl'

    recorder = pickle.load(open(filedir+'/'+rec_name, 'rb'))

    _, _, test_images, test_labels = mnist()

    plt.line1 = plot(recorder['test_acc_rec'], label='test')
    plt.line2 = plot(recorder['train_acc_rec'], label='train')
    plt.legend()