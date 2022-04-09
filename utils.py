import os
import gzip
import numpy as np
from matplotlib import pylab as plt
from decimal import Decimal

def showp(arr):
    # show pixels
    plt.imshow(arr,cmap='gray')

def mnist():
    # Return (train_images, train_labels, test_images, test_labels).
    path = './mnist/'
    files = ['train-images-idx3-ubyte.gz',
             'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz',
             't10k-labels-idx1-ubyte.gz']    

    # Create path if it doesn't exist
    os.makedirs(path, exist_ok=True)

    def _images(path):
        """Return images loaded locally."""
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            pixels = np.frombuffer(f.read(), 'B', offset=16)
        return pixels.reshape(-1, 784).astype('float32') / 255

    def _labels(path):
        """Return labels loaded locally."""
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            integer_labels = np.frombuffer(f.read(), 'B', offset=8)

        def _onehot(integer_labels):
            """Return matrix whose rows are onehot encodings of integers."""
            n_rows = len(integer_labels)
            n_cols = integer_labels.max() + 1
            onehot = np.zeros((n_rows, n_cols), dtype='uint8')
            onehot[np.arange(n_rows), integer_labels] = 1
            return onehot

        return _onehot(integer_labels)

    train_images = _images(os.path.join(path, files[0]))
    train_labels = _labels(os.path.join(path, files[1]))
    test_images = _images(os.path.join(path, files[2]))
    test_labels = _labels(os.path.join(path, files[3]))

    return train_images, train_labels, test_images, test_labels

def sigmoid(arr):
    return 1/(1+np.exp(-arr))

def softmax(arr):
    return np.exp(arr) / np.sum(np.exp(arr))

def ceLoss_onehot(predict_arr, label):
    # predict_arr: 经过softmax后所得的各类概率估计值
    # label: one-hot 的金标准向量
    # 计算 Cross Entropy Loss
    N = label.shape[0]
    return -np.sum(np.log(predict_arr)*label)