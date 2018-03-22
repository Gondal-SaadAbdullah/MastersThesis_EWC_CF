import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
from utilities import *

import utils
from model import imm

class fakeFlags(object):
  def __init__(self):
    self.train_classes = [1,2,3,4,5,6,7,8,9] ;
    self.test_classes = [1,2,3,4,5,6,7,8,9] ;
    self.test2_classes = [0] ;
    self.test3_classes = [0,1,2,3,4,5,6,7,8,9] ;
    self.permuteTrain = -1;
    self.permuteTest = -1;
    self.permuteTest2 = -1;
    self.permuteTest3 = -1;

def SplitPackage(train_classes, train2_classes, test_classes, test2_classes, test3_classes):
# alternate data generation
    tmpObj = fakeFlags() ;
    tmpObj.train_classes = train_classes ;
    tmpObj.test_classes =  test_classes;
    tmpObj.test2_classes = test2_classes ;
    tmpObj.test3_classes =  test3_classes;
    tmpObj.permuteTrain = -1;
    tmpObj.permuteTest = -1;
    tmpObj.permuteTest2 = -1;
    tmpObj.permuteTest3 = -1;
    tr,tst,tst2,tst3 = initDataSetsClasses(tmpObj)
    tmpObj.train_classes = train2_classes ;
    tr2,tst,tst2,tst3 = initDataSetsClasses(tmpObj)
    x = [tr.images,tr2.images] ;
    y = [tr.labels, tr2.labels] ;
    x_ = [tst.images, tst2.images]
    y_ = [tst.labels, tst2.labels] ;
    xyc_info = [[tr.images, tr.labels, "train1"], [tr2.images,tr2.labels, "train2"], [tst.images,tst.labels,"test1"],[tst2.images, tst2.labels,"test2"],[tst3.images,tst3.labels,"test3"]] ;
    return x,y,x_,y_,xyc_info ;


def XycPackage():
    """
    Load Dataset and set package.
    """
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    noOfTask = 3
    x = []
    x_ = []
    y = []
    y_ = []
    xyc_info = []

    x.append(np.concatenate((mnist.train.images,mnist.validation.images)))
    y.append(np.concatenate((mnist.train.labels,mnist.validation.labels)))
    x_.append(mnist.test.images)
    y_.append(mnist.test.labels)
    xyc_info.append([x[0], y[0], 'train-idx1'])

    for i in range(1, noOfTask):
        idx = np.arange(784)                 # indices of shuffling image
        np.random.shuffle(idx)

        x.append(x[0].copy())
        x_.append(x_[0].copy())
        y.append(y[0].copy())
        y_.append(y_[0].copy())

        x[i] = x[i][:,idx]           # applying to shuffle
        x_[i] = x_[i][:,idx]

        xyc_info.append([x[i], y[i], 'train-idx%d' % (i+1)])

    for i in range(noOfTask):
        xyc_info.append([x_[i], y_[i], 'test-idx%d' % (i+1)])

    return x, y, x_, y_, xyc_info
