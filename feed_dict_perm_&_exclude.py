from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from classifiers import Classifier

import tensorflow as tf
import numpy as np
import math

import argparse
import sys
import time

from sys import byteorder
from numpy import size

from tensorflow.python.framework import dtypes

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets.mnist import dense_to_one_hot

FLAGS = None

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

LOG_FREQUENCY = 50


# Initialize the DataSets for the permuted MNIST task
def initDataSetsPermutation():
    # Variable to read out the labels & data of the DataSet Object.
    mnistData = read_data_sets('/tmp/tensorflow/mnist/input_data',
                               one_hot=True)

    # MNIST labels & data for training.
    mnistLabelsTrain = mnistData.train.labels
    mnistDataTrain = mnistData.train.images

    # MNIST labels & data for testing.
    mnistLabelsTest = mnistData.test.labels
    mnistDataTest = mnistData.test.images

    mnistPermutationTrain = np.array(mnistDataTrain, dtype=np.float32)
    mnistPermutationTest = np.array(mnistDataTest, dtype=np.float32)

    # Concatenate both arrays to make sure the shuffling is consistent over
    # the training and testing sets, split them afterwards and create objects
    mnistPermutation = np.concatenate([mnistDataTrain, mnistDataTest])
    np.random.shuffle(mnistPermutation.T)
    mnistPermutationTrain, mnistPermutationTest = np.split(mnistPermutation, [
        mnistDataTrain.shape[0]])

    global dataSetOneTrain
    dataSetOneTrain = DataSet(255. * mnistDataTrain,
                              mnistLabelsTrain, reshape=False)
    global dataSetOneTest
    dataSetOneTest = DataSet(255. * mnistDataTest,
                             mnistLabelsTest, reshape=False)

    global dataSetTwoTrain
    dataSetTwoTrain = DataSet(255. * mnistPermutationTrain,
                              mnistLabelsTrain, reshape=False)
    global dataSetTwoTest
    dataSetTwoTest = DataSet(255. * mnistPermutationTest,
                             mnistLabelsTest, reshape=False)


# Initialize the DataSets for the partitioned digits task
def initDataSetsExcludedDigits():
    args = parser.parse_args()
    if args.exclude[0:]:
        labelsToErase = [int(i) for i in args.exclude[0:]]

    # Variable to read out the labels & data of the DataSet Object.
    mnistData = read_data_sets('/tmp/tensorflow/mnist/input_data',
                               one_hot=False)

    # MNIST labels & data for training.
    mnistLabelsTrain = mnistData.train.labels
    mnistDataTrain = mnistData.train.images

    # MNIST labels & data for testing.
    mnistLabelsTest = mnistData.test.labels
    mnistDataTest = mnistData.test.images

    # Filtered labels & data for training (DataSetOne).
    labelsExcludedTrain = np.array([mnistLabelsTrain[i] for i in xrange(0,
                                                                        mnistLabelsTrain.shape[0]) if
                                    mnistLabelsTrain[i]
                                    in labelsToErase], dtype=np.uint8)

    dataExcludedTrain = np.array([mnistDataTrain[i, :] for i in xrange(0,
                                                                       mnistLabelsTrain.shape[0]) if mnistLabelsTrain[i]
                                  in labelsToErase], dtype=np.float32)

    # Filtered labels & data for testing (DataSetOne).
    labelsExcludedTest = np.array([mnistLabelsTest[i] for i in xrange(0,
                                                                      mnistLabelsTest.shape[0]) if mnistLabelsTest[i]
                                   in labelsToErase], dtype=np.uint8)

    dataExcludedTest = np.array([mnistDataTest[i, :] for i in xrange(0,
                                                                     mnistLabelsTest.shape[0]) if mnistLabelsTest[i]
                                 in labelsToErase], dtype=np.float32)

    # Filtered labels & data for training (DataSetTwo).
    labelsKeepedTrain = np.array([mnistLabelsTrain[i] for i in xrange(0,
                                                                      mnistLabelsTrain.shape[0]) if mnistLabelsTrain[i]
                                  not in labelsToErase], dtype=np.uint8)

    dataKeepedTrain = np.array([mnistDataTrain[i, :] for i in xrange(0,
                                                                     mnistLabelsTrain.shape[0]) if mnistLabelsTrain[i]
                                not in labelsToErase], dtype=np.float32)

    # Filtered labels & data for testing (DataSetTwo).
    labelsKeepedTest = np.array([mnistLabelsTest[i] for i in xrange(0,
                                                                    mnistLabelsTest.shape[0]) if mnistLabelsTest[i]
                                 not in labelsToErase], dtype=np.uint8)

    dataKeepedTest = np.array([mnistDataTest[i, :] for i in xrange(0,
                                                                   mnistLabelsTest.shape[0]) if mnistLabelsTest[i]
                               not in labelsToErase], dtype=np.float32)

    # Transform labels to one-hot vectors
    labelsKeepedTrainOnehot = dense_to_one_hot(labelsKeepedTrain, 10)
    labelsExcludedTrainOnehot = dense_to_one_hot(labelsExcludedTrain, 10)

    labelsKeepedTestOnehot = dense_to_one_hot(labelsKeepedTest, 10)
    labelsExcludedTestOnehot = dense_to_one_hot(labelsExcludedTest, 10)

    labelsAllTrainOnehot = dense_to_one_hot(mnistLabelsTrain, 10)
    labelsAllTestOnehot = dense_to_one_hot(mnistLabelsTest, 10)

    # Create DataSets (1: kept digits, 2: excluded digits, all: MNIST digits)
    global dataSetOneTrain
    dataSetOneTrain = DataSet(255. * dataKeepedTrain,
                              labelsKeepedTrainOnehot, reshape=False)
    global dataSetOneTest
    dataSetOneTest = DataSet(255. * dataKeepedTest,
                             labelsKeepedTestOnehot, reshape=False)

    global dataSetTwoTrain
    dataSetTwoTrain = DataSet(255. * dataExcludedTrain,
                              labelsExcludedTrainOnehot, reshape=False)
    global dataSetTwoTest
    dataSetTwoTest = DataSet(255. * dataExcludedTest,
                             labelsExcludedTestOnehot, reshape=False)

    global dataSetAllTrain
    dataSetAllTrain = DataSet(255. * mnistDataTrain,
                              labelsAllTrainOnehot, reshape=False)
    global dataSetAllTest
    dataSetAllTest = DataSet(255. * mnistDataTest,
                             labelsAllTestOnehot, reshape=False)


def train():

    # Start an Interactive session
    sess = tf.InteractiveSession()


    # Initialize all global variables
    tf.global_variables_initializer().run()

    print('Fully Connected Neural Network with two hidden layers')
    print('Files being logged to... %s' % (FLAGS.log_dir,))
    print('\nHyperparameters:')
    print('____________________________________________________________')
    print('\nTraining steps for first training (data set 1): %s'
          % (FLAGS.max_steps_ds1,))
    print('Training steps for second training (data set 2): %s'
          % (FLAGS.max_steps_ds2,))
    print('Batch size for data set 1: %s' % (FLAGS.batch_size_1,))
    print('Batch size for data set 2: %s' % (FLAGS.batch_size_2,))
    print('Number of hidden units for layer 1: %s' % (FLAGS.hidden1,))
    print('Number of hidden units for layer 2: %s' % (FLAGS.hidden2,))
    print('Keep probability on input units: %s' % (FLAGS.dropout_input,))
    print('Keep probability on hidden units: %s' % (FLAGS.dropout_hidden,))
    print('Learning rate: %s' % (FLAGS.learning_rate,))
    print('\nInformation about the data sets:')
    print('____________________________________________________________')
    if FLAGS.exclude:
        print('\nExcluded digits: ')
        print('DataSetOne (train) contains: %s images.'
              % (dataSetOneTrain.labels.shape[0],))
        print('DataSetOne (test) contains: %s images.\n'
              % (dataSetOneTest.labels.shape[0],))
        print('DataSetTwo (train) contains: %s images.'
              % (dataSetTwoTrain.labels.shape[0],))
        print('DataSetTwo (test) contains: %s images.\n'
              % (dataSetTwoTest.labels.shape[0],))
        print('Original MNIST data-set (train) contains: %s images.'
              % (dataSetAllTrain.labels.shape[0],))
        print('Original MNIST data-set (test) contains: %s images.'
              % (dataSetAllTest.labels.shape[0],))
    if FLAGS.permutation:
        print('\nPermuted digits: ')
        print('DataSetOne (train) contains: %s images.'
              % (dataSetOneTrain.labels.shape[0],))
        print('DataSetOne (test) contains: %s images.\n'
              % (dataSetOneTest.labels.shape[0],))
        print('DataSetTwo (train) contains: %s images.'
              % (dataSetTwoTrain.labels.shape[0],))
        print('DataSetTwo (test) contains: %s images.\n'
              % (dataSetTwoTest.labels.shape[0],))

    classifier = Classifier(num_class=10, num_features=784, fc_hidden_units=[FLAGS.hidden1 , FLAGS.hidden2], apply_dropout=True)

    print('\nTraining on DataSetOne...')
    print('____________________________________________________________')
    print(time.strftime('%X %x %Z'))

    Classifier.train_mod(classifier, sess=sess, model_name='dataset1', model_init_name="",
                     dataset = dataSetOneTrain,
                     num_updates=(55000 // FLAGS.batch_size_1) * FLAGS.epochs,
                     dataset_lagged = [0],
                     mini_batch_size=FLAGS.batch_size_1,
                     log_frequency=LOG_FREQUENCY,
                     fisher_multiplier=1.0 / FLAGS.learning_rate,
                     learning_rate=FLAGS.learning_rate,
                     testing_data_set=dataSetOneTest
                     )

    x = Classifier.test(classifier, sess=sess,
                                    model_name='dataset1',
                                    batch_xs=dataSetOneTest.images,
                                    batch_ys=dataSetOneTest.labels)
    print (x)


    print('\nTraining on DataSetTwo...')
    print('____________________________________________________________')
    print(time.strftime('%X %x %Z'))
    # Start training on dataSetTwo
    Classifier.train_mod(classifier, sess=sess, model_name='dataset2', model_init_name="dataset1", dataset=dataSetTwoTrain,
                     num_updates=(55000 // FLAGS.batch_size_2) * FLAGS.epochs,
                     dataset_lagged = [0],
                     mini_batch_size=FLAGS.batch_size_2,
                     log_frequency=LOG_FREQUENCY,
                     fisher_multiplier=1.0 / FLAGS.learning_rate,
                     learning_rate=FLAGS.learning_rate,
                     testing_data_set=dataSetOneTest
                     )
    y = Classifier.test(classifier, sess=sess,
                               model_name='dataset2',
                               batch_xs=dataSetOneTest.images,
                               batch_ys=dataSetOneTest.labels)

    print(y)

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
        tf.gfile.MakeDirs(FLAGS.log_dir)
    if FLAGS.permutation:
        initDataSetsPermutation()
    if FLAGS.exclude:
        initDataSetsExcludedDigits()
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exclude', type=int, nargs='*',
                        help="Exclude specified classes from the MNIST DataSet")
    parser.add_argument('--permutation', action='store_true',
                        help='Use a random consistent permutation of MNIST.')
    parser.add_argument('--max_steps_ds1', type=int, default=2000,
                        help='Number of steps to run trainer for data set 1.')
    parser.add_argument('--max_steps_ds2', type=int, default=100,
                        help='Number of steps to run trainer for data set 2.')
    parser.add_argument('--hidden1', type=int, default=128,
                        help='Number of hidden units in layer 1')
    parser.add_argument('--hidden2', type=int, default=32,
                        help='Number of hidden units in layer 2')
    parser.add_argument('--batch_size_1', type=int, default=100,
                        help='Size of mini-batches we feed from dataSetOne.')
    parser.add_argument('--batch_size_2', type=int, default=100,
                        help='Size of mini-batches we feed from dataSetTwo.')
    parser.add_argument('--batch_size_all', type=int, default=100,
                        help='Size of mini-batches we feed from dataSetAll.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--dropout_hidden', type=float, default=0.5,
                        help='Keep probability for dropout on hidden units.')
    parser.add_argument('--dropout_input', type=float, default=0.8,
                        help='Keep probability for dropout on input units.')
    parser.add_argument('--epochs', type=int, default=1,
                        help='the number of training epochs per task')
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str,
                        default='/tmp/tensorflow/mnist/logs',
                        help='Summaries log directory')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)