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


# Initialize the DataSets for MNIST tasks
def initDataSetsClasses():
    global dataSetTrain
    global dataSetTest

    print(FLAGS.train_classes, FLAGS.test_classes)
    # Variable to read out the labels & data of the DataSet Object.
    mnistData = read_data_sets('./',
                               one_hot=True)
    # MNIST labels & data for training.
    mnistLabelsTrain = mnistData.train.labels
    mnistDataTrain = mnistData.train.images

    # MNIST labels & data for testing.
    mnistLabelsTest = mnistData.test.labels
    mnistDataTest = mnistData.test.images
    print("LABELS", mnistLabelsTest.shape, mnistLabelsTrain.shape)

    if FLAGS.permuteTrain != -1:
        # training dataset
        np.random.seed(FLAGS.permuteTrain)
        permTr = np.random.permutation(mnistDataTrain.shape[1])
        mnistDataTrainPerm = mnistDataTrain[:, permTr]
        mnistDataTrain = mnistDataTrainPerm;
        # dataSetTrain = DataSet(255. * dataSetTrainPerm,
        #                       mnistLabelsTrain, reshape=False)
    if FLAGS.permuteTest != -1:
        # testing dataset
        np.random.seed(FLAGS.permuteTest)
        permTs = np.random.permutation(mnistDataTest.shape[1])
        mnistDataTestPerm = mnistDataTest[:, permTs]
        # dataSetTest = DataSet(255. * dataSetTestPerm,
        #                      mnistLabelsTest, reshape=False)
        mnistDataTest = mnistDataTestPerm;

    if True:
        # args = parser.parse_args()
        print(FLAGS.train_classes, FLAGS.test_classes)
        if FLAGS.train_classes[0:]:
            labels_to_train = [int(i) for i in FLAGS.train_classes[0:]]

        if FLAGS.test_classes[0:]:
            labels_to_test = [int(i) for i in FLAGS.test_classes[0:]]

        # Filtered labels & data for training and testing.
        labels_train_classes = np.array([mnistLabelsTrain[i].argmax() for i in xrange(0,
                                                                                      mnistLabelsTrain.shape[0]) if
                                         mnistLabelsTrain[i].argmax()
                                         in labels_to_train], dtype=np.uint8)
        data_train_classes = np.array([mnistDataTrain[i, :] for i in xrange(0,
                                                                            mnistLabelsTrain.shape[0]) if
                                       mnistLabelsTrain[i].argmax()
                                       in labels_to_train], dtype=np.float32)

        labels_test_classes = np.array([mnistLabelsTest[i].argmax() for i in xrange(0,
                                                                                    mnistLabelsTest.shape[0]) if
                                        mnistLabelsTest[i].argmax()
                                        in labels_to_test], dtype=np.uint8)
        data_test_classes = np.array([mnistDataTest[i, :] for i in xrange(0,
                                                                          mnistDataTest.shape[0]) if
                                      mnistLabelsTest[i].argmax()
                                      in labels_to_test], dtype=np.float32)

        labelsTrainOnehot = dense_to_one_hot(labels_train_classes, 10)
        labelsTestOnehot = dense_to_one_hot(labels_test_classes, 10)

        dataSetTrain = DataSet(255. * data_train_classes,
                               labelsTrainOnehot, reshape=False)
        dataSetTest = DataSet(255. * data_test_classes,
                              labelsTestOnehot, reshape=False)


def train():

    # Start an Interactive session
    sess = tf.InteractiveSession()

    classifier = Classifier(num_class=10, num_features=784, fc_hidden_units=[FLAGS.hidden1 , FLAGS.hidden2], apply_dropout=True)

    print('\nTraining on DataSet started...')
    print('____________________________________________________________')
    print(time.strftime('%X %x %Z'))

    print("Total updates: %s "%((55000 // FLAGS.batch_size) * FLAGS.epochs))
    Classifier.train_mod(classifier, sess=sess, model_name=FLAGS.save_model, model_init_name=FLAGS.load_model,
                     dataset = dataSetTrain,
                     num_updates=(55000 // FLAGS.batch_size) * FLAGS.epochs,
                     dataset_lagged = [0],
                     mini_batch_size=FLAGS.batch_size,
                     log_frequency=LOG_FREQUENCY,
                     fisher_multiplier=1.0 / FLAGS.learning_rate,
                     learning_rate=FLAGS.learning_rate,
                     testing_data_set=dataSetTest,
                     plot_file=FLAGS.plot_file,
                     start_at_step = FLAGS.start_at_step
                     )

    x = Classifier.test(classifier, sess=sess,
                                    model_name=FLAGS.save_model,
                                    batch_xs=dataSetTest.images,
                                    batch_ys=dataSetTest.labels)
    print (x)


def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
        tf.gfile.MakeDirs(FLAGS.log_dir)
    initDataSetsClasses()
    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_classes', type=int, nargs='*',
                        help="Take only the specified Train classes from MNIST DataSet")
    parser.add_argument('--test_classes', type=int, nargs='*',
                        help="Take the specified Test classes from MNIST DataSet")

    parser.add_argument('--max_steps', type=int, default=2000,
                        help='Number of steps to run trainer for given data set.')

    parser.add_argument('--dropout_hidden', type=float, default=0.5,
                        help='Keep probability for dropout hidden.')
    parser.add_argument('--dropout_input', type=float, default=0.8,
                        help='Keep probability for dropout input.')

    parser.add_argument('--permuteTrain', type=int, default=-1,
                        help='Provide random seed for permutation train.')
    parser.add_argument('--permuteTest', type=int, default=-1,
                        help='Provide random seed for permutation test.')

    parser.add_argument('--hidden1', type=int, default=128,
                        help='Number of hidden units in layer 1')
    parser.add_argument('--hidden2', type=int, default=32,
                        help='Number of hidden units in layer 2')
    parser.add_argument('--hidden3', type=int, default=-1,
                        help='Number of hidden units in layer 3')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Size of mini-batches we feed from dataSet.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Initial learning rate')

    parser.add_argument('--load_model', type=str,
                        help='Load previously saved model. Leave empty if no model exists.')
    parser.add_argument('--save_model', type=str,
                        help='Provide path to save model.')
    parser.add_argument('--test_frequency', type=int, default='50',
                        help='Frequency after which a test cycle runs.')
    parser.add_argument('--start_at_step', type=int, default='0',
                        help='Global step should start here, and continue for the specified number of iterations')
    parser.add_argument('--epochs', type=int, default=1,
                        help='the number of training epochs per task')
    parser.add_argument('--plot_file', type=str,
                        default='ewc_with_options.csv',
                        help='Filename for csv file to plot. Give .csv extension after file name.')
    parser.add_argument('--data_dir', type=str,
                        default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    parser.add_argument('--log_dir', type=str,
                        default='/tmp/tensorflow/mnist/logs',
                        help='Summaries log directory')
    parser.add_argument('--checkpoints_dir', type=str,
                        default='./checkpoints/',
                        help='Checkpoints log directory')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)