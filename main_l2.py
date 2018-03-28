# mode-imm and mean-imm with weight transfer
import time
import argparse
import numpy as np
import tensorflow as tf

import preprocess.mnist as preprocess
import utils
from model import model_utils
from model import imm
from utilities import initDataSetsClasses ;


print("==> parsing input arguments")
flags = tf.app.flags

## Data input settings
flags.DEFINE_boolean("mean_imm", True, "include Mean-IMM")
flags.DEFINE_boolean("mode_imm", True, "include Mode-IMM")

## Model Hyperparameter
flags.DEFINE_float("alpha", -1, "alpha(K) of Mean & Mode IMM (cf. equation (3)~(8) in the article)")
flags.DEFINE_float("regularizer", -1, "L2 Regularization parameter")


## Training Hyperparameter
flags.DEFINE_float("epoch", -1, "the number of training epoch")
flags.DEFINE_string("optimizer", 'SGD', "the method name of optimization. (SGD|Adam|Momentum)")
flags.DEFINE_float("learning_rate", -1, "learning rate of optimizer")
flags.DEFINE_float("learning_rate2", -1, "learning rate of optimizer")
flags.DEFINE_integer("batch_size", 50, "mini batch size")
flags.DEFINE_integer("tasks", 2, "number of tasks")
flags.DEFINE_string("db", 'mnist.pkl.gz', "database")
flags.DEFINE_string("train_classes", '0 1 2 3 4', "trainclasses")
flags.DEFINE_string("train2_classes", '5 6 7 8 9', "train2classes")
flags.DEFINE_string("test_classes", '0 1 2 3 4', "testclasses")
flags.DEFINE_string("test2_classes", '5 6 7 8 9', "test2classes")
flags.DEFINE_string("test3_classes", '0 1 2 3 4 5 6 7 8 9', "test3classes")
flags.DEFINE_integer("hidden1", 200, "neurons in hl1")
flags.DEFINE_integer("hidden2", 200, "neurons in hl2")
flags.DEFINE_integer("hidden3", 0, "neurons in hl3")
#flags.DEFINE_integer("max_steps", 1000, "steps to perform")
flags.DEFINE_string("plot_file", '', "where to store results?")
#flags.DEFINE_string("plot2_file", 'plot2.csv', "where to store results?")
#flags.DEFINE_string("plot3_file", 'plot3.csv', "where to store results?")
flags.DEFINE_string("save_model", 'xxx', "dummy")
flags.DEFINE_string("permuteTrain", '-1', "-1")
flags.DEFINE_string("permuteTrain2", '-1', "-1")
flags.DEFINE_string("permuteTest",'-1', "-1")
flags.DEFINE_string("permuteTest2", '-1', "-1")
flags.DEFINE_string("permuteTest3", '-1', "-1")
flags.DEFINE_float("dropout_hidden", 0.5, "dropout hidden layer")
flags.DEFINE_float("dropout_input", 0.8, "dropout input layer")
#flags.DEFINE_float("learning_rate", 0.01, "lr")
#flags.DEFINE_integer("training_readout_layer", 1, "srl")



FLAGS = flags.FLAGS
utils.SetDefaultAsNatural(FLAGS)


mean_imm = FLAGS.mean_imm
mode_imm = FLAGS.mode_imm
alpha = FLAGS.alpha
lmbda = FLAGS.regularizer
optimizer = FLAGS.optimizer
learning_rate = FLAGS.learning_rate
epoch = int(FLAGS.epoch)
batch_size = FLAGS.batch_size

no_of_task = FLAGS.tasks ;
no_of_node = [] ;
keep_prob_info = [] ;

if FLAGS.hidden3==0:
  no_of_node = [784,FLAGS.hidden1,FLAGS.hidden2,10]
  keep_prob_info = [0.8, 0.5, 0.5]
else:
  no_of_node = [784,FLAGS.hidden1,FLAGS.hidden2,FLAGS.hidden3, 10] ;
  keep_prob_info = [0.8, 0.5, 0.5, 0.5]
#keep_prob_info = [0.8, 0.5, 0.5, 0.5]

plotfile=None ;
if len(FLAGS.plot_file) > 0:
  plotfile = file(FLAGS.plot_file,"w") ;



def convert2List (s):
  return [int (x) for x in s.split()] ;

# data preprocessing
# x: train data, y: train labels
# x_:test data, y_:test labels
#x, y, x_, y_, xyc_info = preprocess.XycPackage()
#x, y, x_, y_, xyc_info = preprocess.SplitPackage(train_classes = [0,1,2,3,4], train2_classes = [5,6,7,8,9], test_classes = [0,1,2,3,4], test2_classes = [5,6,7,8,9], test3_classes = [0,1,2,3,4,5,6,7,8,9]) ;

x, y, x_, y_, xyc_info = preprocess.SplitPackage(train_classes = convert2List(FLAGS.train_classes),
                         train2_classes = convert2List(FLAGS.train2_classes),
                         test_classes = convert2List(FLAGS.test_classes),
                         test2_classes = convert2List(FLAGS.test2_classes),
                         test3_classes = convert2List(FLAGS.test3_classes),
                         permuteTrain=int(FLAGS.permuteTrain),
                         permuteTrain2 = int(FLAGS.permuteTrain2),
                         permuteTest=int(FLAGS.permuteTest),
                         permuteTest2 = int(FLAGS.permuteTest2) ) ;

print ([_x.shape for _x in x], [_y.shape for _y in y])
print ([_x.shape for _x in x_], [_y.shape for _y in y_])
#print (x_.shape, y_.shape)



start = time.time()

with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

    mlp = imm.TransferNN(no_of_node, (optimizer, learning_rate), keep_prob_info=keep_prob_info)
    mlp.RegPatch(lmbda)

    sess.run(tf.global_variables_initializer())
    L_copy = []
    FM = []
    for i in range(no_of_task):
        print("")
        print("================= Train task #%d (%s) ================" % (i+1, optimizer))
        if plotfile is not None:
          plotfile.write("\n")
          plotfile.write("================= Train task #%d (%s) ================" % (i+1, optimizer)+"\n")

        if i > 0:
            model_utils.CopyLayers(sess, mlp.Layers, mlp.Layers_reg)    # Regularization from weight of pre-task

        mlp.Train(sess, x[i], y[i], np.concatenate(x_), np.concatenate(y_), epoch, mb=batch_size, logTo=plotfile)
        mlp.Test(sess, [[x[i],y[i]," train"], [x_[i],y_[i]," test"]],logTo=plotfile)

        if mean_imm or mode_imm:
            L_copy.append(model_utils.CopyLayerValues(sess, mlp.Layers))
        if mode_imm:
            FM.append(mlp.CalculateFisherMatrix(sess, x[i], y[i]))

    mlp.TestAllTasks(sess, x_, y_, logTo=plotfile)

    for alphaInt in range(0,50):
      alpha = float(alphaInt)/50.0
      alpha_list = [(1-alpha)/(no_of_task-1)] * (no_of_task-1)
      alpha_list.append(alpha)
      ######################### Mean-IMM ##########################
      if mean_imm:
          print("")
          print("Main experiment on %s + Mean-IMM, alpha=%.03f, shuffled MNIST" % (optimizer,alpha))
          print("============== Train task #%d (Mean-IMM) ==============" % no_of_task)
          if plotfile is not None:
            plotfile.write("\n")
            plotfile.write("Main experiment on %s + Mean-IMM, alpha=%.03f, shuffled MNIST\n" % (optimizer,alpha))
            plotfile.write("============== Train task #%d (Mean-IMM) ==============\n" % no_of_task)

          LW = model_utils.UpdateMultiTaskLwWithAlphas(L_copy[0], alpha_list, no_of_task)
          model_utils.AddMultiTaskLayers(sess, L_copy, mlp.Layers, LW, no_of_task)
          ret = mlp.TestTasks(sess, x, y, x_, y_, debug = False)
          utils.PrintResults(alpha, ret, logTo=plotfile)

          mlp.TestAllTasks(sess, x_, y_,logTo=plotfile)

      ######################### Mode-IMM ##########################
      if mode_imm:
        print("")
        print("Main experiment on %s + Mode-IMM, alpha=%.03f, shuffled MNIST" % (optimizer,alpha))
        print("============== Train task #%d (Mode-IMM) ==============" % no_of_task)
        if plotfile is not None:
          plotfile.write("\n")
          plotfile.write("Main experiment on %s + Mode-IMM, alpha=%.03f, shuffled MNIST\n" % (optimizer,alpha))
          plotfile.write("============== Train task #%d (Mode-IMM) ==============\n" % no_of_task)

        LW = model_utils.UpdateMultiTaskWeightWithAlphas(FM, alpha_list, no_of_task)
        model_utils.AddMultiTaskLayers(sess, L_copy, mlp.Layers, LW, no_of_task)
        ret = mlp.TestTasks(sess, x, y, x_, y_, debug = False)
        utils.PrintResults(alpha, ret, logTo=plotfile)

        mlp.TestAllTasks(sess, x_, y_,logTo=plotfile)

    print("")
    print("Time: %.4f s" % (time.time()-start))
