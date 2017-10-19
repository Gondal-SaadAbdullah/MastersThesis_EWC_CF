# 1) when doing fc there muist be no mrl experiments
# 2) saving model file and csv nales are not unique
import os, sys, itertools


def getScriptName(expID):
    if expID in ["fc", "D-fc", "fc-MRL", "D-fc-MRL"]:
        return "./Dropout_Experiments/dropout_more_layers.py"
    elif expID in ["conv", "D-conv", "conv-MRL", "D-conv-MRL"]:
        return "./Dropout_Experiments/convnet_more_layers.py"
    elif expID in ["LWTA-fc", "LWTA-fc-MRL"]:
        return "./LWTA_Experiments/lwta_more_layers.py"


# not complete:!!!!!!!
def generateTaskString(task):
    D1 = []
    D2 = []
    D3 = []
    D4 = []
    if task == "D5-5":
        D1 = "0 1 2 3 4"
        D2 = "5 6 7 8 9"
    elif task == "D5-5b":
        D1 = "0 2 4 6 8"
        D2 = "1 3 5 7 9"
    elif task == "D5-5c":
        D1 = "3 4 6 8 9"
        D2 = "0 1 2 5 7"
    elif task == "D9-1":
        D1 = "0 1 2 3 4 5 6 7 8"
        D2 = "9"
    elif task == "D9-1b":
        D1 = "1 2 3 4 5 6 7 8 9"
        D2 = "0"
    elif task == "D9-1c":
        D1 = "0 2 3 4 5 6 7 8 9"
        D2 = "1"
    elif task == "D8-1-1":
        D1 = "0 2 3 4 5 6 7 8"
        D2 = "9"
        D3 = "1"
    elif task == "D7-1-1-1":
        D1 = "2 3 4 5 6 7 8"
        D2 = "9"
        D3 = "1"
        D4 = "0"
    elif task == "DP10-10":
        D1 = "0 1 2 3 4 5 6 7 8 9"
        D2 = "0 1 2 3 4 5 6 7 8 9"
    return D1, D2, D3, D4

def generateUniqueId(expID,params):
  h1 = params[3] ;
  h2 = params[4] ;
  if len(params) > 5:
    h3 = params[5] ;
  else:
    h3=0 ;
  return expID + "_" + params[0] + "_lr_" + str(params[1]) + "_retrainlr_"+str(params[2])+"_layers_"+str(h1)+"_"+str(h2)+"_"+str(h3) ;

  

# not complete!!!
def generateCommandLine(expID,scriptName, action, params,maxSteps=2000):

    # create layer conf parameters
    if len(params) == 5:
        nrHiddenLayers = 2
    else:
        nrHiddenLayers = 3
    hidden_layers = ""
    
    for i in range(0, nrHiddenLayers):
        hidden_layers += "--hidden" + str(i + 1) + " " + str(params[3 + i]) + " "

    D1, D2, D3, D4 = generateTaskString(params[0])

    mlrExperiment = False;
    if expID.find("MRL") != -1:
      mlrExperiment = True;

    trainingReadoutStr = " --training_readout_layer 1" ;
    testingReadoutStr = " --testing_readout_layer 1" ;
    if mlrExperiment == True:
      if action=="D1D1":
        pass ;
      elif action=="D2D2":
        trainingReadoutStr = " --training_readout_layer 2" ;
        testingReadoutStr = " --testing_readout_layer 2" ;
      elif action=="D2D1":
        trainingReadoutStr = " --training_readout_layer 2" ;
        testingReadoutStr = " --testing_readout_layer 1" ;
      if action=="D2D-1":
        trainingReadoutStr = " --training_readout_layer 2" ;
        testingReadoutStr = " --testing_readout_layer -1" ;
    else:
      if action=="D2D-1":
        trainingReadoutStr = " --training_readout_layer 1" ;
        testingReadoutStr = " --testing_readout_layer -1" ;




    model_name = generateUniqueId(expID,params)
    print(model_name)

    # execString that is command to all experiments..
    execStr = scriptName + " " + hidden_layers + "--max_steps "+str(maxSteps)+" " ;

    if action == "D1D1":
        train_classes = " --train_classes " + D1 + trainingReadoutStr
        test_classes = " --test_classes " + D1 + testingReadoutStr
        train_lr = " --learning_rate " + str(params[1])
        if params[0] == "DP10-10":
            execStr = execStr + " --permuteTrain 0 --permuteTest 0 "
        execStr = execStr + " " + train_lr + " " + train_classes + " " + test_classes + \
                  " --save_model " + model_name + "_D1D1 --plot_file " + model_name + "_D1D1.csv" + " --start_at_step 0"
    elif action == "D2D2":
        train_classes = " --train_classes " + D2 + trainingReadoutStr
        test_classes = " --test_classes " + D2 + testingReadoutStr
        retrain_lr = " --learning_rate " + str(params[2])
        if params[0] == "DP10-10":
            execStr = execStr + " --permuteTrain 1 --permuteTest 1"
        execStr = execStr + " " + retrain_lr + " " + train_classes + " " + test_classes + \
                  " --save_model " + model_name + "_D2D2" + " --load_model " + model_name + "_D1D1 --plot_file " + model_name + "_D2D2.csv" + " --start_at_step "+str(maxSteps)
    elif action == "D2D1" or action=="D2D-1":
        supp = "_"+action ;
        train_classes = " --train_classes " + D2 + trainingReadoutStr
        test_classes = " --test_classes " + D1 + testingReadoutStr
        retrain_lr = " --learning_rate " + str(params[2])
        if params[0] == "DP10-10":
            execStr = execStr + "--permuteTrain 1 --permuteTest 0"
        execStr = execStr + " " + retrain_lr + " " + train_classes + " " + test_classes + \
                  " --save_model " + model_name + supp + " --load_model " + model_name + "_D1D1 --plot_file " + model_name + supp+".csv" + " --start_at_step "+str(maxSteps)
    elif action == "D3D3":
        train_classes = " --train_classes " + D3 + " --training_readout_layer 3"
        test_classes = " --test_classes " + D3 + " --testing_readout_layer 3"
        retrain_lr = " --learning_rate " + str(params[2])
        execStr = execStr + " " + retrain_lr + " " + train_classes + " " + test_classes + \
                  " --save_model " + model_name + "_D3D3" + " --load_model " + model_name + "_D2D1 --plot_file " + model_name + "_D3D3.csv" + " --start_at_step 6000"
    elif action == "D3D1":
        train_classes = " --train_classes " + D3 + " --training_readout_layer 3"
        test_classes = " --test_classes " + D1 + " --testing_readout_layer 1"
        retrain_lr = " --learning_rate " + str(params[2])
        execStr = execStr + " " + retrain_lr + " " + train_classes + " " + test_classes + \
                  " --save_model " + model_name + "_D3D1" + " --load_model " + model_name + "_D3D3 --plot_file " + model_name + "_D3D1.csv" + " --start_at_step 8000"
    elif action == "D4D4":
        train_classes = " --train_classes " + D4 + " --training_readout_layer 4"
        test_classes = " --test_classes " + D4 + " --testing_readout_layer 4"
        retrain_lr = " --learning_rate " + str(params[2])
        execStr = execStr + " " + retrain_lr + " " + train_classes + " " + test_classes + \
                  " --save_model " + model_name + "_D4D4" + " --load_model " + model_name + "_D3D1 --plot_file " + model_name + "_D4D4.csv" + " --start_at_step 10000"
    elif action == "D4D1":
        train_classes = " --train_classes " + D4 + " --training_readout_layer 4"
        test_classes = " --test_classes " + D1 + " --testing_readout_layer 1"
        retrain_lr = " --learning_rate " + str(params[2])
        execStr = execStr + " " + retrain_lr + " " + train_classes + " " + test_classes + \
                  " --save_model " + model_name + "_D4D1" + " --load_model " + model_name + "_D4D4 --plot_file " + model_name + "._D4D1.csv" + " --start_at_step 12000"
    else:
        return "??" + action

    # Dropout is default in the programs, this disables dropout
    if scriptName == "fc":
        execStr = execStr + " --dropout_hidden 1 --dropout_input 1"
    elif scriptName == "conv":
        execStr = execStr + " --dropout 1"
    # Dropout is default in the programs, this disables dropout


    if expID == "conv":
        execStr = execStr + " --dropout 1"
    elif expID.find("D-")!= -1:
        execStr = execStr + " --dropout_hidden 0.8 --dropout_input 0.5"
    else:
        execStr = execStr + " --dropout_hidden 1 --dropout_input 1"
    

    return execStr.replace("\n"," ")


expID = sys.argv[1]
N_files = sys.argv[2]   # number of files the experiment is divided into!

scriptName = "python "+getScriptName(expID)
#tasks = ["DP10-10", "D5-5", "D5-5b", "D5-5c", "D9-1", "D9-1b", "D9-1c", "D8-1-1", "D7-1-1-1"]  # missing D8-1-1, D7-1-1-1 for now
tasks = ["DP10-10", "D5-5", "D5-5b", "D5-5c", "D9-1", "D9-1b", "D9-1c"]  # missing D8-1-1, D7-1-1-1 for now
train_lrs = [0.001]
retrain_lrs = [0.001,0.0001, 0.00001]
# layerSizes = [0,200,400,800]
if expID.find("conv") != -1:
    layerSizes = [1]
else:
    layerSizes = [0,200,400,800]

def validParams(t):
  task,lrTrain,lrRetrain,h1,h2,h3 = t;
  if h1==0 or h2==0:
    return False;
  else:
    return True;

def correctParams(t):
  task,lrTrain,lrRetrain,h1,h2,h3 = t;
  if h3==0:
    return (task,lrTrain,lrRetrain,h1,h2);
  else:
    return t ;

combinations = itertools.product(tasks, train_lrs, retrain_lrs, layerSizes, layerSizes, layerSizes)
validCombinations = [correctParams(t) for t in combinations if validParams(t)]
#print len(validCombinations) ;

maxSteps = 1000 ;
limit=40000 ;
n = 0
index=0 ;
alreadyDone={}
files = [file(expID + "-part-" + str(n) + ".bash","w") for n in xrange(0,int(N_files))] ;
for t in validCombinations:
    uniqueID = generateUniqueId(expID,t) ;
    #print uniqueID
    if alreadyDone.has_key(uniqueID):
      print "CONT"
      continue;
    alreadyDone[uniqueID]=True;
    f = files[n] ;
    f.write(generateCommandLine(expID,scriptName, "D1D1", t,maxSteps=maxSteps) + "\n")   # initial training
    f.write(generateCommandLine(expID,scriptName, "D2D2", t,maxSteps=maxSteps) + "\n")  # retraining and eval on D2
    f.write(generateCommandLine(expID,scriptName, "D2D1", t,maxSteps=maxSteps) + "\n")  # retraining andf eval on D1
    f.write(generateCommandLine(expID,scriptName, "D2D-1", t,maxSteps=maxSteps) + "\n")  # retraining andf eval on D1
    if t[0] == "D8-1-1":
        f.write(generateCommandLine(scriptName, "D3D3", t) + "\n")
        f.write(generateCommandLine(scriptName, "D3D1", t) + "\n")
    elif t[0] == "D7-1-1-1":
        f.write(generateCommandLine(scriptName, "D4D4", t) + "\n")
        f.write(generateCommandLine(scriptName, "D4D1", t) + "\n")

    n += 1
    if n >= int(N_files):
        n = 0
    index+=1;
    if index>=limit:
      break ;

for f in files:
  f.close() ;
#print alreadyDone
