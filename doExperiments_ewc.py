# 1) when doing fc there muist be no mrl experiments
# 2) saving model file and csv nales are not unique
import os, sys, itertools

def getScriptName(expID):
    if expID in ["ewc", "D-ewc"]:
        return "ewc_with_options.py"

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
    elif task == "DP10-10":
        D1 = "0 1 2 3 4 5 6 7 8 9"
        D2 = "0 1 2 3 4 5 6 7 8 9"
    elif task == "DP5-5":
        D1 = "0 1 2 3 4"
        D2 = "5 6 7 8 9"

    return D1, D2, D3, D4


def generateUniqueId(expID, params):
    h1 = params[3];
    h2 = params[4];
    if len(params) > 5:
        h3 = params[5];
    else:
        h3 = 0;
    return expID + "_" + params[0] + "_lr_" + str(params[1]) + "_retrainlr_" + str(params[2]) + "_layers_" + str(
        h1) + "_" + str(h2) + "_" + str(h3);


# not complete!!!
def generateCommandLine(expID, scriptName, action, params, maxSteps=2000):
    # create layer conf parameters
    if len(params) == 5:
        nrHiddenLayers = 2
    else:
        nrHiddenLayers = 3
    hidden_layers = ""

    for i in range(0, nrHiddenLayers):
        hidden_layers += "--hidden" + str(i + 1) + " " + str(params[3 + i]) + " "

    D1, D2, D3, D4 = generateTaskString(params[0])

    model_name = generateUniqueId(expID, params)
    print(model_name)

    # execString that is command to all experiments..
    execStr = scriptName + " " + hidden_layers + "--max_steps " + str(maxSteps) + " ";

    if action == "D1D1":
        train_classes = " --train_classes " + D1
        test_classes = " --test_classes " + D1
        train_lr = " --learning_rate " + str(params[1])
        if params[0] in ["DP10-10", "DP5-5"]:
            execStr = execStr + " --permuteTrain 0 --permuteTest 0 "
        execStr = execStr + " " + train_lr + " " + train_classes + " " + test_classes + \
                  " --save_model " + model_name + "_D1D1 --plot_file " + model_name + "_D1D1.csv" + " --start_at_step 0"
    elif action == "D2D2":
        train_classes = " --train_classes " + D2
        test_classes = " --test_classes " + D2
        retrain_lr = " --learning_rate " + str(params[2])
        if params[0] in ["DP10-10", "DP5-5"]:
            execStr = execStr + " --permuteTrain 1 --permuteTest 1"
        execStr = execStr + " " + retrain_lr + " " + train_classes + " " + test_classes + \
                  " --load_model " + model_name + "_D1D1 --plot_file " + model_name + "_D2D2.csv" + " --start_at_step " + str(
            maxSteps)
    elif action == "D2D1" or action == "D2D-1":
        supp = "_" + action;
        train_classes = " --train_classes " + D2
        test_classes = " --test_classes " + D1
        retrain_lr = " --learning_rate " + str(params[2])
        if params[0] in ["DP10-10", "DP5-5"]:
            execStr = execStr + "--permuteTrain 1 --permuteTest 0"
        execStr = execStr + " " + retrain_lr + " " + train_classes + " " + test_classes + \
                  " --load_model " + model_name + "_D1D1 --plot_file " + model_name + supp + ".csv" + " --start_at_step " + str(
            maxSteps)
    elif action == "D3D3":
        train_classes = " --train_classes " + D3
        test_classes = " --test_classes " + D3
        retrain_lr = " --learning_rate " + str(params[2])
        execStr = execStr + " " + retrain_lr + " " + train_classes + " " + test_classes + \
                  " --load_model " + model_name + "_D2D1 --plot_file " + model_name + "_D3D3.csv" + " --start_at_step 6000"
    elif action == "D3D1":
        train_classes = " --train_classes " + D3
        test_classes = " --test_classes " + D1
        retrain_lr = " --learning_rate " + str(params[2])
        execStr = execStr + " " + retrain_lr + " " + train_classes + " " + test_classes + \
                  " --load_model " + model_name + "_D3D3 --plot_file " + model_name + "_D3D1.csv" + " --start_at_step 8000"
    elif action == "D4D4":
        train_classes = " --train_classes " + D4
        test_classes = " --test_classes " + D4
        retrain_lr = " --learning_rate " + str(params[2])
        execStr = execStr + " " + retrain_lr + " " + train_classes + " " + test_classes + \
                  " --load_model " + model_name + "_D3D1 --plot_file " + model_name + "_D4D4.csv" + " --start_at_step 10000"
    elif action == "D4D1":
        train_classes = " --train_classes " + D4
        test_classes = " --test_classes " + D1
        retrain_lr = " --learning_rate " + str(params[2])
        execStr = execStr + " " + retrain_lr + " " + train_classes + " " + test_classes + \
                  " --load_model " + model_name + "_D4D4 --plot_file " + model_name + "._D4D1.csv" + " --start_at_step 12000"
    else:
        return "??" + action

    # Dropout is default in the programs, this disables dropout
    if scriptName.find("D-") == -1:
        if scriptName.find("conv") != -1:
            execStr = execStr + " --dropout 1"
        else:
            execStr = execStr + " --dropout_hidden 1 --dropout_input 1"
    else:
        # Dropout is default in the programs, this enables dropout
        if expID.find("conv") != -1:
            execStr = execStr + " --dropout 0.5"
        else:
            execStr = execStr + " --dropout_hidden 0.8 --dropout_input 0.5"

    return execStr.replace("\n", " ")


expID = sys.argv[1]
N_files = sys.argv[2]  # number of files the experiment is divided into!

scriptName = "python " + getScriptName(expID)

tasks = ["DP5-5", "DP10-10", "D5-5", "D5-5b", "D5-5c", "D9-1", "D9-1b", "D9-1c"]
train_lrs = [0.001]
retrain_lrs = [0.001, 0.0001, 0.00001]
layerSizes = [0, 200, 400, 800]


def validParams(t):
    task, lrTrain, lrRetrain, h1, h2, h3 = t;
    if h1 == 0 or h2 == 0:
        return False;
    else:
        return True;


def correctParams(t):
    task, lrTrain, lrRetrain, h1, h2, h3 = t;
    if h3 == 0:
        return (task, lrTrain, lrRetrain, h1, h2);
    else:
        return t;


# def removeCheckpoints(checkpointDir,uniqueID):
#  list =

combinations = itertools.product(tasks, train_lrs, retrain_lrs, layerSizes, layerSizes, layerSizes)
validCombinations = [correctParams(t) for t in combinations if validParams(t)]
# print len(validCombinations) ;

maxSteps = 1500;
limit = 40000;
n = 0
index = 0;
alreadyDone = {}
files = [file(expID + "-part-" + str(n) + ".bash", "w") for n in xrange(0, int(N_files))];
for t in validCombinations:
    uniqueID = generateUniqueId(expID, t);
    # print uniqueID
    if alreadyDone.has_key(uniqueID):
        print "CONT"
        continue;
    alreadyDone[uniqueID] = True;
    f = files[n];
    f.write(generateCommandLine(expID, scriptName, "D1D1", t, maxSteps=maxSteps) + "\n")  # initial training
    f.write(generateCommandLine(expID, scriptName, "D2D2", t, maxSteps=maxSteps) + "\n")  # retraining and eval on D2
    f.write(generateCommandLine(expID, scriptName, "D2D1", t, maxSteps=maxSteps) + "\n")  # retraining and eval on D1
    f.write(generateCommandLine(expID, scriptName, "D2D-1", t, maxSteps=maxSteps) + "\n")  # retraining and eval on D1
    f.write("rm checkpoints/" + uniqueID + "*\n")

    n += 1
    if n >= int(N_files):
        n = 0
    index += 1;
    if index >= limit:
        break;

for f in files:
    f.close();
# print alreadyDone
