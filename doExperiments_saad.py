import os, sys, itertools


def getScriptName(expID):
    if expID in ["fc", "D-fc", "fc-MRL", "D-fc-MRL"]:
        return "./Dropout_Experiments/dropout_more_layers.py"
    elif expID in ["conv", "D-conv", "conv-MRL", "D-conv-MRL"]:
        return "./Dropout_Experiments/convnet_more_layers.py"
    elif expID in ["LWTA-fc-", "LWTA-fc-MRL"]:
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
    elif task == "D8-1-1-1":
        D1 = "2 3 4 5 6 7 8"
        D2 = "9"
        D3 = "1"
        D4 = "0"
    elif task == "DP10-10":
        D1 = "0 1 2 3 4 5 6 7 8 9"
        D2 = "0 1 2 3 4 5 6 7 8 9"
    return D1, D2, D3, D4


# not complete!!!
def generateCommandLine(scriptName, action, params):
    model_name = expID + "_" + params[0] + "_" + str(params[1])
    print(model_name)
    # create layer conf parameters
    if len(params) == 5:
        nrHiddenLayers = 2
    else:
        nrHiddenLayers = 3
    hidden_layers = ""
    for i in range(0, nrHiddenLayers):
        hidden_layers += "--hidden" + str(i + 1) + " " + str(params[3 + i]) + " "

    D1, D2, D3, D4 = generateTaskString(params[0])

    # execString that is command to all experiments..
    execStr = scriptName + " " + hidden_layers + "--max_steps 2000"

    if action == "D1D1":
        train_classes = "--train_classes " + D1 + " --training_readout_layer 1"
        test_classes = "--test_classes " + D1 + " --testing_readout_layer 1"
        train_lr = "--learning_rate " + str(params[1])
        if params[0] == "DP10-10":
            execStr = execStr + "--permuteTrain 0 --permuteTest 0"
        execStr = execStr + " " + train_lr + " " + train_classes + " " + test_classes + \
                  " --save_model " + model_name + "_D1D1 --plot_file" + model_name + "_D1D1.csv" + " --start_at_step 0"
    elif action == "D2D2":
        train_classes = "--train_classes " + D2 + " --training_readout_layer 2"
        test_classes = "--test_classes " + D2 + " --testing_readout_layer 2"
        retrain_lr = "--learning_rate " + str(params[2])
        if params[0] == "DP10-10":
            execStr = execStr + "--permuteTrain 1 --permuteTest 1"
        execStr = execStr + " " + retrain_lr + " " + train_classes + " " + test_classes + \
                  " --save_model " + model_name + "_D2D2" + " --load_model " + model_name + "_D1D1 --plot_file" + model_name + "_D2D2.csv" + " --start_at_step 2000"
    elif action == "D2D1":
        train_classes = "--train_classes " + D2 + " --training_readout_layer 2"
        test_classes = "--test_classes " + D1 + " --testing_readout_layer 1"
        retrain_lr = "--learning_rate " + str(params[2])
        if params[0] == "DP10-10":
            execStr = execStr + "--permuteTrain 0 --permuteTest 1"
        execStr = execStr + " " + retrain_lr + " " + train_classes + " " + test_classes + \
                  " --save_model " + model_name + "_D2D1" + " --load_model " + model_name + "_D2D2 --plot_file" + model_name + "_D2D1.csv" + " --start_at_step 4000"
    elif action == "D3D3":
        train_classes = "--train_classes " + D3 + " --training_readout_layer 3"
        test_classes = "--test_classes " + D3 + " --testing_readout_layer 3"
        retrain_lr = "--learning_rate " + str(params[2])
        execStr = execStr + " " + retrain_lr + " " + train_classes + " " + test_classes + \
                  " --save_model " + model_name + "_D3D3" + " --load_model " + model_name + "_D2D1 --plot_file" + model_name + "_D3D3.csv" + " --start_at_step 6000"
    elif action == "D3D1":
        train_classes = "--train_classes " + D3 + " --training_readout_layer 3"
        test_classes = "--test_classes " + D1 + " --testing_readout_layer 1"
        retrain_lr = "--learning_rate " + str(params[2])
        execStr = execStr + " " + retrain_lr + " " + train_classes + " " + test_classes + \
                  " --save_model " + model_name + "_D3D1" + " --load_model " + model_name + "_D3D3 --plot_file" + model_name + "_D3D1.csv" + " --start_at_step 8000"
    elif action == "D4D4":
        train_classes = "--train_classes " + D4 + " --training_readout_layer 4"
        test_classes = "--test_classes " + D4 + " --testing_readout_layer 4"
        retrain_lr = "--learning_rate " + str(params[2])
        execStr = execStr + " " + retrain_lr + " " + train_classes + " " + test_classes + \
                  " --save_model " + model_name + "_D4D4" + " --load_model " + model_name + "_D3D1 --plot_file" + model_name + "_D4D4.csv" + " --start_at_step 10000"
    elif action == "D4D1":
        train_classes = "--train_classes " + D4 + " --training_readout_layer 4"
        test_classes = "--test_classes " + D1 + " --testing_readout_layer 1"
        retrain_lr = "--learning_rate " + str(params[2])
        execStr = execStr + " " + retrain_lr + " " + train_classes + " " + test_classes + \
                  " --save_model " + model_name + "_D4D1" + " --load_model " + model_name + "_D4D4 --plot_file" + model_name + "._D4D1.csv" + " --start_at_step 12000"
    else:
        return "??" + action

    # Dropout is default in the programs, this disables dropout
    if scriptName == "fc":
        execStr = execStr + " --dropout_hidden 1 --dropout_input 1"
    elif scriptName == "conv":
        execStr = execStr + " --dropout 1"

    return execStr


expID = sys.argv[1]

scriptName = getScriptName(expID)
tasks = ["DP10-10", "D5-5", "D5-5b", "D5-5c", "D9-1", "D9-1b", "D9-1c", "D8-1-1", "D7-1-1-1"]  # missing D8-1-1, D7-1-1-1 for now
train_lrs = [0.01, 0.001]
retrain_lrs = [0.001, 0.0001, 0.00001]
layerSizes = [200, 400, 800]

combinations2Layers = itertools.product(tasks, train_lrs, retrain_lrs, layerSizes, layerSizes)
combinations3Layers = itertools.product(tasks, train_lrs, retrain_lrs, layerSizes, layerSizes, layerSizes)

for t in combinations2Layers:
    print (generateCommandLine(scriptName, "D1D1", t))  # initial training
    print (generateCommandLine(scriptName, "D2D2", t))  # retraining and eval on D2
    print (generateCommandLine(scriptName, "D2D1", t))  # retraining andf eval on D1
    if t[0] == "D8-1-1":
        print (generateCommandLine(scriptName, "D3D3", t))
        print (generateCommandLine(scriptName, "D3D1", t))
    elif t[0] == "D7-1-1-1":
        print (generateCommandLine(scriptName, "D4D4", t))
        print (generateCommandLine(scriptName, "D4D1", t))

# need to do the same for 3 layers
for t in combinations3Layers:
    print (generateCommandLine(scriptName, "D1D1", t))  # initial training
    print (generateCommandLine(scriptName, "D2D2", t))  # retraining and eval on D2
    print (generateCommandLine(scriptName, "D2D1", t))  # retraining andf eval on D1
    if t[0] == "D8-1-1":
        print (generateCommandLine(scriptName, "D3D3", t))
        print (generateCommandLine(scriptName, "D3D1", t))
    elif t[0] == "D7-1-1-1":
        print (generateCommandLine(scriptName, "D4D4", t))
        print (generateCommandLine(scriptName, "D4D1", t))
