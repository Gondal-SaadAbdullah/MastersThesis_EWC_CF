import os,sys,itertools

def getScriptName(expID):
  if expID in ["fc","D-fc","fc-MRL","D-fc-MRL"]:
    return "./Dropout_Experiments/dropout_more_layers.py"
  elif expID in ["conv","D-conv","conv-MRL","D-conv-MRL"]:
    return "./Dropout_Experiments/convnet_more_layers.py"
  elif expID in ["LWTA-fc-","LWTA-fc-MRL"]:
    return "./LWTA_Experiments/lwta_more_layers.py"

# not complete:!!!!!!!
def generateTaskString(task):
  if task == "D5-5":
    return "--train_classes 0 1 2 3 4 --test_classes 5 6 7 8 9"
  elif task == "D5-5b":
    return "--train_classes 0 2 4 6 8 --test_classes 1 3 5 7 9"
  elif task == "D5-5c":
    return "--train_classes 3 4 6 8 9 --test_classes 0 1 2 5 7"
  elif task == "D9-1":
    return "--train_classes 0 1 2 3 4 5 6 7 8 --test_classes 9"
  elif task == "D9-1b":
    return "--train_classes 1 2 3 4 5 6 7 8 9 --test_classes 0"
  elif task == "D9-1c":
    return "--train_classes 0 2 3 4 5 6 7 8 9 --test_classes 1"
  else:
    return "--train_classes X X X --test_classes Y Y Y"

# not complete!!!
def generateCommandLine (scriptName, action, params):
  # create layer conf parameters
  if len(params) == 5:
    nrHiddenLayers=2
  else:
    nrHiddenLayers=3
  layerCfg = ""
  for i in range(0,nrHiddenLayers):
    layerCfg += "--hidden"+str(i+1)+" "+str(params[3+i])+" "

  # create class config. parameters
  taskCfg = generateTaskString(params[0])

  # execString that is command to all experiments..
  execStr = scriptName + " "+taskCfg + " "+layerCfg

  if action=="D1D1":
    train_lr = "--learning_rate "+str(params[1])
    return execStr+" "+train_lr
  elif action=="D2D2":
    train_lr = "--learning_rate "+str(params[1])
    return execStr+" "+train_lr
  elif action=="D2D1":
    train_lr = "--learning_rate "+str(params[1])
    return execStr+" "+train_lr
  else: return "??"+action



expID = sys.argv[1]

scriptName = getScriptName(expID)
tasks = ["DP10-10","D5-5","D5-5b","D5-5c","D9-1","D9-1b","D9-1c"] # missing D8-1-1, D7-1-1-1 for now
train_lrs = [0.01, 0.001]
retrain_lrs = [0.001, 0.0001, 0.00001]
layerSizes = [200, 400, 800]

combinations2Layers = itertools.product(tasks,train_lrs,retrain_lrs,layerSizes,layerSizes)
combinations3Layers = itertools.product(tasks,train_lrs,retrain_lrs,layerSizes,layerSizes,layerSizes)

for t in combinations2Layers:
  print (generateCommandLine(scriptName,"D1D1",t)) # initial training
  print (generateCommandLine(scriptName,"D2D2",t)) # retraining and eval on D2
  print (generateCommandLine(scriptName,"D2D1",t)) # retraining andf eval on D1

# need to do the same for 3 layers
    




