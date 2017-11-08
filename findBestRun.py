# reads all csv files in current dir. that come from from one experiment and gives the best run results
# "best"-->fitness function, can be adapted to fit a particular need
# re-uses the data reading method from the plotting script

from __future__ import division
from plotOneExp import readResults
import sys, os
import re, numpy as np, math

from plotOneExp import readResults
import sys, os, numpy as np

w1 = 0.5
w2 = 0.5


# fitness function
# takes a list of 3 np arrays, containing the accuracies over time from D1D1,D2D2, D2D1 (in that order)
def measureQuality(D):
    return (D[0][:, 1]).max()  # criterion: highest initial training accuracy


def measureQualityWithAvg(D, task):
    temp_acc = 0
    for idx, acc_D2 in enumerate(D[1][:, 1]):
        if (round(acc_D2) != 0) and (0 <= acc_D2 - temp_acc <= 0.5):
            D1_weight, D2_weight = getWeightsForAvg(task)
            acc_D1 = D[2][idx, 1]
            return np.average([acc_D1, acc_D2], weights=[D1_weight, D2_weight])
        else:
            temp_acc = acc_D2


def measureQualityWithPcnt(D, task):
    maxVal = D[1][:, 1].max()
    for idx, acc_D2 in enumerate(D[1][:, 1]):
        if acc_D2 >= maxVal * 0.98:
            D1_weight, D2_weight = getWeightsForAvg(task)
            acc_D1 = D[2][idx, 1]
            return np.average([acc_D1, acc_D2], weights=[D1_weight, D2_weight])


# best performance on D1
# quality = performance auf gesamtdatensatz zum ZP i
def measureQualityAlexD1(D):
    if D is None:
      return -1.0 ;
    for d in D:
      if d.shape[0] < 20:
        return -1.0 ;
    D1D1 = D[0]
    return D1D1[:,1].max() ;



# i = wann erreich D2D2 k M seines max?
# quality = performance auf gesamtdatensatz zum ZP i
def measureQualityAlex(D):
    if D is None:
      return -1.0 ;
    D2D2 = D[1]
    for d in D:
      if d.shape[0] < 20:
        return -1.0 ;
    maxD2D2 = D2D2[:, 1].max() * 0.99
    for i in xrange(0, D2D2.shape[0]):
        if D2D2[i, 1] >= maxD2D2:
            return w1 * D[2][i, 1] + w2 * D[1][i, 1]


# i = wann erreich D2D2 k M seines max?
# quality = performance auf gesamtdatensatz zum ZP i
# aber: lies performance auf D1 aus Datei _D2D-1 ab
def measureQualityAlexD2D_1(D):
    if D is None:
      return -1.0 ;
    if len(D) < 4:
      return -1
    for d in D:
      if d.shape[0] < 20:
        return -1.0 ;

    D2D2 = D[1]
    maxD2D2 = D2D2[:, 1].max() * 0.99
    for i in xrange(0, D2D2.shape[0]):
        if D2D2[i, 1] >= maxD2D2:
          return w1 * D[3][i, 1] + w2 * D[1][i, 1]              


def extractTask(runID):
    fields = runID.split("_")
    params=fields[2] ;
    for f in fields[3:]:
      params = params+"_"+f;

    return fields[1],params


def getWeightsForAvg(task):
    p1 = int(re.search(r'\d+', task.split("-")[0]).group())
    p2 = int(re.search(r'\d+', task.split("-")[1]).group())
    return p1 / 10, p2 / 10

def calcPerfMatrix(expDict,qualityMeasure,qualityMeasureMRL,useMRL=False):

  tasks = {}
  taskLookup = {}
  paramLookup = {}
  taskCount = 0 ;
  paramCount=0;

  for key in expDict:
    _t,_p = extractTask(key) ; 
    tasks[_t] = True;
    if taskLookup.has_key(_t)==False:
       taskLookup[_t]=taskCount ;
       taskCount+=1;
    if paramLookup.has_key(_p)==False:
       paramLookup[_p]=paramCount ;
       paramCount+=1;

  resultMatrix = np.zeros([len(taskLookup.keys()),len(paramLookup.keys())]) ;


  bestRunID={key:None for key in tasks};
  bestFitness={key:-1 for key in tasks} ;
  worstRunID={key:None for key in tasks};
  worstFitness={key:1000. for key in tasks} ;
  sumX={key:0. for key in tasks} ;
  sumX2={key:0. for key in tasks} ;
  count={key:0. for key in tasks} ;
  validExps = 0 ;
  for key,value in expDict.iteritems():
    if len(value.keys()) >= 3:
      #print "valid exp", key ;
      validExps += 1 ;
      if key.find("9-1")!= -1:
        w1 = 0.9 ; w2 = 0.1 ;
      if key.find("5-5")!= -1:
        w1 = 0.5 ; w2 = 0.5 ;
      if key.find("10-10")!= -1:
        w1 = 0.5 ; w2 = 0.5 ;

      fitness = qualityMeasure(readResults(key,pathString)) ;
      if useMRL==True:
        fitness = qualityMeasureMRL(readResults(key,pathString)) ;
  
      task,params = extractTask(key) ;
      resultMatrix[taskLookup[task],paramLookup[params]] = fitness ;
      """sumX [task ]+= fitness ;
      sumX2 [task] += fitness*fitness ;
      count [task] += 1.0 ;
      if fitness> bestFitness[task]:
        bestFitness[task]=fitness ;
        bestRunID[task] = key ;
      if fitness< worstFitness[task]:
        worstFitness[task]=fitness ;
        worstRunID[task] = key ;
      """

    else:
      print "invalid exp", key, len(value.keys()) ;
  return resultMatrix,taskLookup,paramLookup ;
  
def printResultMatrix(resultMatrix,taskLookup,paramLookup):
  tpm= resultMatrix.transpose()
  for task,taskI in taskLookup.iteritems():
    print "%6s"%(task),
  print
  for param,paramI in paramLookup.iteritems():

    for task,taskI in taskLookup.iteritems():
      print "%.4f"%(resultMatrix[taskI,paramI]),
    print param


expID = sys.argv[1]
pathString = "./"
if len(sys.argv) > 2:
    pathString = sys.argv[2]
useMRL = False
if len(sys.argv) > 3:
    useMRL = True

csvfiles = [f for f in os.listdir(pathString) if (f.find(".csv") != -1 and (f.split("_"))[0] == expID)]

expDict = {}

for f in csvfiles:

    fields = f.replace(".csv", "").split("_")
    action = fields[-1]
    runID = f.replace("_" + action + ".csv", "")
    # print runID,action
    if expDict.has_key(runID) == False:
        expDict[runID] = {}
        expDict[runID][action] = f
    else:
        expDict[runID][action] = f

# expDict: keys are runIDs composes of dataset_params
# values are lists of csv files
# tasks contains just the dataset without the params


resultMatrixTrain,taskLookup,paramLookup = calcPerfMatrix(expDict,measureQualityAlexD1,measureQualityAlexD1,useMRL) ;
resultMatrixRetrain,taskLookup,paramLookup = calcPerfMatrix(expDict,measureQualityAlex,measureQualityAlexD2D_1,useMRL) ;
#printResultMatrix(resultMatrixTrain,taskLookup,paramLookup)

for task,taskI in taskLookup.iteritems():
  bestModelIndex = resultMatrixTrain[taskI,:].argmax() ;
  perfMeasure = resultMatrixRetrain[taskI,bestModelIndex] ;
  print 'Task',task, "retrain perf incremental=",perfMeasure

"""
invalid_tasks = []
for key in tasks:
    if bestRunID[key] is not None:
      print "Task ", key, ": best/worst run was", bestRunID[key],"/",worstRunID[key], " with a fitness of ", bestFitness[key],      "/",worstFitness[key], "mean/var=",sumX[key]/(count[key]+0.001),math.sqrt((sumX2[key]/(count[key]+0.001)-(sumX[key]/(count[key]+0.001))**2.)) ;
    elif key not in invalid_tasks:
        invalid_tasks.append(key)

if invalid_tasks:
    print "Some invalid experiment results for %s were omitted" % invalid_tasks
"""


