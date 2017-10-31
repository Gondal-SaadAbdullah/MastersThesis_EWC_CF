# reads all csv files in current dir. that come from from one experiment and gives the best run results
# "best"-->fitness function, can be adapted to fit a particular need
# re-uses the data reading method from the plotting script

from __future__ import division
from plotOneExp import readResults
import sys, os
import re, numpy as np


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
    maxVal = D[1][:,1].max()
    for idx, acc_D2 in enumerate(D[1][:, 1]):
        if acc_D2 >= maxVal * 0.95:
            D1_weight, D2_weight = getWeightsForAvg(task)
            acc_D1 = D[2][idx, 1]
            return np.average([acc_D1, acc_D2], weights=[D1_weight, D2_weight])

def extractTask(runID):
    return runID.split("_")[1]

def getWeightsForAvg(task):
    p1 = int(re.search(r'\d+', task.split("-")[0]).group())
    p2 = int(re.search(r'\d+', task.split("-")[1]).group())
    return p1/10, p2/10


expID = sys.argv[1]
pathString = sys.argv[2]
csvfiles = [f for f in os.listdir(pathString) if (f.find(".csv") != -1 and (f.split("_"))[0] == expID)]

print len(csvfiles)

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

tasks = {}
for key in expDict:
    tasks[extractTask(key)] = True

bestRunID = {key: None for key in tasks}
bestFitness = {key: -1 for key in tasks}
for key, value in expDict.iteritems():
    if len(value.keys()) >= 3:
        print "valid exp", key
        # fitness = measureQuality(readResults(key, pathString))
        task = extractTask(key)
        # print("quality ::: %s" % (measureQualityWithAvg(readResults(key, pathString), task)))
        fitness = measureQualityWithPcnt(readResults(key, pathString), task)
        if fitness > bestFitness[task]:
            bestFitness[task] = fitness
            bestRunID[task] = key
    else:
        print "invalid exp", key, len(value.keys())

for key in tasks:
    print "Task ", key, ": best run was", bestRunID[key], " with a fitness of ", bestFitness[key]
