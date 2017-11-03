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


def measureQualityAlex(D):
    D2D2 = D[1]
    maxD2D2 = D2D2[:, 1].max() * 0.95
    for i in xrange(0, D2D2.shape[0]):
        if D2D2[i, 1] >= maxD2D2:
            return w1 * D[2][i, 1] + w2 * D[1][i, 1]


def measureQualityAlexD2D_1(D):
    D2D2 = D[1]
    maxD2D2 = D2D2[:, 1].max() * 0.99
    for i in xrange(0, D2D2.shape[0]):
        if D2D2[i, 1] >= maxD2D2:
            return w1 * D[3][i, 1] + w2 * D[1][i, 1]


def extractTask(runID):
    return runID.split("_")[1]


def getWeightsForAvg(task):
    p1 = int(re.search(r'\d+', task.split("-")[0]).group())
    p2 = int(re.search(r'\d+', task.split("-")[1]).group())
    return p1 / 10, p2 / 10


expID = sys.argv[1]
pathString = "./"
if len(sys.argv) > 2:
    pathString = sys.argv[2]
useMRL = False
if len(sys.argv) > 3:
    useMRL = True

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
sumX = {key: 0. for key in tasks}
sumX2 = {key: 0. for key in tasks}
count = {key: 0. for key in tasks}
for key, value in expDict.iteritems():
    if len(value.keys()) >= 3:
        print "valid exp", key
        if key.find("9-1") != -1:
            w1 = 0.9
            w2 = 0.1
        if key.find("5-5") != -1:
            w1 = 0.5
            w2 = 0.5
        if key.find("10-10") != -1:
            w1 = 0.5
            w2 = 0.5

        fitness = measureQualityAlex(readResults(key, pathString))
        if useMRL == True:
            fitness = measureQualityAlexD2D_1(readResults(key, pathString))

        task = extractTask(key)
        sumX[task] += fitness
        sumX2[task] += fitness * fitness
        count[task] += 1.0
        if fitness > bestFitness[task]:
            bestFitness[task] = fitness
            bestRunID[task] = key
    else:
        print "invalid exp", key, len(value.keys())

print tasks

invalid_tasks = []
for key in tasks:
    if bestRunID[key] is not None:
        print "Task ", key, ": best run was", bestRunID[key], " with a fitness of ", \
            bestFitness[key], "mean/var=", sumX[key] / count[key], \
            math.sqrt((sumX2[key] / count[key] - (sumX[key] / count[key]) ** 2.))
    elif key not in invalid_tasks:
        invalid_tasks.append(key)

if invalid_tasks:
    print "Some invalid experiment results for %s were omitted" % invalid_tasks
