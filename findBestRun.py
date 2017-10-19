# reads all csv files in current dir. that come from from one experiment and gives the best run results
# "best"-->fitness function, can be adapted to fit a particular need
# re-uses the data reading method from the plotting script


from plotOneExp import readResults ;
import sys,os ;

# fitness function
# takes a list of 3 np arrays, containing the accuracies over time from D1D1,D2D2, D2D1 (in that order)
def measureQuality(D):
  return (D[0][:,1]).max() ; # criterion: highest initial training accuracy

def extractTask(runID):
  return runID.split("_")[1] ;


expID = sys.argv[1] ;
csvfiles = [f for f in os.listdir("./") if (f.find(".csv") != -1 and (f.split("_"))[0]==expID ) ] ;

print len(csvfiles) ;

expDict = {}

for f in csvfiles:

  fields = f.replace(".csv","").split("_") ;
  action = fields[-1] ;
  runID = f.replace("_"+action+".csv","")
  print runID,action ;
  if expDict.has_key(runID)==False:
    expDict[runID]={}
    expDict[runID][action]=f ;
  else:
    expDict[runID][action]=f ;

tasks = {}
for key in expDict:
  tasks[extractTask(key)] = True;



bestRunID={key:None for key in tasks};
bestFitness={key:-1 for key in tasks} ;
for key,value in expDict.iteritems():
  if len(value.keys()) >= 3:
    print "valid exp", key ;
    fitness = measureQuality(readResults(key)) ;
    task = extractTask(key) ;
    if fitness> bestFitness[task]:
      bestFitness[task]=fitness ;
      bestRunID[task] = key ;
  else:
    print "invalid exp", key ;
  

for key in tasks:
  print "Task ", key, ": best run was", bestRunID[key], " with a fitness of ", bestFitness[key] ;


  


