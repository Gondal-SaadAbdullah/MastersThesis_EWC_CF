import matplotlib.pyplot as plt ;
import numpy as np,random ;
import pickle,sys ;

f = plt.figure(1) ;
ax=plt.gca() ;

random.seed(None)

fdata = pickle.load(file("matrix.pkl","rb"))
resultMatrixTrainRetrain ,taskLookup,paramLookup = fdata ;
print resultMatrixTrainRetrain.shape
resultMatrixTrainRetrain *= (resultMatrixTrainRetrain>0)

nModels = 10 ;
indices = range(0,resultMatrixTrainRetrain.shape[1]) ;
random.shuffle(indices) ;
indices = indices[0:nModels]

data = resultMatrixTrainRetrain[:,indices]
print data.shape


#ax.set_xticklabels(xrange(0,data.shape[1]),taskLookup.keys(),rotation='vertical')
print "tasks",len(taskLookup.keys())

taskLabels=[]
taskIndices=[]
for taskI in xrange(0,len(taskLookup.keys())):
  ind = taskLookup.values().index(taskI) ;
  key = taskLookup.keys()[ind] ;
  taskLabels.append(key) ;
  taskIndices.append(taskI)

plt.xticks(taskIndices,taskLabels,rotation='vertical')


for i in xrange(0,nModels):
  line = data[:,i] ;
  print line.shape
  lIndex = indices[i] ;
  lIndexPos = paramLookup.values().index(lIndex) ;
  paramStr = paramLookup.keys()[lIndexPos]

  
  ax.plot(line,linewidth=3,label=paramStr) ;

ax.set_title (sys.argv[1], size=25)
ax.tick_params(labelsize=20)
ax.set_xlabel ("task", size=30)
ax.set_ylabel ("incr. performance", size=30)
ax.legend(fontsize=7,loc='upper right')

#plt.show() ;
plt.tight_layout()
plt.savefig(sys.argv[1]+".svg") ;

