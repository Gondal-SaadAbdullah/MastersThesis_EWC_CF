# plots thee csv files from cf experiments into a single png file that is named according to the experiments parameters

import matplotlib.pyplot as plt ;
import os, sys, numpy as np ;
from matplotlib.ticker import MultipleLocator


# takes a model string and returns a list of three numpy arrays with the experimental results stored in 2D matrices
# dim0 - iteration, dim1 - accuracy
def readResults(modelString):
  F = [modelString+"_D1D1.csv", modelString+"_D2D2.csv",modelString+"_D2D1.csv"] ;
  if os.path.exists(modelString+"_D2D-1.csv"):
    F.append(modelString+"_D2D-1.csv") ;

  L = [file(f,"r").readlines() for f in F ]

  _D = [[l.strip().split(",") for l in lines if len(l)>2 ] for lines in L] ;

  d = [None for i in F]
  i=0;
  for _data in _D:
    d[i] = [(float(_d[0]),float(_d[1])) for _d in _data]  
    i+=1

  D = [np.zeros([len(dv),2]) for dv in d] ;

  j=0;
  for _d in D:
    i=0;
    for tup in d[j]:
      #print tup
      _d[i,:] = tup ;
      i+=1;
    j+=1;

  return D ;





if __name__=="__main__":
  params = sys.argv[1].split("_") ;
  titleStr = "Model: "+params[0]+", Task: "+params[1] ;


  fig = plt.figure(1) ;
  ax = plt.gca() ;


  D = readResults(sys.argv[1]) ;

  ax.plot(D[0][:,0],D[0][:,1], linewidth=3,label='D1D1')
  ax.plot(D[1][:,0],D[1][:,1], linewidth=3,label='D2D2')
  ax.plot(D[2][:,0],D[2][:,1], linewidth=3,label='D2D1')
  if len(D)>3:
    ax.plot(D[3][:,0],D[2][:,1], linewidth=3,label='D2D1All')
 
  ax.set_title (titleStr, size=25)
  ax.set_xlabel ("iteration", size=30)
  ax.set_ylabel ("test accuracy", size=30)
  ax.tick_params(labelsize=22)
  ax.xaxis.set_major_locator(MultipleLocator (500)) ;
  ax.yaxis.set_major_locator(MultipleLocator (0.1)) ;
  ax.yaxis.set_minor_locator(MultipleLocator (0.05)) ;
  ax.legend(fontsize=20,loc='lower left')
  ax.grid(True,which='both');
  x = np.arange(0,(D[1][:,0]).max()+50,1) ;
  ax.fill_between(x,0,1,where=(x>(x.shape[0]/2)),facecolor='gray',alpha=0.3)
  plt.tight_layout()
  figName = sys.argv[1]+".png" ;
  plt.savefig("fig.png") ;
  plt.savefig(figName) ;

