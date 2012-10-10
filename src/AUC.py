import csv
from sklearn import metrics
import numpy as np

def tiedrank(X):  
  Z = [(x, i) for i, x in enumerate(X)]  
  Z.sort()  
  n = len(Z)  
  Rx = [0]*n   
  for j, (x,i) in enumerate(Z):  
    Rx[i] = j+1  
  s = 1           # sum of ties.  
  start = end = 0 # starting and ending marks.  
  for i in range(1, n):  
    if Z[i][0] == Z[i-1][0] and i != n-1:  
      pos = Z[i][1]  
      s+= Rx[pos]  
      end = i   
    else: #end of similar x values.  
      tiedRank = float(s)/(end-start+1)  
      for j in range(start, end+1):  
        Rx[Z[j][1]] = tiedRank  
      for j in range(start, end+1):  
        Rx[Z[j][1]] = tiedRank  
      start = end = i  
      s = Rx[Z[i][1]]    
  return Rx

def AUC(labels, posterior):
  r = tiedrank(posterior)
  auc = (sum(r*(labels==1)) - sum(labels==1)*(sum(labels==1)+1)/2) / (sum(labels<1)*sum(labels==1));
  return auc

