from sklearn.tree import DecisionTreeClassifier
import numpy as np
from numpy import genfromtxt
import csv
from sklearn.metrics import auc, roc_curve
from sklearn import metrics
from sklearn import cross_validation
import SMOTE
import numexpr as ne


#global variables
global trainingInput
global trainingNoNa 
global X
global Y

def extractNA(trainingInputfile, trainingNoNafile):
    global trainingInput, trainingNoNa 
    trainingInput=trainingInputfile
    trainingNoNa=trainingNoNafile
    data = csv.reader(open(trainingInput, 'rb'), delimiter=',')
    data_writer = csv.writer(open(trainingNoNa, 'wt'), delimiter = ',')
    for row in data:
        if('NA' not in row):
            data_writer.writerow(row)

def loadNoNA(trainingNoNa):
    #Import data from kaggle training set
    trainingdata = genfromtxt(trainingNoNa, delimiter=',')
    columns_to_keep = range(trainingdata.shape[1])
    columns_to_keep.remove(0) #first column is just row #
    columns_to_keep.remove(1) #second col is Y

    Y=trainingdata[:,1].astype('int32') #change Y to int
    X=trainingdata[:,columns_to_keep]
    return(X,Y)

def savePredictions():
    global predictions
    #predictions = np.asarray(Yheldout)
    #np.savetxt('/Users/kat/Dropbox/Public/kaggle/Gimme Some Credit/Yheldout.csv', Yheldout, delimiter=',')

def createDecisionTree(X,Y):
    classifier = DecisionTreeClassifier(random_state=12345678)
    unbalancedTree = classifier.fit(X,Y)
    return unbalancedTree #unbalanced_predictions = unbalancedTree.predict(Xheldout)


def avg(listofnumbers):
    return sum(listofnumbers)/(1.0*len(listofnumbers))
    
def decisionTree(X,Y):
    # Load data
    #extractNA('/Users/kat/Dropbox/Public/kaggle/Gimme Some Credit/training-without-header.csv','/Users/kat/Dropbox/Public/kaggle/Gimme Some Credit/training-noheader-noNA.csv')
    # Run classifier with crossvalidation and plot ROC curves
    cv = cross_validation.StratifiedKFold(Y, k=10)
    classifier = createDecisionTree(X,Y)
    avg_roc=[]
    for i, (train,test) in enumerate(cv):
        prob = classifier.fit(X[train], Y[train]).predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(Y[test], prob[:, 1])
        roc = auc(fpr,tpr)
        print("area under ROC: %f" % roc)
        avg_roc.append(roc)
    print("kfold area under ROC: %f" % avg(avg_roc))

def createSMOTEsamples(X,Y,nearestneigh,numNeighbors,majoritylabel,minoritylabel):
    global synX, synY
    (synX,synY) = SMOTE.createSyntheticSamples(X,Y,nearestneigh,numNeighbors,majoritylabel,minoritylabel)

def saveSMOTEsamples(outputfilename):
    global synX, synY
    np.save(outputfilename, synX,synY)
    
def loadSMOTEsamples(inputfilename):
    (X,Y) = np.load(inputfilename)
    return(X,Y)
    
    

if __name__ == '__main__':
    #first pass:
    #extractNA('/Users/kat/Dropbox/Public/kaggle/Gimme Some Credit/training-without-header.csv','/Users/kat/Dropbox/Public/kaggle/Gimme Some Credit/training-noheader-noNA.csv')
    (X,Y)=loadNoNA('/Users/kat/Dropbox/Public/kaggle/Gimme Some Credit/training-noheader-noNA.csv')
    createSMOTEsamples(X,Y,6,3,0,1)
    saveSMOTEsamples('/Users/kat/Dropbox/Public/kaggle/Gimme Some Credit/smote')
    
    #secondpass:
    
    decisionTree(X,Y)
    
    (synX,synY) = loadSMOTEsamples('/Users/kat/Dropbox/Public/kaggle/Gimme Some Credit/smote.npy')
    
    decisionTree(synX,synY)
