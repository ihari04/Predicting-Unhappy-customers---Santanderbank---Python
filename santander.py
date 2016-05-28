
# coding: utf-8

# In[2]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import cross_validation
from sklearn.metrics import roc_auc_score as auc
import time


# In[3]:

#Load the data
trainDataFrame = pd.read_csv('C:/Users/Haribabu/Google Drive/Semester-2/DataMining-II/Group Project/train.csv')


# In[4]:

# remove constant columns
colsToRemove = []
for col in trainDataFrame.columns:
    if trainDataFrame[col].std() == 0:
        colsToRemove.append(col)

trainDataFrame.drop(colsToRemove, axis=1, inplace=True)

# remove duplicate columns
colsToRemove = []
columns = trainDataFrame.columns
for i in range(len(columns)-1):
    v = trainDataFrame[columns[i]].values
    for j in range(i+1,len(columns)):
        if np.array_equal(v,trainDataFrame[columns[j]].values):
            colsToRemove.append(columns[j])
print("Number of columns to be removed", len(colsToRemove))
# 29 columns are to be removed


# In[5]:

trainDataFrame1 = trainDataFrame[trainDataFrame['TARGET']==1]
trainDataFrame0 = trainDataFrame[trainDataFrame['TARGET']==0]
print("Number of ones in Dataset", len(trainDataFrame1))
print("Number of zeros in Dataset", len(trainDataFrame0))
##len(trainDataFrame1) - 3008 records are ones
##len(trainDataFrame0) - 73012 records are zeros


# In[6]:

#Split test train from ones
trainLabels = trainDataFrame1['TARGET']
trainFeatures = trainDataFrame1.drop(['ID','TARGET'], axis=1)
from sklearn.cross_validation import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(trainFeatures,trainLabels,test_size=0.30)

#Split test train from Zeros
trainLabels = trainDataFrame0['TARGET']
trainFeatures = trainDataFrame0.drop(['ID','TARGET'], axis=1)
from sklearn.cross_validation import train_test_split
X_train0, X_test0, y_train0, y_test0 = train_test_split(trainFeatures,trainLabels,test_size=0.30)

#Get triaining data set
X_traincombined0 = pd.concat([X_train0,y_train0], axis=1)
X_traincombined1 = pd.concat([X_train1,y_train1], axis=1)
traincombined = pd.concat([X_traincombined0,X_traincombined1])

X_train = pd.concat([X_train1,X_train0], axis=0)
X_test = pd.concat([X_test1,X_test0], axis=0)
y_train = pd.concat([y_train1,y_train0], axis=0)
y_test = pd.concat([y_test1,y_test0], axis=0)


# In[7]:

# Using Extra tree classsifier for feature selection
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X_train,y_train)
# display the relative importance of each attribute
#print(model.feature_importances_)
#print(model)

featuresImportance = pd.DataFrame()
featuresImportance[1]=model.feature_importances_
featuresImportance[0]= range(0,len(model.feature_importances_))

featuresImportanceSorted=featuresImportance.sort_values(1,axis=0,ascending=False)
featuresImportanceSorted=featuresImportanceSorted[featuresImportanceSorted[1]>0.001]
collist=X_train.columns[(featuresImportanceSorted[0]).tolist()]
print ('Number of features selected', len(collist))

# Number of columns selected 72


# In[8]:

# Subset the training and testing dataset based on col names from extratree classifier
trainSet=X_train[collist.tolist()]
testSet=X_test[collist.tolist()]


# In[34]:

#Fitting Logistic regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import time
model = LogisticRegression()
# create the RFE model and select 1 attribute, this would give ranking for all the features
rfe = RFE(model,1)
start_time = time.clock()
rfe = rfe.fit(trainSet, y_train)
rfeExecutionTime= time.clock() - start_time
rfeExecutionTime=rfeExecutionTime/(60)
print("execution time in minutes : %d " %rfeExecutionTime)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)

model.fit(trainSet, y_train)

#print ("Predicted class %s, real class %s" % (model.predict(trainSet),y_train))
#print ("Probabilities for each class from : %s" % model.predict_proba(trainSet))

#Insample performace
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
ypred=model.predict(trainSet)
Confusionmatrix=confusion_matrix(y_train, ypred)
#print(Confusionmatrix)
accuracy_score(y_train, ypred)
preds = model.predict_proba(trainSet)[:,1]
metrics.roc_auc_score(y_train, preds)
print('In sample AUC score', metrics.roc_auc_score(y_train, preds))
#In sample AUC score 0.605591624053

#Out of sample performance
ypred=model.predict(testSet)
Confusionmatrix=confusion_matrix(y_test, ypred)
#print(Confusionmatrix)
accuracy_score(y_test, ypred)
preds = model.predict_proba(testSet)[:,1]
metrics.roc_auc_score(y_test, preds)
print('Out of sample AUC score', metrics.roc_auc_score(y_test, preds))
#Out of sample AUC score 0.604693378617


# In[28]:

#Fitting the classification tree
from sklearn import tree
clfDT = tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=500,max_depth=10)
clfDT = clfDT.fit(trainSet, y_train)

#Print the structure of the Tree
#from sklearn.externals.six import StringIO
#import pydot
#dotfile = StringIO()
#tree.export_graphviz(clfDT, out_file=dotfile)
#pydot.graph_from_dot_data(dotfile.getvalue()).write_pdf("dtree2.pdf")
#dotfile = tree.export_graphviz(clfDT, out_file = 'tree.dot', feature_names = trainSet.columns)
#dotfile.close()
#system("dot -Tpng D:.dot -o D:/dtree2.png")

#Insample performace
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
ypred=clfDT.predict(trainSet)
Confusionmatrix=confusion_matrix(y_train, ypred)
#print(Confusionmatrix)
accuracy_score(y_train, ypred)
preds = clfDT.predict_proba(trainSet)[:,1]
metrics.roc_auc_score(y_train, preds)
print('In sample AUC score', metrics.roc_auc_score(y_train, preds))
#In sample AUC score 0.862390644227

#Out of sample performance
ypred=clfDT.predict(testSet)
Confusionmatrix=confusion_matrix(y_test, ypred)
#print(Confusionmatrix)
accuracy_score(y_test, ypred)
preds = clfDT.predict_proba(testSet)[:,1]
metrics.roc_auc_score(y_test, preds)
print('Out of sample AUC score', metrics.roc_auc_score(y_test, preds))
#Out of sample AUC score 0.816644962171

#Plot AUC for train
preds = clfDT.predict_proba(trainSet)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_train, preds)
plt.plot(fpr,tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive")
plt.title('ROC Plot for Tree-In sample')
plt.show()

#Plot AUC for test
preds = clfDT.predict_proba(testSet)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test, preds)
plt.plot(fpr,tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive")
plt.title('ROC Plot for Tree-Out of sample')
plt.show()




# In[41]:

#Building random forest model on the data
from sklearn.ensemble import RandomForestClassifier
clfRF = RandomForestClassifier(n_estimators=1000, criterion='gini', max_depth=10)
clfRF = clfRF.fit(trainSet, y_train)

#Insample performace
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
ypred=clfRF.predict(trainSet)
Confusionmatrix=confusion_matrix(y_train, ypred)
#print(Confusionmatrix)
accuracy_score(y_train, ypred)
preds = clfRF.predict_proba(trainSet)[:,1]
print('In sample AUC score', metrics.roc_auc_score(y_train, preds))
#In sample AUC score 0.891945369472

#Out of sample performance
ypred=clfRF.predict(testSet)
Confusionmatrix=confusion_matrix(y_test, ypred)
#print(Confusionmatrix)
accuracy_score(y_test, ypred)
preds = clfRF.predict_proba(testSet)[:,1]
print('Out of sample AUC score', metrics.roc_auc_score(y_test, preds))
#Out of sample AUC score 0.821814530253


# In[46]:

# Gradient boosting
from sklearn import ensemble
clf = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate = 0.1, max_depth = 3, verbose = 2)
clf = clf.fit(trainSet, y_train)

#Plotting ROC curve and calculating AUC
preds = clf.predict_proba(trainSet)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_train, preds)
plt.plot(fpr,tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive")
metrics.roc_auc_score(y_train, preds)
plt.title('ROC Plot for Gradient boosting In sample')
plt.show()
#AUC and Misrate Metrics In sample
ypred=clf.predict(trainSet)
Confusionmatrix=confusion_matrix(y_train, ypred)
accuracy_score(y_train, ypred)
metrics.roc_auc_score(y_train, preds)
print('In sample AUC score',metrics.roc_auc_score(y_train, preds) )
#In sample AUC score 0.85794321819

#Plotting ROC curve and calculating AUC
preds = clf.predict_proba(testSet)[:,1]
fpr, tpr, _ = metrics.roc_curve(y_test, preds)
plt.plot(fpr,tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive")
metrics.roc_auc_score(y_test, preds)
plt.title('ROC Plot for Gradient boosting Out sample')
plt.show()
#AUC and Misrate Metrics In sample
ypred=clf.predict(testSet)
Confusionmatrix=confusion_matrix(y_test, ypred)
accuracy_score(y_test, ypred)
metrics.roc_auc_score(y_test, preds)
print('Out of sample AUC score',metrics.roc_auc_score(y_test, preds) )
#Out of sample AUC score 0.83066220908


