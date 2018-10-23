from __future__ import division
import numpy as np 
import nltk
from textblob import TextBlob, Word
import glob, re, operator
from tqdm import tqdm
import matplotlib.pyplot as plt
from string import digits
import pandas as pd
import sklearn.linear_model as sklm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def test_num(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


df_y = pd.read_csv('VIXCLS.csv', sep=',')
#print df_y['VIX']
difference = 0.5
labels = []
dates = []
for index, row in df_y.iterrows():
    if row['VIX']!='.':
        if test_num(df_y.iloc[[index-1]]['VIX']):
            if np.abs(float(row['VIX'])-float(df_y.iloc[[index-1]]['VIX']))<difference:
                labels.append(0)
                dates.append(row['DATE'].replace('-',''))
            else:
                if float(row['VIX'])>float(df_y.iloc[[index-1]]['VIX']):
                    print 'plus'
                    labels.append(1)
                    dates.append(row['DATE'].replace('-',''))
                else:
                    print 'minus'
                    labels.append(-1)
                    dates.append(row['DATE'].replace('-',''))
        else:
            if test_num(df_y.iloc[[index-2]]['VIX']):
                if np.abs(float(row['VIX'])-float(df_y.iloc[[index-2]]['VIX']))<difference:
                    labels.append(0)
                    dates.append(row['DATE'].replace('-',''))
                else:
                    if float(row['VIX'])>float(df_y.iloc[[index-2]]['VIX']):
                        print 'plus'
                        labels.append(1)
                        dates.append(row['DATE'].replace('-',''))
                    else:
                        print 'minus'
                        labels.append(-1)
                        dates.append(row['DATE'].replace('-',''))

#np.savetxt('classlabels3class.csv',[labels,dates], delimiter=';')
pd.DataFrame({'date':dates,'y_value':labels}).to_csv('VIXclasslabels3class.csv', sep=';')
'''X_matrix = np.loadtxt('X_matrix2.csv', delimiter=';')
for row in X_matrix:
    row = (row)/(sum(row))
    print(sum(row))
np.savetxt('normalizedX.csv',X_matrix, delimiter=';')'''
difference=0.5
def get_label(new, old):
    dif = new-old
    if np.abs(dif)<difference:
        return 0
    else:
        if dif<0:
            return -1
        else:
            return 1
labels = []
dates = []
indexnum=0
vallist=[]
for i in list(df_y['VIX']):
    if i=='.':
        vallist.append(np.nan)
    else:
        vallist.append(float(i))
for index in range(len(vallist)):
    if index%5==0:
        #print vallist[index:index+8], vallist[index-7:index+1]
        labels.append(get_label(np.nanmean(vallist[index:index+8]), np.nanmean(vallist[index-7:index])))
        dates.append(indexnum)
        indexnum+=1


X_matrix = np.loadtxt('X_matrix2.csv', delimiter=';').T
print X_matrix.shape
print len(vallist), len(labels)
newmat=[]
for row in range(len(X_matrix)):

    tmprow=X_matrix[row]
    if row%5==0:
        for i in range(5):
            tmprow = tmprow+X_matrix[row-i-1]
        newmat.append(tmprow)
print np.array(newmat).shape
np.savetxt('weeklyX.csv',newmat, delimiter=';')
pd.DataFrame({'date':dates[:len(newmat)+1],'y_value':labels[:len(newmat)+1]}).to_csv('VIXweeklabels3class.csv', sep=';')