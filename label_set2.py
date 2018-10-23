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


df_y = pd.read_csv('aapl.us.txt', sep=',')
#print df_y['Close']
difference = 0.5
labels = []
Dates = []
for index, row in df_y.iterrows():
    if row['Close']!='.':
        if test_num(df_y.iloc[[index-1]]['Close']):
            if np.abs(float(row['Close'])-float(df_y.iloc[[index-1]]['Close']))<difference:
                labels.append(0)
                Dates.append(row['Date'].replace('-',''))
            else:
                if float(row['Close'])>float(df_y.iloc[[index-1]]['Close']):
                    print 'plus'
                    labels.append(1)
                    Dates.append(row['Date'].replace('-',''))
                else:
                    print 'minus'
                    labels.append(-1)
                    Dates.append(row['Date'].replace('-',''))
        else:
            if np.abs(float(row['Close'])-float(df_y.iloc[[index-2]]['Close']))<difference:
                labels.append(0)
                Dates.append(row['Date'].replace('-',''))
            else:
                if float(row['Close'])>float(df_y.iloc[[index-2]]['Close']):
                    print 'plus'
                    labels.append(1)
                    Dates.append(row['Date'].replace('-',''))
                else:
                    print 'minus'
                    labels.append(-1)
                    Dates.append(row['Date'].replace('-',''))

#np.savetxt('classlabels3class.csv',[labels,Dates], delimiter=';')
pd.DataFrame({'date':Dates,'y_value':labels}).to_csv('APPclasslabels3class.csv', sep=';')

difference=1
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
start=Dates.index('20150501')

Dates = []
indexnum=0
vallist=[]

for i in list(df_y['Close'])[start:]:
    if i=='.':
        vallist.append(np.nan)
    else:
        vallist.append(float(i))
for index in range(len(vallist)):
    if index%5==0:
        #print vallist[index:index+8], vallist[index-7:index+1]
        labels.append(get_label(np.nanmean(vallist[index:index+8]), np.nanmean(vallist[index-7:index])))
        Dates.append(indexnum)
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
pd.DataFrame({'date':Dates[:len(newmat)+1],'y_value':labels[:len(newmat)+1]}).to_csv('APweeklabels3class.csv', sep=';')