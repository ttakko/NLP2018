from __future__ import division
import numpy as np 
import nltk
from textblob import Word
import glob, re, operator
from tqdm import tqdm
import matplotlib.pyplot as plt
from string import digits
import pandas as pd

dataset = pd.read_csv('VIXCLS.csv')
dataset = dataset[dataset.VIXCLS != '.']
#print dataset.describe()
Y = [float(i) for i in dataset['VIXCLS'] if i != '.']
maxvol = float(max(Y))
Y = [float(i)/maxvol for i in Y if i != '.']
Y2 = []
step = 20
for i in range(len(Y)):
    if i >= step:
        Y2.append(sum(Y[i-step:i+step])/(2*step))
    else:
        Y2.append(sum(Y[0:i+step])/(i+step))

'''
Format of resulting file:
date ; tokens ;
'''
alldates_tmp=[str(i) for i in dataset['DATE']]
alldates = []
for i in alldates_tmp:
    alldates.append(int(i.translate(None, '-')))

Y_norm = pd.DataFrame({'date':alldates,'y_value':Y})
Y_norm.to_csv('normalized_Y.csv', sep=';')

df2 = pd.read_csv('abcnews-date-text.csv')

dates = []
headlines = []
y_values = []


index_news = 0
for i in tqdm(alldates):
    headlines_tmp = ''
    
    df_tmp = df2[(df2.publish_date == i)]
    if len(df_tmp)!=0:
        dates.append(i)
        for index, row in df_tmp.iterrows():
            headlines_tmp+=row['headline_text']+'.'
        headlines.append(headlines_tmp)
    else:
        print i
    
result_df = pd.DataFrame({'date':dates,'headline':headlines})
result_df.to_csv('filtered_data.csv', sep=';')
print 'Saved'
#20030219
#2013-10-07

'''
plt.plot(np.arange(0,len(Y)), Y)
plt.plot(np.arange(0,len(Y2)), Y2)
plt.ion()
plt.show()
input()
'''