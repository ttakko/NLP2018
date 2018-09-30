from __future__ import division
import numpy as np 
import nltk
import textblob as tb 
import glob, re, operator
from tqdm import tqdm
import matplotlib.pyplot as plt
from string import digits
import pandas as pd

dataset = pd.read_csv('VIXCLS.csv')
#print dataset.describe()
Y = [float(i) for i in dataset['VIXCLS'] if i != '.']


df2 = pd.read_csv('abcnews-date-text.csv')
df2= df2.loc[df2['publish_date'] >= 20130900]
dates = []
headlines = []
y_values = []
alldates=[str(i) for i in dataset['DATE']]
for index, row in tqdm(df2.iterrows()):
    date = row['publish_date']
    dates_temp = str(date)[:4]+'-'+str(date)[4:6]+'-'+str(date)[6:]
    if dates_temp in alldates:
        vol =  dataset.loc[dataset['DATE']==dates_temp]['VIXCLS']
        #print float(vol)
        try:
            y_values.append(float(vol))
            dates.append(dates_temp)
            headlines.append(row['headline_text'])
        except: pass
        
result_df = pd.DataFrame({'date':dates,'headline':headlines,'y_value':y_values})
result_df.to_csv('filtered_data.csv', sep=';')
print 'Saved'
#20030219
#2013-10-07

plt.plot(np.arange(0,len(Y)), Y)
plt.ion()
plt.show()
input()