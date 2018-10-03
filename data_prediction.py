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

time_to_start = 20150500 #filtering the dataframe to be more comprehensible

# Dataframe containing the wanted y-variable and the dates
# Variable should be normalized

df_y = pd.read_csv('normalized_Y.csv', sep=';')
# The dataframe containing the corpus and the dates
df_X = pd.read_csv('filtered_data.csv', sep=';')
#Filtering smaller date
df_X = df_X[df_X.date>=time_to_start]
finalx = list(df_X['date'])[len(df_X)-1]
df_y = df_y[(df_y.date>=time_to_start) & (df_y.date <= finalx)]
#print df_X
# The fraction of data to be used in the training 
training_fraction = 0.8

'''
First we have to construct the DxT Matrix, where...
- D = number of unique tokens
- T = number of days in the set
'''
def const_frequencies(df_X, ngram=1, save=False):
    unique_tokens = {}
    full_len = len(df_X)
    print 'Calculating the unique tokens...'
    for index, row in df_X.iterrows():
        print index, '/', full_len
        tmp_row = row['headline'].split('.')
        for sentence in tmp_row:
            sent_words = TextBlob(sentence)
            for i in sent_words.words.lemmatize():
                if i not in unique_tokens.keys():
                    unique_tokens[i]=1
                else: unique_tokens[i]=unique_tokens[i]+1
    #Just in case save the dict
    if save:
        pd.DataFrame({'word':unique_tokens.keys(),'frequency':unique_tokens.values()}).to_csv('word_frequencies.csv', sep=';')
    return unique_tokens

def const_matrix(df_X, ngram=1, save=False, threshold=20, pre_dict=False):
    #Calculating the nummber of unique and frequent tokens
    if pre_dict == False:
        unique_tokens = const_frequencies(df_X=df_X,ngram=ngram, save=save)
    else:
        unique_tokens_df = pd.read_csv(pre_dict, sep=';')
        unique_tokens = dict(zip(unique_tokens_df['word'], unique_tokens_df['frequency']))
    #Filtering the least frequent tokens out
    filtered_tokens = []
    for key, value in unique_tokens.iteritems():
        if value>=threshold:
            filtered_tokens.append(key)
    # Empty DxT matrix
    print 'Filtered the tokens, proceeding to filling the matrix...'
    X_matr = np.zeros([len(filtered_tokens), len(df_X)])
    realindex=0
    for index, row in df_X.iterrows():
        print realindex, '/', len(df_X)
        tmp_row = row['headline'].split('.')
        for sentence in tmp_row:
            sent_words = TextBlob(sentence)
            for i in sent_words.words.lemmatize():
                if i in filtered_tokens:
                    word_index = filtered_tokens.index(i)
                    X_matr[word_index][realindex]+=1
        realindex+=1
    #Save the whole thing if param save = True
    if save:
        print 'Matrix is saved'
        np.savetxt('X_matrix.csv', X=X_matr, delimiter=';')
    return X_matr
                

def construct_regressor(df_y,matrix=False, save=True):
    if matrix==False:
        X_matrix=const_matrix(df_X, save=True, pre_dict='word_frequencies.csv')
    else:
        X_matrix = np.loadtxt(matrix, delimiter=';')
    train_len = int(training_fraction*len(df_y))
    train_X = X_matrix[:,:train_len]
    train_y = list(df_y['y_value'])[:train_len]
    test_X = X_matrix[:,train_len:]
    test_y = list(df_y['y_value'])[train_len:]
    
    #print train_X.shape, test_X.shape
    clf = sklm.LinearRegression().fit(train_X.T, train_y)
    y_pred = clf.predict(test_X.T)
    #print len(y_pred), len(test_y), len(test_X), len(train_X)
    print 'MSE: '
    print mean_squared_error(test_y, y_pred)
    if save: 
        np.savetxt('coefficients.csv', X=clf.coef_, delimiter=';')
        np.savetxt('prediction.csv', X=y_pred, delimiter=';')


def most_important_words(word_dict, coefficients):
    words = list(pd.read_csv(word_dict, sep=';')['word'])
    coefficients = list(np.loadtxt(coefficients, delimiter=';'))
    fulldict = dict(zip(words,coefficients))
    sorted_dict = sorted(fulldict.items(), key=operator.itemgetter(1))
    indx = 0
    for key, value in sorted_dict:
        if indx>100:
            break
        print key, value
        indx +=1

def visualise_prediction(prediction, df_y):
    train_len = int(training_fraction*len(df_y))
    test_y = list(df_y['y_value'])[train_len:]
    preds = np.loadtxt(prediction, delimiter=';')
    rang = np.arange(len(test_y))
    plt.figure('Prediction')
    plt.title('Predicting X')
    plt.plot(rang, test_y, label='True')
    plt.plot(rang, preds, label='Pred')
    plt.legend()
    plt.ion()
    plt.show()
    input()
        


#construct_regressor(matrix='X_matrix.csv',save=True,df_y=df_y)

most_important_words('word_frequencies.csv', 'coefficients.csv')
visualise_prediction('prediction.csv', df_y)