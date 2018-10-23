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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools

time_to_start = 20150500 #filtering the dataframe to be more comprehensible

# Dataframe containing the wanted y-variable and the dates
# Variable should be normalized

df_y = pd.read_csv('normalized_Y.csv', sep=';')
df_y = pd.read_csv('VIXclasslabels3class.csv', sep=';')
#df_y = pd.read_csv('APweeklabels3class.csv', sep=';')



# The dataframe containing the corpus and the dates
df_X = pd.read_csv('filtered_data.csv', sep=';')
#Filtering smaller date

#df_X = df_X[(df_X.date>=time_to_start) & (df_X.date<=int(list(df_y['date'])[-1]))]
#print int(list(df_y['date'])[-1]), int(list(df_X['date'])[-1])
finalx = list(df_X['date'])[len(df_X)-1]

'''
important for single day!!!
'''
df_y = df_y[(df_y.date>=time_to_start) & (df_y.date <= finalx)]
#df_X = df_X[(df_X.date.isin(list(df_y['date'])))]#Taa toimiki :D
'''missingindexes=[]
print len(list(df_X.date)), len(list(df_y.date))
for i in range(len(list(df_X.date))):
    if list(df_X.date)[i] not in list(df_y.date):
        missingindexes.append(i)'''
#print df_X
# The fraction of data to be used in the training 
training_fraction = 0.6

'''
First we have to construct the DxT Matrix, where...
- D = number of unique tokens
- T = number of days in the set
'''

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def const_frequencies(df_X, ngram=3, save=False):
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

def const_matrix(df_X, ngram=1, save=False, threshold=1, pre_dict=False):
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
        np.savetxt('X_matrix2.csv', X=X_matr, delimiter=';')
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

def construct_classifier(df_y,matrix=False, save=True, combine =False):
    from sklearn.neural_network import MLPClassifier  
    if matrix==False:
        X_matrix=const_matrix(df_X, save=True, pre_dict='word_frequencies.csv')
    else:
        X_matrix = np.loadtxt(matrix, delimiter=';')
    if combine!=False:
        newX = []
        newY = []
        for i in range(len(list(df_y['y_value']))):
            if i%combine==0:
                newrow=X_matrix[:,i]
                newval = []
                for y in range(combine-1):
                    for c in range(len(newrow)):
                        newrow[c]=newrow[c]+X_matrix[:,i-y-1][c]
                for y in range(combine):
                    newval.append(list(df_y['y_value'])[i-y])
                newX.append(newrow)
                if np.average(newval)>0:
                    newY.append(1)
                else: newY.append(0)
        #print X_matrix.shape
        X_matrix=np.array(newX).T
        #print X_matrix.shape, len(newY)
        train_len = int(training_fraction*len(newY))
        train_X = X_matrix[:,:train_len]
        train_y = np.array(list(newY)[:train_len])
        test_X = X_matrix[:,train_len:]
        test_y = np.array(list(newY)[train_len:])
        #print train_X.shape, np.array(train_y).shape
        X_train, X_test, y_train, y_test = train_test_split(X_matrix, newY, random_state=0, train_size=0.8)
        clf = MLPClassifier(hidden_layer_sizes=(200, 50,30), max_iter=1000, activation='tanh', solver='lbfgs').fit(train_X.T, train_y)
        y_pred = clf.predict(test_X.T)
    else:
        train_len = int(training_fraction*len(df_y))
        train_X = X_matrix[:,:train_len]
        train_y = list(df_y['y_value'])[:train_len]
        test_X = X_matrix[:,train_len:]
        test_y = list(df_y['y_value'])[train_len:]
        #PASKAA
        '''nX=[]
        print len(missingindexes)
        for i in range(len(X_matrix.T)):
            if i not in missingindexes:
                nX.append(X_matrix[:,i])
        X_matrix = np.array(nX).T
        print X_matrix'''
        #TEST
        print len(list(df_y['y_value'])[:-1]), X_matrix[:,1:].shape
        X_matrix=np.append(X_matrix[:,1:len(list(df_y['y_value']))],[list(df_y['y_value'])[:-1]], axis=0) #Transpose with binned values
        ylst = list(df_y['y_value'])[1:]
        #print train_X.shape, test_X.shape
        unequal = True
        #while unequal:
        train_X, test_X, train_y, test_y = train_test_split(X_matrix.T, ylst, random_state=None, shuffle=True, stratify=ylst, train_size=0.6) #
        print train_X.shape, train_y.count(-1),train_y.count(0),train_y.count(1)
        '''if np.abs(train_y.count(-1)-train_y.count(1))<10:
            unequal=False'''
        clf = MLPClassifier(hidden_layer_sizes=(300, 100,60,20), max_iter=1000, activation='tanh',solver='lbfgs', tol=10**-2).fit(train_X, train_y)
        y_pred = clf.predict(test_X)
    #print len(y_pred), len(test_y), len(test_X), len(train_X)
    evaluation = []
    cnf_matrix = confusion_matrix(test_y, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[-1,0,1],
                        title='Confusion matrix, without normalization')

    for i in range(len(test_y)):
        if test_y[i]==y_pred[i]:
            evaluation.append(1)
        else: evaluation.append(0)
    print 'MSE: '
    print mean_squared_error(test_y, y_pred)
    print 'ACC: '
    print np.average(evaluation)
    if save: 
        #np.savetxt('coefficients2.csv', X=clf.coef_, delimiter=';')
        np.savetxt('prediction2.csv', X=y_pred, delimiter=';')
    if combine!= False: return newY


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

def visualise_prediction(prediction, df_y, new=True):
    train_len = int(training_fraction*len(df_y))
    if new:
        test_y = np.array(list(df_y)[train_len:])
    else:test_y = list(df_y['y_value'])[train_len:]
    preds = np.loadtxt(prediction, delimiter=';')
    rang = np.arange(len(test_y))
    plt.figure('Prediction')
    plt.title('Predicting X')
    plt.plot(rang, test_y, label='True')
    plt.plot(rang, preds, label='Pred')
    plt.legend()
    
        
#const_matrix(df_X,save=True, pre_dict='word_frequencies.csv', threshold=6)

#construct_regressor(matrix='normalizedX.csv',save=True,df_y=df_y)
df_y1=construct_classifier(matrix='X_matrix2.csv',save=True,df_y=df_y, combine=False)
#df_y1=construct_classifier(matrix='weeklyX.csv',save=True,df_y=df_y, combine=False)

#most_important_words('word_frequencies.csv', 'coefficients2.csv')
#visualise_prediction('prediction2.csv', df_y)

'''df_y2=construct_classifier(matrix='normalizedX.csv',save=True,df_y=df_y, combine=9)

#most_important_words('word_frequencies.csv', 'coefficients2.csv')
visualise_prediction('prediction2.csv', df_y2)'''

plt.ion()
plt.show()
input()