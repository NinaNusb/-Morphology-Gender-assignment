import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_transformation import encode, get_X_y
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

from collections import defaultdict

from data_cleaning import df_lang


def experiment_1(df):
    """ encoded, trains and tests on each language individually  """
    results = defaultdict(lambda :defaultdict(list)) # to store scores from ML Models. dict->dict->list
    max_length = df['noun'].str.len().max() # get longest noun in whole dataset
    dfs = df_lang(df) # break df into smalled dfs based on language: spanish_df, french_df etc
    
    for sub_df in dfs: # for each language df
        for n in range(1, max_length + 1): # 
            encoded_df = encode(sub_df.df, n) # encode n amount of letters for the ith langauge df
            X, y = get_X_y(encoded_df) # X is the vector representationf for n amount of letters, y are the labels
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # now that we have a train test split, we can plug them into our ML models
            # KNN
            knn = KNeighborsClassifier(n_neighbors=3) # initialize a KNN class, 3 neighbors
            knn.fit(X_train, y_train) # train it 
            results['KNN'][n].append((knn.score(X_test, y_test), sub_df.lang)) # append score into results dict, along with name of sub df (French, German, etc)

            # Perceptron
            p = Perceptron(random_state=42) # initialize a Perceptron class, random state 42
            p.fit(X_train, y_train) # train it
            results['Perceptron'][n].append((p.score(X_test, y_test), sub_df.lang)) # append score to results dict, along with name of sub df (French, German, etc)

    return results # return s scores of l sub dfs in df, of n amount of letters encode, for c amount of ML models. Ex: results[c][n]: (s,l)


def experiment_2(df, max_length):
    """ encoded as a whole, 
    then trains on one language and tests on another
    """
    results = defaultdict(lambda :defaultdict(lambda: defaultdict()))
    training_data = defaultdict()
    testing_data = defaultdict()
    trans_df = encode(df, max_length) # encode
    dfs = df_lang(trans_df)
    for sub_df in dfs: # for each language
        X, y = get_X_y(sub_df.df) # get X and y
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # split
        training_data[sub_df.lang] = (X_train, y_train) # add the x and y train for each language into a dict
        testing_data[sub_df.lang] = (X_test, y_test) # add the x and y testing for each language into a dict
        
    for i in training_data: # for every language training
        for j in testing_data: # for every language testing
       
            X_train__, y_train__ = training_data[i]
            X_test__, y_test__ = testing_data[j]


            knn = KNeighborsClassifier(n_neighbors=3) # initialize a KNN
            knn.fit(X_train__, y_train__) # fit it with training
            results['KNN'][i][j] = knn.score(X_test__, y_test__) # test it
 
            p = Perceptron(random_state=42) # initialize a Perceptron
            p.fit(X_train__, y_train__) # fit it with training
            results['Perceptron'][i][j] = p.score(X_test__, y_test__) # test it

    return results


def experiment_3(df, n=4):
    """ encodes whole dataframe , trains and tests on self
    as a whole """
    results = {}
    encoded_df = encode(df, n) #  4 letters encoded
    X, y = get_X_y(encoded_df) # X is the vector representationf for n amount of letters, y are the labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # KNN
    knn = KNeighborsClassifier(n_neighbors=3) # initialize a KNN class, 3 neighbors
    knn.fit(X_train, y_train) # train it 
    # knn_score = knn.score(X_test, y_test) # get score
    results['KNN'] = knn.score(X_test, y_test)

    # Perceptron
    p = Perceptron(random_state=42) # initialize a Perceptron class, random state 42
    p.fit(X_train, y_train) # train it
    # p_score = p.score(X_test, y_test) # get score
    results['Perceptron'] = p.score(X_test, y_test)

    # Baseline
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    results['Baseline'] = dummy_clf.score(X_test, y_test)

    return results



