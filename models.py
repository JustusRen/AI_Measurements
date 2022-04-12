import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import preprocessing
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.model_selection import train_test_split


def preprocess_data(train, test, vectorizer):
    train_bow = vectorizer.fit_transform(train['text'])
    test_bow = vectorizer.transform(test['text'])
    print("Shape of train matrix : ",train_bow.shape)
    print("Shape of test matrix : ",test_bow.shape) 

    X_tr = train_bow
    y_tr = train['label']
    X_ts = test_bow
    y_ts = test['label']
    return X_tr.array, y_tr.array, X_ts.array, y_ts.array

def train_models(X_tr, y_tr, X_ts, y_ts, models):
    for model in models:
        score = cross_val_score(model, X_tr, y_tr, cv=5)
        msg = ("{0}:\n\tMean accuracy on development set\t= {1:.3f} "
            "(+/- {2:.3f})".format(model.__class__.__name__,
                                    score.mean(),
                                    score.std()))
        print(msg)
        
        model.fit(X_tr, y_tr)
        pred_eval = model.predict(X_ts)
        acc_eval = accuracy_score(y_ts, pred_eval)
        print("\tAccuracy on evaluation set\t\t= {0:.3f}".format(acc_eval))

if __name__ == "__main__":

    df = pd.read_csv('preprocessing/data.csv', encoding='latin')
    train, test = train_test_split(df, test_size=0.995, random_state=1)
    ensamble_models = [RandomForestClassifier(random_state=1),
            GradientBoostingClassifier(random_state=1),
            AdaBoostClassifier(random_state=1)]

    proba_models = [GaussianNB(), 
                   GaussianProcessClassifier(random_state=1)]

    countVectorizer = CountVectorizer()
    X_tr, y_tr, X_ts, y_ts = preprocess_data(train, test, countVectorizer)

    # train_models(X_tr, y_tr, X_ts, y_ts, ensamble_models)
    train_models(X_tr, y_tr, X_ts, y_ts, proba_models)

    #tfidfVectorizer = TfidfVectorizer()
    #X_tr, y_tr, X_ts, y_ts = preprocess_data(train, test, tfidfVectorizer)

    #train_models(X_tr, y_tr, X_ts, y_ts, ensamble_models)
    #train_models(X_tr, y_tr, X_ts, y_ts, proba_models)

    """print(X_tr.shape)
    ann = Sequential()
    ann.add(Input(shape=(X_tr.shape)))
    ann.add(Dense(32, activation='relu'))
    ann.add(Dense(32, activation='relu'))
    ann.add(Dense(1, activation='sigmoid'))

    ann.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    ann.fit(X_tr, y_tr, batch_size=32, epochs=10,
            validation_data=(X_ts, y_ts))"""