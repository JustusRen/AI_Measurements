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
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn import metrics


def preprocess_data(train, test, vectorizer):
    train_bow = vectorizer.fit_transform(train['text'])
    test_bow = vectorizer.transform(test['text'])
    print("Shape of train matrix : ",train_bow.shape)
    print("Shape of test matrix : ",test_bow.shape) 

    X_tr = train_bow
    y_tr = train['label']
    X_ts = test_bow
    y_ts = test['label']
    return X_tr, y_tr, X_ts, y_ts

def train_models(X_tr, y_tr, X_ts, y_ts, models):
    for model in models:
        
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_ts)

        # compute the performance measures
        score1 = metrics.accuracy_score(y_ts, y_pred)
        print("accuracy:   %0.3f" % score1)

        print(metrics.classification_report(y_ts, y_pred,
                                                    target_names=['Positive', 'Negative']))

        print("confusion matrix:")
        print(metrics.confusion_matrix(y_ts, y_pred))

if __name__ == "__main__":

    df = pd.read_csv('preprocessing/data.csv', encoding='latin')
    train, test = train_test_split(df, test_size=0.2, random_state=1)
   

    proba_models = [MultinomialNB()]

    countVectorizer = CountVectorizer()
    X_tr, y_tr, X_ts, y_ts = preprocess_data(train, test, countVectorizer)

    train_models(X_tr, y_tr, X_ts, y_ts, proba_models)

    tfidfVectorizer = TfidfVectorizer()
    X_tr, y_tr, X_ts, y_ts = preprocess_data(train, test, tfidfVectorizer)

    train_models(X_tr, y_tr, X_ts, y_ts, proba_models)

    """print(X_tr.shape)
    ann = Sequential()
    ann.add(Input(shape=(X_tr.shape)))
    ann.add(Dense(32, activation='relu'))
    ann.add(Dense(32, activation='relu'))
    ann.add(Dense(1, activation='sigmoid'))

    ann.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    ann.fit(X_tr, y_tr, batch_size=32, epochs=10,
            validation_data=(X_ts, y_ts))"""