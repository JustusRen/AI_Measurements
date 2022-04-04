import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
from tqdm import tqdm_notebook as tqdm

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier)

from sklearn.metrics import accuracy_score
import warnings 

df = pd.read_csv('preprocessing/data.csv', encoding='latin')

from sklearn.utils import shuffle
df = shuffle(df)

msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

vectorizer = CountVectorizer()
train_bow = vectorizer.fit_transform(train['text'])
test_bow = vectorizer.transform(test['text'])
print("Shape of train matrix after BOW : ",train_bow.shape)
print("Shape of test matrix after BOW : ",test_bow.shape) 

X_tr = train_bow
y_tr = train['label']
X_ts = test_bow
y_ts = test['label']

models = [RandomForestClassifier(random_state=77),
          GradientBoostingClassifier(random_state=77),
          AdaBoostClassifier(random_state=77)]

from sklearn.model_selection import cross_val_score, GridSearchCV

for model in models:
    score = cross_val_score(model, X_tr, y_tr, cv=5)
    msg = ("{0}:\n\tMean accuracy on development set\t= {1:.3f} "
           "(+/- {2:.3f})".format(model.__class__.__name__,
                                  score.mean(),
                                  score.std()))
    print(msg)
    
    # Fit the model on the dev set and predict and eval independent set
    model.fit(X_tr, y_tr)
    pred_eval = model.predict(X_ts)
    acc_eval = accuracy_score(y_ts, pred_eval)
    print("\tAccuracy on evaluation set\t\t= {0:.3f}".format(acc_eval))

vectorizer = TfidfVectorizer()
train_tfidf = vectorizer.fit_transform(train['text'])
test_tfidf = vectorizer.transform(test['text'])
print("Shape of train matrix after Tfidf : ",train_tfidf.shape)
print("Shape of test matrix after Tfidf : ",test_tfidf.shape) 

X_tr = train_tfidf
y_tr = train['label']
X_ts = test_tfidf
y_ts = test['label']


models = [RandomForestClassifier(random_state=77),
          GradientBoostingClassifier(random_state=77),
          AdaBoostClassifier(random_state=77)]

from sklearn.model_selection import cross_val_score, GridSearchCV

for model in models:
    score = cross_val_score(model, X_tr, y_tr, cv=5)
    msg = ("{0}:\n\tMean accuracy on development set\t= {1:.3f} "
           "(+/- {2:.3f})".format(model.__class__.__name__,
                                  score.mean(),
                                  score.std()))
    print(msg)
    
    # Fit the model on the dev set and predict and eval independent set
    model.fit(X_tr, y_tr)
    pred_eval = model.predict(X_ts)
    acc_eval = accuracy_score(y_ts, pred_eval)
    print("\tAccuracy on evaluation set\t\t= {0:.3f}".format(acc_eval))