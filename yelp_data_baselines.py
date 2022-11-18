import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, RandomTreesEmbedding
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from tokenizers.normalizers import NFD
from common import RANDOM_SEED
from nltk.corpus import stopwords


def preprocess(text: str) -> str:
    punctuation = ',()@#%^&*_-+={}[];:"/?.~`'
    normalizer = NFD()

    text = text.lower()
    text = normalizer.normalize_str(text)
    text = text.replace("<br />", " ")
    text = text.replace("'", "")
    text = text.translate(str.maketrans(" ", " ", punctuation))

    while "  " in text:
        text = text.replace("  ", " ")

    for word in stopwords.words("english"):
        text = text.replace(word, "")

    return text


if __name__ == "__main__":
    # train/test split
    SPLIT_RATIO = 0.8
    # validation set split
    VALID_RATIO = 0.2

    # get trimmed data (see data_analysis.py)
    data = pd.read_csv("dataset.csv", usecols=["text", "label"], encoding="utf-8")
    data = data.dropna()
    data = data.sample(frac=1, random_state=RANDOM_SEED)

    text = data["text"].to_list()
    label = data["label"].to_list()

    # calculation for number of samples in train/test
    total_samples = int(len(text) * 0.05)
    train_size = int(total_samples * SPLIT_RATIO)
    test_size = total_samples - train_size

    # basic stats of data
    print(f"Total samples in dataset: {total_samples}")
    print(f"Samples in train set: {train_size}")
    print(f"Samples in test set: {test_size}")

    # split into train, preprocess text
    x_train = [preprocess(x) for x in text[:train_size]]
    y_train = label[:train_size]
    x_test = [preprocess(x) for x in text[train_size:total_samples]]
    y_test = label[train_size:total_samples]

    # generate numerical embeddings from text
    vectorizer = TfidfVectorizer(min_df=5)

    x_train = vectorizer.fit_transform(x_train).toarray()
    y_train = np.array(y_train)
    x_test = vectorizer.transform(x_test).toarray()
    y_test = np.array(y_test)

    print(f"X train shape: {x_train.shape}")
    print(f"y train shape: {y_train.shape}")
    print(f"X test shape: {x_test.shape}")
    print(f"y test shape: {y_test.shape}")

    ## END TRAIN/TEST DATA

    classes, counts = np.unique(y_train, return_counts=True)
    class_weights = {class_label: count/len(y_train) for class_label, count in zip(classes, counts)}

    ## LOGISTIC REGRESSION

    logistic_regression = LogisticRegression(
        random_state=RANDOM_SEED,
        class_weight=class_weights
    )

    logistic_regression.fit(x_train, y_train)

    train_accuracy = logistic_regression.score(x_train, y_train)
    test_accuracy = logistic_regression.score(x_test, y_test)

    print("Logistic Regression Results:")
    print(f"\tAccuracy on train set: {train_accuracy}")
    print(f"\tAccuracy on test set: {test_accuracy}")

    ## ADABOOST

    adaboost = AdaBoostClassifier(random_state=RANDOM_SEED)
    adaboost.fit(x_train, y_train)

    train_accuracy = adaboost.score(x_train, y_train)
    test_accuracy = adaboost.score(x_test, y_test)

    print("AdaBoost Classifier Results:")
    print(f"\tAccuracy on train set: {train_accuracy}")
    print(f"\tAccuracy on test set: {test_accuracy}")

    ## GRADIENTBOOST

    gradientboost = GradientBoostingClassifier(random_state=RANDOM_SEED)
    gradientboost.fit(x_train, y_train)

    train_accuracy = gradientboost.score(x_train, y_train)
    test_accuracy = gradientboost.score(x_test, y_test)

    print("GradientBoost Classifier Results:")
    print(f"\tAccuracy on train set: {train_accuracy}")
    print(f"\tAccuracy on test set: {test_accuracy}")

    ## RANDOM FOREST

    randomforest = RandomForestClassifier(
        class_weight=class_weights,
        random_state=RANDOM_SEED
    )
    randomforest.fit(x_train, y_train)

    train_accuracy = randomforest.score(x_train, y_train)
    test_accuracy = randomforest.score(x_test, y_test)

    print("RandomForest Classifier Results:")
    print(f"\tAccuracy on train set: {train_accuracy}")
    print(f"\tAccuracy on test set: {test_accuracy}")

    ## MULTINOMIAL NAIVE BAYES
    naivebayes = MultinomialNB()
    naivebayes.fit(x_train, y_train)

    train_accuracy = naivebayes.score(x_train, y_train)
    test_accuracy = naivebayes.score(x_test, y_test)

    print("Multinomial Naive Bayes Classifier Results:")
    print(f"\tAccuracy on train set: {train_accuracy}")
    print(f"\tAccuracy on test set: {test_accuracy}")

    decisiontree = DecisionTreeClassifier(random_state=RANDOM_SEED)
    decisiontree.fit(x_train, y_train)

    train_accuracy = decisiontree.score(x_train, y_train)
    test_accuracy = decisiontree.score(x_test, y_test)

    print("Decision Tree Classifier Results")
    print(f"\tAccuracy on train set: {train_accuracy}")
    print(f"\tAccuracy on test set: {test_accuracy}")

