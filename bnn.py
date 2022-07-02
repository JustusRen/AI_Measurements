import os
import glob
import tarfile
import string
import time
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import gc
import pandas as pd
from tensorflow import keras
from common import *
from typing import *
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    stopwords.words("english")
except Exception:
    import nltk

    nltk.download("stopwords")


# zero mean, unit variance multivariate normal
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


# variational multivariate normal (learnable means and variances)
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


def remove_stopwords(frame: pd.Series) -> pd.Series:
    for i in range(frame.size):
        document: List = frame.values[i].split()

        for word in stopwords.words("english"):
            try:
                document.remove(word)
            except ValueError:
                continue
        frame.values[i] = " ".join(document)

    return frame


def stem_doc(document: str) -> str:
    words = []
    stemmer = PorterStemmer()

    for word in document.split():
        words.append(stemmer.stem(word))

    return " ".join(words)


def preprocess(frame: pd.Series) -> pd.Series:
    # preprocess data and transform embeddings
    frame = frame.str.lower()
    # remove line breaks
    frame = frame.apply(lambda text: text.replace("<br />", ""))
    # first stopword removal pass
    frame = remove_stopwords(frame)
    # remove punctuation
    frame = frame.apply(
        lambda text: text.translate(str.maketrans("", "", string.punctuation))
    )
    # need two passes to get rid of all stopwords (e.g. stopwords in-between parenthesis)
    frame = remove_stopwords(frame)
    # stem each word in each document
    frame = frame.apply(lambda text: stem_doc(text))
    return frame


def build_model(input_shape, kl_weight):
    # input layer
    inputs = keras.Input(shape=(input_shape,), dtype=tf.float64)
    # normalization/hidden layers
    # TODO: test without batch normalization
    features = keras.layers.BatchNormalization()(inputs)
    # TODO: test different numbers of neurons (probably way more than 8; input size is typically around 100k)
    features = keras.layers.Dense(8, activation="sigmoid")(features)
    features = keras.layers.Dense(8, activation="sigmoid")(features)
    # probabilistic layer(s)
    distribution_params = tfp.layers.DenseVariational(
        units=2, make_prior_fn=prior, make_posterior_fn=posterior, kl_weight=kl_weight
    )(features)
    # output layer
    outputs = tfp.layers.IndependentNormal(1)(distribution_params)
    # final model
    return keras.Model(inputs=inputs, outputs=outputs)


def plot_stats(history):
    # ripped from Justus' code :)
    plt.style.use("ggplot")
    acc = history.history["binary_accuracy"]
    val_acc = history.history["val_binary_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, "b", label="Training acc")
    plt.plot(x, val_acc, "r", label="Validation acc")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, "b", label="Training loss")
    plt.plot(x, val_loss, "r", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()

    plt.savefig(f"figures/training_stats_{time.time_ns()}.png", format="png")


if __name__ == "__main__":
    # force garbage collection
    gc.collect()

    # check for and download, extract dataset if necessary
    if not os.path.exists("aclImdb"):
        if not os.path.exists("aclImdb_v1.tar.gz"):
            files = download_data()
        else:
            files = ["aclImdb_v1.tar.gz"]

        tarfile.open(files[0]).extractall()

    # training data
    if not os.path.exists("train.csv"):
        paths = glob.glob("aclImdb/train/pos/*.txt")
        pos_frame = get_data(paths, 1)
        paths = glob.glob("aclImdb/train/neg/*.txt")
        neg_frame = get_data(paths, 0)

        # combine and shuffle data
        tr_df = pd.concat([pos_frame, neg_frame])
        tr_df = tr_df.sample(frac=1, random_state=RANDOM_SEED)
        tr_df.to_csv("train.csv", columns=["text", "label"])
    else:
        tr_df = pd.read_csv("train.csv", usecols=["text", "label"])

    # preprocess training data
    tr_df["text"] = preprocess(tr_df["text"])
    print(tr_df.head())

    # generate numerical embeddings
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(tr_df["text"].to_numpy()).toarray()
    y_train = tr_df["label"].to_numpy()
    del tr_df

    # training data shapes
    print(f"X train shape: {x_train.shape}")
    print(f"y train shape: {y_train.shape}")

    # create model
    model = build_model(input_shape=x_train.shape[1], kl_weight=(1 / x_train.shape[0]))
    # TODO: mess with learning rate a little bit
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        verbose=True,
        validation_split=0.1,
    )

    # stats from training
    plot_stats(history)
    loss, accuracy = model.evaluate(x_train, y_train)
    with open("model_results.txt", "w+") as log:
        log.write(f"Training accuracy: {accuracy}, Training loss: {loss}")

    # clear training data from memory (so there's room for the testing data)
    del x_train, y_train
    gc.collect()

    # testing data
    if not os.path.exists("test.csv"):
        paths = glob.glob("aclImdb/test/pos/*.txt")
        pos_frame = get_data(paths, 1)
        paths = glob.glob("aclImdb/test/neg/*.txt")
        neg_frame = get_data(paths, 0)

        ts_df = pd.concat([pos_frame, neg_frame])
        ts_df = ts_df.sample(frac=1, random_state=RANDOM_SEED)
        ts_df.to_csv("test.csv", columns=["text", "labels"])
    else:
        ts_df = pd.read_csv("test.csv", usecols=["text", "labels"])

    # preprocess test data
    ts_df["text"] = preprocess(ts_df["text"])
    print(ts_df.head())

    # generate numerical embeddings for test data
    x_test = vectorizer.transform(ts_df["text"].to_numpy()).toarray()
    y_test = ts_df["labels"].to_numpy()
    del ts_df

    # testing data shapes
    print(f"X test shape: {x_test.shape}\ny test shape: {y_test.shape}")

    # evaluate on test data
    loss, accuracy = model.evaluate(x_test, y_test)
    with open("model_results.txt", "w+") as log:
        log.write(f"Test accuracy: {accuracy}, Test loss: {loss}")

    gc.collect()
