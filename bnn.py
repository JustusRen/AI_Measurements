from enum import auto
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
import nltk
import numpy as np
from tensorflow import keras
from common import *
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from autoencoder import Autoencoder


if not os.path.exists(f"{os.environ['HOME']}/nltk_data/corpora/wordnet.zip"):
    nltk.download("wordnet")
if not os.path.exists(f"{os.environ['HOME']}/nltk_data/corpora/stopwords.zip"):
    nltk.download("stopwords")
if not os.path.exists(f"{os.environ['HOME']}/nltk_data/corpora/omw-1.4.zip"):
    nltk.download("omw-1.4")


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
        document: List = frame.values[i].split()  # type: ignore

        for word in stopwords.words("english"):
            try:
                document.remove(word)
            except ValueError:
                continue
        frame.values[i] = " ".join(document)

    return frame


def lemmatize(document: str) -> str:
    words = []
    lemmatizer = WordNetLemmatizer()

    for word in document.split():
        words.append(lemmatizer.lemmatize(word))

    return " ".join(words)


def preprocess(frame: pd.Series) -> pd.Series:
    # preprocess data and transform embeddings
    frame = frame.str.lower()
    frame = frame.str.strip()
    # remove line breaks
    frame = frame.apply(lambda text: text.replace("<br />", ""))  # type: ignore
    # stem each word in each document
    frame = remove_stopwords(frame)
    frame = frame.apply(lambda text: lemmatize(text))  # type: ignore
    # remove punctuation
    frame = frame.str.replace(",", " ", regex=False)
    frame = frame.str.replace("(", " ", regex=False)
    frame = frame.str.replace(")", " ", regex=False)
    frame = frame.str.replace('"', " ", regex=False)
    frame = frame.str.replace(".", " ", regex=False)
    frame = frame.str.replace("-", " ", regex=False)
    frame = frame.str.replace("!", " ", regex=False)
    frame = frame.str.replace("?", " ", regex=False)

    for i in range(frame.size):
        document: str = frame.values[i] # type: ignore
        while "  " in document:
            document = document.replace("  ", " ")
        frame.values[i] = document

    return frame


def build_model(input_shape, kl_weight):
    # input layer
    inputs = keras.Input(shape=(input_shape,), dtype=tf.float64)
    # normalization/hidden layers
    # TODO: test without batch normalization
    features = keras.layers.BatchNormalization()(inputs)
    # TODO: test different numbers of neurons (probably way more than 8)
    features = tfp.layers.DenseVariational(
        units=8, make_prior_fn=prior, make_posterior_fn=posterior, activation="sigmoid"
    )(features)
    features = tfp.layers.DenseVariational(
        units=8, make_prior_fn=prior, make_posterior_fn=posterior, activation="sigmoid"
    )(features)
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
        training_df = pd.concat([pos_frame, neg_frame])
        training_df = training_df.sample(frac=1, random_state=RANDOM_SEED)
        training_df.to_csv("train.csv", columns=["text", "label"])
    else:
        training_df = pd.read_csv("train.csv", usecols=["text", "label"])

    # preprocess training data
    training_df["text"] = preprocess(training_df["text"])
    # generate numerical embeddings
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(training_df["text"].to_numpy()).toarray()
    y_train = training_df["label"].to_numpy()
    del training_df

    # autoencoder = Autoencoder(1000, embeddings.shape[1])
    # autoencoder.compile(optimizer="adam", loss="mse")
    # print("Training autoencoder")
    # autoencoder.fit(embeddings, embeddings, validation_split=0.1, epochs=5)
    autoencoder = tf.keras.models.load_model(
        "models/autoencoder.tf", custom_objects={"Autoencoder": Autoencoder}
    )

    print(f"Embeddings shape: {embeddings.shape}")

    x_train = np.zeros(shape=(embeddings.shape[0], 1000))
    for i in range(x_train.shape[0]):
        inputs = embeddings[i].reshape(1, embeddings.shape[1])
        x_train[i] = autoencoder.encode(inputs)
    del embeddings

    # training data shapes
    print(f"X train shape: {x_train.shape}")

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

    # testing data
    if not os.path.exists("test.csv"):
        paths = glob.glob("aclImdb/test/pos/*.txt")
        pos_frame = get_data(paths, 1)
        paths = glob.glob("aclImdb/test/neg/*.txt")
        neg_frame = get_data(paths, 0)

        testing_df = pd.concat([pos_frame, neg_frame])
        testing_df = testing_df.sample(frac=1, random_state=RANDOM_SEED)
        testing_df.to_csv("test.csv", columns=["text", "label"])
    else:
        testing_df = pd.read_csv("test.csv", usecols=["text", "label"])

    # preprocess test data
    testing_df["text"] = preprocess(testing_df["text"])
    # generate numerical embeddings for test data
    embeddings = vectorizer.transform(testing_df["text"].to_numpy()).toarray()
    x_test = np.zeros(shape=(embeddings.shape[0], 1000))
    y_test = testing_df["label"].to_numpy()
    del testing_df

    for i in range(x_test.shape[0]):
        inputs = embeddings[i].reshape(1, embeddings.shape[1])
        x_test[i] = autoencoder.encode(inputs)
    del embeddings

    # testing data shapes
    print(f"X test shape: {x_test.shape}\ny test shape: {y_test.shape}")

    # evaluate on test data
    loss, accuracy = model.evaluate(x_test, y_test)
    with open("model_results.txt", "w+") as log:
        log.write(f"Test accuracy: {accuracy}, Test loss: {loss}")
