import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
from common import RANDOM_SEED
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import IncrementalPCA
from keras.backend import clear_session
from nltk.corpus import stopwords
from tokenizers.normalizers import NFD


def preprocess(text: str) -> str:
    punctuation = ',()@#%^&*_-+={}[];:"/?.~`'
    normalizer = NFD()

    text = text.lower()
    text = normalizer.normalize_str(text)
    text = text.replace("<br />", " ")
    text = text.replace("'", "")
    text = text.translate(str.maketrans(" ", " ", punctuation))

    # extra space removal
    while "  " in text:
        text = text.replace("  ", " ")

    # stopword removal
    # for word in stopwords.words("english"):
    #     text = text.replace(word, "")

    return text


def build_model(input_shape, kl_weight, units=10):
    kl_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) * tf.cast(
        kl_weight, dtype=tf.float32
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.BatchNormalization(input_shape=(input_shape,)),
            tfp.layers.DenseFlipout(
                units=units, activation="relu", kernel_divergence_fn=kl_fn
            ),
            tfp.layers.DenseFlipout(
                units=units, activation="relu", kernel_divergence_fn=kl_fn
            ),
            tfp.layers.DenseFlipout(units=5, activation="softmax", kernel_divergence_fn=kl_fn),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=3e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

    return model


if __name__ == "__main__":
    # train/test split
    SPLIT_RATIO = 0.8
    # validation set split
    VALID_RATIO = 0.2

    # get trimmed data (see data_analysis.py)
    data = pd.read_csv("dataset.csv", usecols=["text", "label"], encoding="utf-8")
    data = data.sample(frac=1, random_state=RANDOM_SEED)

    text = data["text"].to_list()
    labels = data["label"].to_list()

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
    train_labels = labels[:train_size]
    y_train = []
    for i, label in enumerate(train_labels):
        logits = [0, 0, 0, 0, 0]
        logits[int(label) - 1] = 1.
        y_train.append(logits)

    x_test = [preprocess(x) for x in text[train_size:total_samples]]
    test_labels = labels[train_size:total_samples]
    y_test = []
    for i, label in enumerate(test_labels):
        logits = [0, 0, 0, 0, 0]
        logits[int(label) - 1] = 1.
        y_test.append(logits)

    # generate numerical embeddings from text
    vectorizer = TfidfVectorizer(min_df=5)
    pca = IncrementalPCA(n_components=500)

    x_train = vectorizer.fit_transform(x_train).toarray()
    for batch in np.array_split(x_train, 100):
        pca.partial_fit(batch)

    x_train = pca.transform(x_train)
    y_train = np.array(y_train)
    x_test = vectorizer.transform(x_test).toarray()
    x_test = pca.transform(x_test)
    y_test = np.array(y_test)

    print(f"X train shape: {x_train.shape}")
    print(f"y train shape: {y_train.shape}")
    print(f"X test shape:  {x_test.shape}")
    print(f"y test shape:  {y_test.shape}")

    clear_session()
    gc.collect()

    # create model
    model = build_model(
        input_shape=x_train.shape[1], kl_weight=(1.0 / x_train.shape[0])
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=250,
        validation_split=0.2,
    )

    plt.style.use("ggplot")
    fig, ax = plt.subplots(1, 2)
    x = range(0, len(history.history["categorical_accuracy"]))
    ax[0].plot(x, history.history["categorical_accuracy"], label="Training")
    ax[0].plot(x, history.history["val_categorical_accuracy"], label="Validation")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_title("Accuracy Over Time")
    ax[0].legend()
    ax[1].plot(x, history.history["loss"], label="Training")
    ax[1].plot(x, history.history["val_loss"], label="Validation")
    ax[1].set_ylabel("Loss")
    ax[1].set_title("Loss Over Time")
    ax[1].legend()
    fig.supxlabel("Epochs")
    fig.savefig("figures/train_val_stats.svg", format="svg")

    loss, accuracy = model.evaluate(x_train, y_train)
    # TODO: should get more metrics than just accuracy and loss (sklearn confusion matrix)
    print(f"Training accuracy: {accuracy}, Training loss: {loss}\n")

    del x_train, y_train
    clear_session()
    gc.collect()

    # evaluate on test data
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {accuracy}, Test loss: {loss}\n")
