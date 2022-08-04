import os
import pandas as pd
import wget
import matplotlib.pyplot as plt
import tarfile
import glob
import string
from typing import List, Tuple


# constant random state value for reproducibility
RANDOM_SEED = 1

# urls for true/false datasets
DATASET_URLS = ["https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"]

# number of epochs to train for
EPOCHS = 150


def plot_history(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    x = range(1, len(acc) + 1)

    plt.style.use("ggplot")
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


def download_data() -> List[str]:
    files = []

    for url in DATASET_URLS:
        files.append(wget.download(url))

    return files


def label_data(input, output, label):
    df = pd.read_csv(input, sep=",")
    df["label"] = label
    df = df.drop(columns=["title", "subject", "date"])
    df.to_csv(output, index=False)


def get_data(paths: List[str], label: int) -> pd.DataFrame:
    text = []
    for file in paths:
        with open(file, "r", encoding="utf-8") as fin:
            text.append(fin.readlines())

    frame = pd.DataFrame(data=text, columns=["text"])
    frame["label"] = label

    return frame


def preprocess(frame: pd.Series) -> pd.Series:
    # preprocess data and transform embeddings
    frame = frame.str.lower()
    # remove line breaks
    frame = frame.apply(lambda text: text.replace("<br />", " "))  # type: ignore
    # remove punctuation
    frame = frame.str.replace(",", " ", regex=False)
    frame = frame.str.replace("(", " ", regex=False)
    frame = frame.str.replace(")", " ", regex=False)
    frame = frame.str.replace('"', " ", regex=False)
    frame = frame.str.replace(".", " ", regex=False)
    frame = frame.str.replace("-", " ", regex=False)
    frame = frame.str.replace("?", " ", regex=False)

    for i in range(frame.size):
        document: str = frame.values[i]  # type: ignore
        while "  " in document:
            document = document.replace("  ", " ")
        frame.values[i] = document

    return frame


def get_train_test_frames() -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    return training_df, testing_df
