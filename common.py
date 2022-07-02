import pandas as pd
import wget
import matplotlib.pyplot as plt
from typing import *


# constant random state value for reproducability
RANDOM_SEED = 1

# urls for true/false datasets
DATASET_URLS = [
    "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
]

# number of epochs to train for
EPOCHS = 120


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.style.use('ggplot')
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


def download_data() -> List[str]:
    files = []

    for url in DATASET_URLS:
        files.append(wget.download(url))

    return files


def label_data(input, output, label):
    df = pd.read_csv(input, sep=',')
    df['label'] = label
    df = df.drop(columns=['title', 'subject', 'date'])
    df.to_csv(output, index=False)


def get_data(paths: str, label:int) -> pd.DataFrame:
    text = []
    for file in paths:
        with open(file, 'r', encoding='utf-8') as fin:
            text.append(fin.readlines())

    frame = pd.DataFrame(data=text, columns=['text'])
    frame['label'] = label

    return frame
