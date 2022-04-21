import pandas as pd
import wget
import string
import numpy as np
from typing import *


# constant random state value for reproducability
RANDOM_SEED = 1

# urls for true/false datasets
DATASET_URLS = [
    # old data; super biased
    # "https://raw.githubusercontent.com/ozzgural/MA-540-TEAM3-DATA/main/input-data/True.csv",
    # "https://raw.githubusercontent.com/ozzgural/MA-540-TEAM3-DATA/main/input-data/Fake.csv"

    # new data
    "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
]

# number of samples to test/train on
NUM_SAMPLES = 5000

# train/test split factor
SPLIT_FACTOR = 0.5

# train/test batch size
TRAIN_SIZE = int(NUM_SAMPLES * SPLIT_FACTOR)
TEST_SIZE = int(NUM_SAMPLES - TRAIN_SIZE)

# number of epochs to train for
EPOCHS = 30


def preprocess_data(dataset: pd.Series, vectorizer) -> np.ndarray:
    dataset = dataset.str.lower()
    dataset = dataset.apply(
        lambda text: text.translate(
            str.maketrans('', '', string.punctuation)
        )
    )

    embeddings = vectorizer.fit_transform(dataset)
    return embeddings


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
        with open(file, 'r') as fin:
            text.append(fin.readlines())

    frame = pd.DataFrame(data=text, columns=['text'])
    frame['label'] = label

    return frame
