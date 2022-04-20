import pandas as pd
import wget
import string
import numpy as np
from typing import *


# constant random state value for reproducability
RANDOM_SEED = 1

# urls for true/false datasets
DATASET_URLS = [
    "https://raw.githubusercontent.com/ozzgural/MA-540-TEAM3-DATA/main/input-data/True.csv",
    "https://raw.githubusercontent.com/ozzgural/MA-540-TEAM3-DATA/main/input-data/Fake.csv"
]

# number of samples to test/train on
NUM_SAMPLES = 3000

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
