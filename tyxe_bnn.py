from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import *
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import wget
import glob
import pandas
import tyxe
import os
import pyro
import pyro.distributions as dist
import string


# constant random state value for reproducability
RANDOM_SEED = 1

# urls for true/false datasets
DATASET_URLS = [
    "https://raw.githubusercontent.com/ozzgural/MA-540-TEAM3-DATA/main/input-data/True.csv",
    "https://raw.githubusercontent.com/ozzgural/MA-540-TEAM3-DATA/main/input-data/Fake.csv"
]

# sbert embedding dimension(s)
SBERT_DIM = 384

# number of samples to test/train on
NUM_SAMPLES = 100

# train/test split factor
SPLIT_FACTOR = 0.5

# train/test batch size
TRAIN_SIZE = int(NUM_SAMPLES * SPLIT_FACTOR)
TEST_SIZE = int(NUM_SAMPLES - TRAIN_SIZE)

EPOCHS = 30


def download_data() -> List[str]:
    files = []

    for url in DATASET_URLS:
        files.append(wget.download(url))

    return files


if __name__ == "__main__":
    if os.path.exists('True.csv') and os.path.exists('Fake.csv'):
        files = glob.glob('*.csv')
    else:
        files = download_data()

    frames = {}

    for file in files:
        df = pandas.read_csv(file, usecols=['text'])

        if file == 'True.csv':
            df['label'] = torch.ones(size=(df.size, 1))
        else:
            df['label'] = torch.zeros(size=(df.size, 1))

        frames[file] = df

    # TODO: Use NLTK to perform proper NLP
    # combine into one dataset and perform basic preprocessing
    dataset = pandas.concat(frames.values())
    # lowercase all strings
    dataset['text'] = dataset['text'].str.lower()
    # remove punctuation
    dataset['text'] = dataset['text'].apply(lambda text: text.translate(str.maketrans('', '', string.punctuation)))

    # reduce dataset size (just trying to get the model to train)
    dataset = dataset.sample(n=NUM_SAMPLES, random_state=RANDOM_SEED)

    # generate sbert encodings (these are probably yielding bad data atm. TODO)
    # sbert = SentenceTransformer('all-MiniLM-L6-v2')
    # returns a numpy matrix (ndarray) of the numerical embeddings (samples x 384)
    # embeddings = sbert.encode(dataset['text'].array, convert_to_numpy=True)
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(dataset['text'].values).toarray()
    feature_size = embeddings.shape[1]
    print(f'Feature size: {feature_size}')

    # split dataset into train/test
    x = embeddings
    # convert labels to numpy matrix (reshape to (samples x 1))
    y = dataset['label'].to_numpy().reshape(NUM_SAMPLES, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=SPLIT_FACTOR, random_state=RANDOM_SEED)

    x_train = torch.tensor(data=x_train, device=torch.device('cpu'), dtype=torch.double)
    x_test = torch.tensor(data=x_test, device=torch.device('cpu'), dtype=torch.double)
    # x_train = x_train.reshape(TRAIN_SIZE, feature_size).astype('float64')
    # x_test = x_test.reshape(TEST_SIZE, feature_size).astype('float64')

    y_train = torch.tensor(data=y_train, device=torch.device('cpu'), dtype=torch.double)
    y_test = torch.tensor(data=y_test, device=torch.device('cpu'), dtype=torch.double)
    # y_train = y_train.reshape(TRAIN_SIZE, 1).astype('float64')
    # y_test = y_test.reshape(TEST_SIZE, 1).astype('float64')

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Test dataset size: {len(test_dataset)}')

    input('Press enter to continue...')

    # set up data loader for input to the model
    train_loader = torch.utils.data.DataLoader(
        train_dataset, TRAIN_SIZE
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, TEST_SIZE
    )

    # deterministic model
    model = nn.Sequential(
        # nn.BatchNorm1d(num_features=feature_size),
        nn.Linear(in_features=feature_size, out_features=feature_size),
        nn.Sigmoid(),
        nn.Linear(in_features=feature_size, out_features=feature_size),
        nn.Sigmoid()
    )

    # probabilsitic properties of bnn
    prior = tyxe.priors.IIDPrior(dist.Normal(0, 1))
    likelihood = tyxe.likelihoods.HomoskedasticGaussian(scale=0.1, dataset_size=TRAIN_SIZE)
    # combine for bnn
    bnn = tyxe.VariationalBNN(model, prior, likelihood, tyxe.guides.AutoNormal)
    bnn.fit(train_loader, pyro.optim.Adam({'lr': 1e-4}), EPOCHS, device=torch.device('cpu'))
