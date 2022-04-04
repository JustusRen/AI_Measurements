import wget
import pandas
import string
import numpy as np
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict
from sentence_transformers import SentenceTransformer


DATASET_URLS = [
    "https://raw.githubusercontent.com/ozzgural/MA-540-TEAM3-DATA/main/input-data/True.csv",
    "https://raw.githubusercontent.com/ozzgural/MA-540-TEAM3-DATA/main/input-data/Fake.csv"
]


def download_data(urls: List[str]=DATASET_URLS) -> List[str]:
    """
    Downloads data sets from URLs specified (defaults to `DATASET_URLS` global list).

    Arguments
    ---------
    `urls`: A list of strings (URLs/URIs) to download datasets from. Defaults to `DATASET_URLS` global list in `preprocess.py`.

    Returns
    -------
    `file_paths`: A list of file paths as strings to the downloaded data.
    """
    file_paths = []

    for url in urls:
        file_paths.append(wget.download(url))

    return file_paths

def preprocess(path: str) -> pandas.DataFrame:
    """
    Processes the specified file into a pandas DataFrame and converts the 'text' fields to lowercase and strips punctuation.

    Arguments
    ---------
    `path`: Path to the file to read.


    Returns
    -------
    `frame`: pandas DataFrame objects created from `path`.
    """
    frame = pandas.read_csv(path, usecols=['text'])
    frame['text'] = frame['text'].str.lower()
    frame['text'] = frame['text'].apply(lambda text: text.translate(str.maketrans('', '', string.punctuation)))

    return frame

def tfidf(series: pandas.Series) -> pandas.Series:
    """
    Converts the given sequence of text data to text frequency-inverse document frequency (TF-IDF) embeddings.

    Arguments
    ---------
    `series`: the pandas Series of the raw data in the data frame (e.g. the 'text' column) to be converted to numerical embeddings.

    Returns
    -------
    `embeddings_series`: csr matrix representation of the input data. Returned in a pandas Series.
    """
    vectorizer = TfidfVectorizer()

    # tf-idf weighted matrix shape: (samples x features) aka (# of documents x argmax(len(words)))
    embeddings = vectorizer.fit_transform(series.array)
    samples = embeddings.shape[0]
    # separate each row in the matrix into its own instance
    row_embeddings = [embeddings.getrow(i) for i in range(samples)]
    embeddings_series = pandas.Series(data=row_embeddings)

    # clean up memory before returning
    del vectorizer, embeddings, samples, row_embeddings
    return embeddings_series


def sbert(series: pandas.Series) -> pandas.Series:
    """
    Generate SBERT embeddings from a series of documents.

    Arguments
    ---------
    `series`: a pandas Series of documents from which the SBERT embeddings will be generated.

    Returns
    -------
    `embeddings`: a pandas Series of the numerical embeddings of `series`
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(series.array)

    del model
    return pandas.Series(data=embeddings)
