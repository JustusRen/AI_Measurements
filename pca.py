import os
import glob
import pandas
import matplotlib.pyplot as plt
import numpy as np
from transformers import squad_convert_examples_to_features
from bnn import download_data, RANDOM_SEED
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import IncrementalPCA, PCA
from sentence_transformers import SentenceTransformer


SAMPLE_SIZE = 3000
COMPONENTS = 2


if __name__ == "__main__":
    if os.path.exists('True.csv') and os.path.exists('Fake.csv'):
        files = glob.glob('*.csv')
    else:
        files = download_data()

    frames = {}
    for file in files:
        frames[file] = pandas.read_csv(file, usecols=['text'])
        if file == 'True.csv':
            frames[file]['label'] = np.ones(frames[file].size)
        else:
            frames[file]['label'] = np.zeros(frames[file].size)

    dataset = pandas.concat(frames.values())

    # TF-IDF and BOW vectorizers
    tfidf_vectorizer = TfidfVectorizer()
    analyzer = PCA(n_components=COMPONENTS)

    figure = plt.figure()
    for i in range(6):
        # subset of the larger dataset to perform principle component analysis on
        # (dataset is too large to process all at once)
        subset = dataset.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED + i)

        # split data and labels into x and y
        x = subset['text'].values
        y = subset['label'].values

        # generate embeddings for TF-IDF
        print(f'Iteration: {i + 1}')
        tfidf_embeddings = tfidf_vectorizer.fit_transform(x).toarray()
        print('TF-IDF matrix shape:', tfidf_embeddings.shape)

        # principle component analysis of the tf-idf matrix/embeddings (keep 2 components and plot)
        tfidf_pca = analyzer.fit_transform(tfidf_embeddings)
        print('PCA matrix shape:', tfidf_pca.shape)

        # plot data for this iteration
        figure.add_subplot(2, 3, i + 1)
        plt.scatter(tfidf_pca[y == 0, 0], tfidf_pca[y == 0, 1], color='b', s=10, label='False')
        plt.scatter(tfidf_pca[y == 1, 0], tfidf_pca[y == 1, 1], color='r', s=10, label='True')
        plt.title(f'PCA of TF-IDF Vectors (n={SAMPLE_SIZE}, iteration={i + 1})')
        plt.legend()

    plt.show()

    bow_vectorizer = CountVectorizer()

    figure = plt.figure()
    for i in range(6):
        # subset of the larger dataset to perform principle component analysis on
        # (dataset is too large to process all at once)
        subset = dataset.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED + i)

        # split data and labels into x and y
        x = subset['text'].values
        y = subset['label'].values

        # generate embeddings for BOW
        print(f'Iteration: {i + 1}')
        bow_embeddings = bow_vectorizer.fit_transform(x).toarray()
        print('BOW matrix shape:', bow_embeddings.shape)

        # principle component analysis of the bow matrix/embeddings (keep 2 components and plot)
        bow_pca = analyzer.fit_transform(bow_embeddings)
        print('PCA matrix shape:', bow_pca.shape)

        # plot data for this iteration
        figure.add_subplot(2, 3, i + 1)
        plt.scatter(bow_pca[y == 0, 0], bow_pca[y == 0, 1], color='b', s=10, label='False')
        plt.scatter(bow_pca[y == 1, 0], bow_pca[y == 1, 1], color='r', s=10, label='True')
        plt.title(f'PCA of BOW Vectors (n={SAMPLE_SIZE}, iteration={i + 1}')
        plt.legend()

    plt.show()

    sbert_vectorizer = SentenceTransformer('all-MiniLM-L6-v2')

    figure = plt.figure()
    for i in range(6):
        # subset of the larger dataset to perform principle component analysis on
        # (dataset is too large to process all at once)
        subset = dataset.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED + i)

        # split data and labels into x and y
        x = subset['text'].values
        y = subset['label'].values

        # generate embeddings for BOW
        print(f'Iteration: {i + 1}')
        sbert_embeddings = sbert_vectorizer.encode(x, convert_to_numpy=True, show_progress_bar=True)
        print('BOW matrix shape:', sbert_embeddings.shape)

        # principle component analysis of the bow matrix/embeddings (keep 2 components and plot)
        sbert_pca = analyzer.fit_transform(sbert_embeddings)
        print('PCA matrix shape:', sbert_pca.shape)

        # plot data for this iteration
        figure.add_subplot(2, 3, i + 1)
        plt.scatter(sbert_pca[y == 0, 0], sbert_pca[y == 0, 1], color='b', s=10, label='False')
        plt.scatter(sbert_pca[y == 1, 0], sbert_pca[y == 1, 1], color='r', s=10, label='True')
        plt.title(f'PCA of BOW Vectors (n={SAMPLE_SIZE}, iteration={i + 1})')
        plt.legend()

    plt.show()
