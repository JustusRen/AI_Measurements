import os
import glob
import pandas
import matplotlib.pyplot as plt
import numpy as np
from transformers import squad_convert_examples_to_features
from bnn import download_data, RANDOM_SEED
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import IncrementalPCA, PCA


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
    dataset_size = dataset.size
    dataset = dataset.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)

    x = dataset['text'].values
    y = dataset['label'].values

    del dataset, frames, files

    # TF-IDF and BOW vectorizers
    tfidf_vectorizer = TfidfVectorizer()
    # count_vectorizer = CountVectorizer()

    # generate embeddings for TF-IDF and BOW
    tfidf_embeddings = tfidf_vectorizer.fit_transform(x).toarray()
    del tfidf_vectorizer
    # bow_embeddings = count_vectorizer.fit_transform(dataset['text'].values)

    # shapes should be the same
    # print('Check that matrix shapes match:', tfidf_embeddings.shape == bow_embeddings.shape)
    # general matrix shape
    # print('BOW/TF-IDF matrix shapes:', bow_embeddings.shape)
    print('TF-IDF matrix shape:', tfidf_embeddings.shape)

    # principle component analysis of both the tf-idf and bow matrices (keep 2 components and plot)
    analyzer = PCA(n_components=COMPONENTS)
    tfidf_pca = analyzer.fit_transform(tfidf_embeddings)
    del analyzer, tfidf_embeddings
    # bow_pca = analyzer.fit_transform(bow_embeddings.toarray())

    print('PCA shape matrix shape:', tfidf_pca.shape)
    # print(bow_pca.shape)
    plt.scatter(tfidf_pca[y == 0, 0], tfidf_pca[y == 0, 1], color='b', s=10, label='False')
    plt.scatter(tfidf_pca[y == 1, 0], tfidf_pca[y == 1, 1], color='r', s=10, label='True')
    plt.title(f'PCA of TF-IDF Vectors (n={SAMPLE_SIZE}, dataset size={dataset_size})')
    plt.legend()
    plt.savefig('scatter_plot_subset.png', format='png')
    # plt.show()
