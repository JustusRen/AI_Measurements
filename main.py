import os
from preprocess import *


if __name__ == "__main__":
    # download data from default URIs
    files = []
    if os.path.exists('True.csv') and os.path.exists('Fake.csv'):
        files = ['True.csv', 'Fake.csv']
    else:
        files = download_data()

    # preprocess into data frames (files paths corresponding to data frames)
    data_frames = {path: preprocess(path) for path in files}

    # tf-idf and sbert embeddings
    for frame in data_frames.values():
        frame['tf-idf'] = tfidf(frame['text'])
        frame['sbert'] = sbert(frame['text'])

    # dump processed versions to CSVs
    for path, frame in data_frames.items():
        frame.to_csv(f"processed_{path}")
