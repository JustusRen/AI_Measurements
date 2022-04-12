import wget
import glob
import os
import pandas
import string
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from typing import *
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# hidden units for nn
HIDDEN_UNITS = [8, 8]
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
NUM_SAMPLES = 3000
# train/test split factor
SPLIT_FACTOR = 0.5
# train/test batch size
TRAIN_SIZE = int(NUM_SAMPLES * SPLIT_FACTOR)
TEST_SIZE = int(NUM_SAMPLES - TRAIN_SIZE)
# number of prediction samples to take
# (since the network is probabilistic the predictions will be vary for the same
#  sample, need to take multiple predictions to get an accurate result)
PREDICTION_ITERATIONS = 100
# number of epochs to train for
EPOCHS=300


def download_data() -> List[str]:
    files = []

    for url in DATASET_URLS:
        files.append(wget.download(url))

    return files


def create_hybrid_bnn(train_size:int, input_size:int) -> keras.Model:
    def prior(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        prior_model = keras.Sequential(
            [
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.MultivariateNormalDiag(
                        loc=tf.zeros(n), scale_diag=tf.ones(n)
                    )
                )
            ]
        )
        return prior_model
    def posterior(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        posterior_model = keras.Sequential(
            [
                tfp.layers.VariableLayer(
                    tfp.layers.MultivariateNormalTriL.params_size(n),
                    dtype=dtype
                ),
                tfp.layers.MultivariateNormalTriL(n)
            ]
        )
        return posterior_model

    inputs = keras.Input(
        shape=(input_size,), dtype=tf.float64
    )

    # create and normalize features
    features = keras.layers.BatchNormalization()(inputs)

    # deterministic layers
    for unit in HIDDEN_UNITS:
        features = keras.layers.Dense(unit, activation='sigmoid')(features)

    # probabilistic layer
    distribution_params = tfp.layers.DenseVariational(
        units=2,
        make_prior_fn=prior,
        make_posterior_fn=posterior,
        kl_weight=1/train_size
    )(features)

    outputs = tfp.layers.IndependentNormal(1)(distribution_params)

    model = keras.Model(inputs=inputs, outputs=outputs, name='hybrid-bnn')
    return model


if __name__ == '__main__':
    if os.path.exists('True.csv') and os.path.exists('Fake.csv'):
        files = glob.glob('*.csv')
    else:
        files = download_data()

    frames = {}

    for file in files:
        df = pandas.read_csv(file, usecols=['text'])

        if file == 'True.csv':
            df['label'] = tf.ones(shape=df.size, dtype=tf.float64)
        else:
            df['label'] = tf.zeros(shape=df.size, dtype=tf.float64)

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

    # split dataset into train/test
    x = embeddings
    # convert labels to numpy matrix (reshape to (samples x 1))
    y = dataset['label'].to_numpy().reshape(NUM_SAMPLES, 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=SPLIT_FACTOR, random_state=RANDOM_SEED)

    x_train = x_train.reshape(TRAIN_SIZE, feature_size).astype('float64')
    x_test = x_test.reshape(TEST_SIZE, feature_size).astype('float64')

    y_train = y_train.reshape(TRAIN_SIZE, 1).astype('float64')
    y_test = y_test.reshape(TEST_SIZE, 1).astype('float64')

    # create model (train_size = 64 samples) and train
    model = create_hybrid_bnn(TRAIN_SIZE, feature_size)
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.RootMeanSquaredError()]
    )

    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        verbose=1,
        validation_data=tuple([x_test, y_test])
    )

    # evaluate model
    # results = model.evaluate(x_test, y_test)
    # print('test loss, tess acc:', results)

    # predict on 3 test samples
    predictions = []
    for _ in range(PREDICTION_ITERATIONS):
        prediction = model.predict(x_test)
        predictions.append(
            np.array([1 if i[0] > 0 else 0 for i in prediction.tolist()]).reshape(TEST_SIZE, 1)
        )

    predictions = np.concatenate(predictions, axis=1)

    true_predictions = np.count_nonzero(predictions, axis=1).tolist()
    false_predictions = [PREDICTION_ITERATIONS - i for i in true_predictions]
    actual_predictions = [int(y[0]) for y in y_test.tolist()]

    correct_predictions = 0
    for true, false, actual in zip(true_predictions, false_predictions, actual_predictions):
        if true > false and actual == 1:
            correct_predictions += 1

    print(f'Accuracy: {correct_predictions/float(TEST_SIZE)}')
    for i in range(TEST_SIZE):
        print(
            f'Predictions: True: {true_predictions[i]}, '
            f'False: {false_predictions[i]} '
            f'Majority Prediction: {true_predictions[i] > false_predictions[i]} '
            f'- Actual: {bool(actual_predictions[i])}'
        )
