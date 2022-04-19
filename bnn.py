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
EPOCHS = 30


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
    # outputs = keras.layers.ReLU(max_value=1.0)(outputs)

    model = keras.Model(inputs=inputs, outputs=outputs, name='hybrid-bnn')
    return model


if __name__ == '__main__':
    model = create_hybrid_bnn(TRAIN_SIZE, feature_size)
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.001),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()]
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        verbose=1,
        validation_data=tuple([x_test, y_test])
    )

    # evaluate model
    results = model.evaluate(x_test, y_test)
    print(f'\n\ntest loss: {results[0]}, test accuracy: {results[1]}\n')

    # predict on test samples
    for i, sample in enumerate(x_test[:, :]):
        sample = sample.reshape(1, sample.shape[0])
        predictions = []

        for _ in range(PREDICTION_ITERATIONS):
            predictions.append(model.predict(sample))

        predictions = np.concatenate(predictions, axis=1)
        average = np.average(predictions, axis=1)
        standard_deviation = np.std(predictions, axis=1)

        print(
            f'Sample: {i}\n'
            f'Average: {average[0]}\n'
            f'Standard Deviation: {standard_deviation[0]}\n'
            f'Label: {y_test[i][0]}\n\n'
        )
