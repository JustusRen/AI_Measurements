import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import keras
import gc
from common import *
from sklearn.feature_extraction.text import TfidfVectorizer
from autoencoder import Autoencoder
from keras.backend import clear_session


# zero mean, unit variance multivariate normal
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


# variational multivariate normal (learnable means and variances)
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


def build_model(input_shape, kl_weight=None, units=8):
    # input layer
    inputs = keras.Input(shape=(input_shape,), dtype=tf.float64)
    # normalization/hidden layers
    # TODO: test without batch normalization
    features = tf.keras.layers.BatchNormalization()(inputs)
    # TODO: test different numbers of neurons (probably way more than 8)
    features = tfp.layers.DenseVariational(
        units=units, make_prior_fn=prior, make_posterior_fn=posterior, activation="sigmoid"
    )(features)
    features = tfp.layers.DenseVariational(
        units=units, make_prior_fn=prior, make_posterior_fn=posterior, activation="sigmoid"
    )(features)
    distribution_params = tfp.layers.DenseVariational(
        units=2, make_prior_fn=prior, make_posterior_fn=posterior, activation="sigmoid", # kl_weight=kl_weight
    )(features)
    # output layer
    outputs = tfp.layers.IndependentNormal(1)(distribution_params)
    # final model
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=3e-5),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    return model


if __name__ == "__main__":
    log = open("model_results.txt", "w")
    training_df, testing_df = get_train_test_frames()

    # preprocess data
    training_df["text"] = preprocess(training_df["text"])
    testing_df["text"] = preprocess(testing_df["text"])

    # load autoencoder from saved model
    autoencoder: Autoencoder = tf.keras.models.load_model(
        "models/autoencoder.tf", custom_objects={"Autoencoder": Autoencoder}
    )

    # generate numerical embeddings for training data
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(training_df["text"].to_numpy()).toarray()
    print(f"Embeddings shape: {embeddings.shape}")

    # run training embeddings through autoencoder
    x_train = np.zeros(shape=(embeddings.shape[0], autoencoder.latent_dim))
    y_train = training_df["label"].to_numpy()
    for i in range(x_train.shape[0]):
        inputs = embeddings[i].reshape(1, embeddings.shape[1])
        x_train[i] = autoencoder.encode(inputs)
    print(f"X train shape: {x_train.shape}")

    del training_df, embeddings

    clear_session()
    gc.collect()

    # create model
    # TODO: need to do more work on kl_weight (https://en.wikipedia.org/wiki/Mutual_information, https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
    model = build_model(input_shape=x_train.shape[1])
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        validation_split=0.1,
    )

    loss, accuracy = model.evaluate(x_train, y_train)
    # TODO: should get more metrics than just accuracy and loss (sklearn confusion matrix)
    log.write(f"Training accuracy: {accuracy}, Training loss: {loss}\n")

    # generate numerical embeddings for testing data
    embeddings = vectorizer.transform(testing_df["text"].to_numpy()).toarray()
    # run testing embeddings through autoencoder
    x_test = np.zeros(shape=(embeddings.shape[0], autoencoder.latent_dim))
    y_test = testing_df["label"].to_numpy()
    for i in range(x_test.shape[0]):
        inputs = embeddings[i].reshape(1, embeddings.shape[1])
        x_test[i] = autoencoder.encode(inputs)
    print(f"X test shape: {x_test.shape}\ny test shape: {y_test.shape}")

    del testing_df, embeddings, vectorizer

    clear_session()
    gc.collect()

    # evaluate on test data
    loss, accuracy = model.evaluate(x_test, y_test)
    log.write(f"Test accuracy: {accuracy}, Test loss: {loss}\n")

    log.close()
