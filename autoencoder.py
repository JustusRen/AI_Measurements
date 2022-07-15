import gc
import tensorflow as tf
import keras
from keras.backend import clear_session

class Autoencoder(keras.Model):
    def __init__(self, latent_dim, input_dim):
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = keras.Sequential(
            [
                tf.keras.layers.Dense(latent_dim, input_shape=(input_dim,), activation="relu")
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(input_dim, activation="sigmoid"),
            ]
        )

    def call(self, inputs):
        encoded = self.encoder(inputs)
        return self.decoder(encoded)

    def encode(self, inputs):
        return self.encoder(inputs)

    def get_config(self):
        return {
            "latent_dim": self.latent_dim,
            "input_dim": self.input_dim,
            # "encoder": self.encoder,
            # "decoder": self.decoder,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == "__main__":
    from bnn import preprocess
    from sklearn.feature_extraction.text import TfidfVectorizer
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # training data
    training_df = pd.read_csv("train.csv", usecols=["text"])
    # preprocess training data
    training_df["text"] = preprocess(training_df["text"])
    # print(training_df.head())

    # generate numerical embeddings
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(training_df["text"].to_numpy()).toarray()
    print(x_train.shape)
    del training_df, vectorizer

    # setup autoencoder
    latent_dim = 1000
    ae = Autoencoder(input_dim=x_train.shape[1], latent_dim=latent_dim)
    ae.compile(optimizer="adam", loss="mse")
    ae.fit(x_train, x_train, epochs=15, validation_split=0.1)
    ae.save("models/autoencoder.tf", save_format="tf")

    clear_session()
    gc.collect()

    loss = ae.evaluate(x_train, x_train)
    print(f"Training loss: {loss}")
