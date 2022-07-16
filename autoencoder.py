import tensorflow as tf
import keras


class Autoencoder(keras.Model):
    input_dim: int
    latent_dim: int

    def __init__(self, latent_dim, input_dim):
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = keras.Sequential(
            [
                tf.keras.layers.Dense(
                    latent_dim, input_shape=(input_dim,), activation="relu"
                )
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
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == "__main__":
    import gc
    from keras.backend import clear_session
    from sklearn.feature_extraction.text import TfidfVectorizer
    from common import preprocess, get_train_test_frames

    # training/testing data frames
    training_df, testing_df = get_train_test_frames()
    # preprocess data
    training_df["text"] = preprocess(training_df["text"])
    testing_df["text"] = preprocess(testing_df["text"])

    # generate numerical embeddings
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(training_df["text"].to_numpy()).toarray()
    print(f"Training data shape: {x_train.shape}")

    # setup autoencoder
    ae = Autoencoder(input_dim=x_train.shape[1], latent_dim=1000)
    ae.compile(optimizer="adam", loss="mse")
    ae.fit(x_train, x_train, epochs=15, validation_split=0.1)
    ae.save("models/autoencoder.tf", save_format="tf")

    del training_df

    clear_session()
    gc.collect()

    loss = ae.evaluate(x_train, x_train)
    print(f"Training loss: {loss}")
    del x_train, loss

    clear_session()
    gc.collect()

    x_test = vectorizer.transform(testing_df["text"].to_numpy()).toarray()
    print(f"Testing data shape: {x_test.shape}")
    loss = ae.evaluate(x_test, x_test)
    print(f"Testing loss: {loss}")
