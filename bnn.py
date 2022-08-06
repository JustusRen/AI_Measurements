import tensorflow as tf
import tensorflow_probability as tfp
import gc
from common import *
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.backend import clear_session


def build_model(input_shape, kl_weight, units=10):
    kl_fn = lambda q, p, _: tfp.distributions.kl_divergence(q, p) * tf.cast(
        kl_weight, dtype=tf.float32
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.BatchNormalization(input_shape=(input_shape,)),
            tfp.layers.DenseFlipout(
                units=units, activation="relu", kernel_divergence_fn=kl_fn
            ),
            tfp.layers.DenseFlipout(
                units=units, activation="relu", kernel_divergence_fn=kl_fn
            ),
            tfp.layers.DenseFlipout(units=2, kernel_divergence_fn=kl_fn),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    return model


if __name__ == "__main__":
    log = open("model_results_bnn.txt", "w")
    training_df, testing_df = get_train_test_frames()

    # preprocess data
    training_df["text"] = preprocess(training_df["text"])
    testing_df["text"] = preprocess(testing_df["text"])

    # generate numerical embeddings for training data
    vectorizer = TfidfVectorizer(min_df=5)
    x_train = vectorizer.fit_transform(training_df["text"].to_numpy()).toarray()
    y_train = training_df["label"].to_numpy()
    print(f"(X, y) train shape: {x_train.shape, y_train.shape}")

    # generate numerical embeddings for testing data
    x_test = vectorizer.transform(testing_df["text"].to_numpy()).toarray()
    y_test = testing_df["label"].to_numpy()
    print(f"(X, y) test shape: {x_test.shape, y_test.shape}")

    del training_df, testing_df
    clear_session()
    gc.collect()

    # create model
    model = build_model(
        input_shape=x_train.shape[1], kl_weight=(1.0 / x_train.shape[0])
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=300,
        validation_split=0.1,
    )

    loss, accuracy = model.evaluate(x_train, y_train)
    # TODO: should get more metrics than just accuracy and loss (sklearn confusion matrix)
    log.write(f"Training accuracy: {accuracy}, Training loss: {loss}\n")

    del x_train, y_train
    clear_session()
    gc.collect()

    # evaluate on test data
    loss, accuracy = model.evaluate(x_test, y_test)
    log.write(f"Test accuracy: {accuracy}, Test loss: {loss}\n")

    log.close()
