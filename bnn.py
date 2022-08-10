import tensorflow as tf
import tensorflow_probability as tfp
import gc
from common import *
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.backend import clear_session

from datetime  import datetime
import os,time

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

    iterations = 10
    
    filename = "bnnActionLog.txt"
    features = "bnnFeatures.txt"
    
    time.sleep(60)
    
    for counter in range(iterations):
        time.sleep(10)

        #os.system("sudo sync")
        #os.system("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'")

    
        with open(filename,"a") as f:
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            f.write(";startTestrun\n")
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            f.write(";startAction;preprocess data\n")
            f.close()
        print("start")


        log = open("model_results_bnn.txt", "w")
        training_df, testing_df = get_train_test_frames()

        # preprocess data
        training_df["text"] = preprocess(training_df["text"])
        testing_df["text"] = preprocess(testing_df["text"])

        with open(filename,"a") as f:
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            f.write(";stopAction\n")
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            f.write(";startAction;numberical embeddings for training\n")
            f.close()

        # generate numerical embeddings for training data
        vectorizer = TfidfVectorizer(min_df=5)
        x_train = vectorizer.fit_transform(training_df["text"].to_numpy()).toarray()
        y_train = training_df["label"].to_numpy()
        print(f"(X, y) train shape: {x_train.shape, y_train.shape}")

        with open(filename,"a") as f:
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            f.write(";stopAction\n")
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            f.write(";startAction;numberical embeddings for testing\n")
            f.close()

        # generate numerical embeddings for testing data
        x_test = vectorizer.transform(testing_df["text"].to_numpy()).toarray()
        y_test = testing_df["label"].to_numpy()
        print(f"(X, y) test shape: {x_test.shape, y_test.shape}")


        with open(filename,"a") as f:
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            f.write(";stopAction\n")
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            f.write(";startAction;delete dataframes\n")
            f.close()

        del training_df, testing_df
        clear_session()
        gc.collect()

        with open(filename,"a") as f:
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            f.write(";stopAction\n")
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            f.write(";startAction;build model\n")
            f.close()
            
        # create model
        model = build_model(
            input_shape=x_train.shape[1], kl_weight=(1.0 / x_train.shape[0])
        )
        
        with open(filename,"a") as f:
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            f.write(";stopAction\n")
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            f.write(";startAction;train model\n")
            f.close()

        history = model.fit(
            x_train,
            y_train,
            epochs=300,
            validation_split=0.1,
        )

        with open(filename,"a") as f:
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            f.write(";stopAction\n")
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            f.write(";startAction;evaluate training\n")
            f.close()

        loss, accuracy = model.evaluate(x_train, y_train)
        # TODO: should get more metrics than just accuracy and loss (sklearn confusion matrix)
        log.write(f"Training accuracy: {accuracy}, Training loss: {loss}\n")


        with open(filename,"a") as f:
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            f.write(";stopAction\n")
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            f.write(";startAction;delete train data\n")
            f.close()

        del x_train, y_train
        clear_session()
        gc.collect()

        with open(filename,"a") as f:
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            f.write(";stopAction\n")
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            f.write(";startAction;evaluate testing\n")
            f.close()

        # evaluate on test data
        loss, accuracy = model.evaluate(x_test, y_test)
        log.write(f"Test accuracy: {accuracy}, Test loss: {loss}\n")

        log.close()

        with open(filename,"a") as f:
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            f.write(";stopAction\n")
            f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            f.write(";stopTestrun\n")
            f.close()

        with open(features,"a") as g:
            g.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3])
            g.write(", ")
            g.write(str(counter))
            g.write(": ")
            g.write(str(model.get_support()))
            g.write("\n")
            g.close()


