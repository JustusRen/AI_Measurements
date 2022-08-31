from pprint import pprint
import numpy as np
import tensorflow as tf
from common import *
from transformers import (
    TFAutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)


if __name__ == "__main__":
    training_df, testing_df = get_train_test_frames()
    training_df["text"] = preprocess(training_df["text"])
    testing_df["text"] = preprocess(testing_df["text"])

    tokenizer = AutoTokenizer.from_pretrained("xlnet-large-cased")
    model = TFAutoModelForSequenceClassification.from_pretrained(
        "xlnet-large-cased", num_labels=1
    )

    def training_data_generator():
        for text, label in zip(training_df["text"], training_df["label"]):
            # should have 2 inputs: input ids (from tokenizer) and attention mask
            inputs: tf.Tensor = tokenizer(text, return_tensors="tf").input_ids

            yield inputs, label

    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy"]
    )

    model.fit(training_data_generator(), epochs=10)
