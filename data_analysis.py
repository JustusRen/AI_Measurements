"""
average review length
class balance (occurrence of 1-5 star reviews)
"""
import matplotlib.pyplot as plt
import os
import tarfile
import json
import numpy as np
import pandas as pd


if __name__ == "__main__":
    if not os.path.exists("yelp_dataset"):
        dataset = tarfile.open("yelp_dataset.tgz")
        dataset.extractall("yelp_dataset")

    text = []
    text_lengths = []
    labels = []

    plt.style.use("ggplot")

    with open("yelp_dataset/yelp_academic_dataset_review.json", "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = json.loads(line)

            text.append(line["text"])
            text_lengths.append(len(line["text"]))
            labels.append(int(line["stars"]))

    plt.clf(), plt.cla()
    plt.hist(text_lengths, bins=75)
    plt.ylabel("Occurrences")
    plt.xlabel("Review Length")
    plt.title("Number of Occurrences of Review Lengths over entire dataset")
    plt.savefig("figures/occurrences_over_length_hist.svg", format="svg")

    plt.clf(), plt.cla()
    plt.hist(labels)
    plt.ylabel("Occurrences")
    plt.xlabel("Class (stars)")
    plt.title("Class balance of Ratings over entire dataset")
    plt.savefig("figures/class_balance.svg", format="svg")

    df = pd.DataFrame(data={"text": text, "length": text_lengths, "label": labels})
    mps = df["length"].mean() + df["length"].std()
    mms = df["length"].mean() - df["length"].std()
    df = df.where(df["length"] < mps)
    df = df.where(df["length"] > mms)
    df.dropna().to_csv("dataset.csv")

    with open("data_analysis_log.txt", "w") as f:
        f.write(f"Average review length: {np.average(text_lengths)}\n")
        f.write(f"Min/Max lengths: {np.min(text_lengths)}/{np.max(text_lengths)}\n")
        f.write(f"Standard deviation of review length: {np.std(text_lengths)}\n\n")

        f.write(f"Average review rating: {np.average(labels)}\n")
        f.write(f"Median review rating: {np.median(labels)}\n")

        classes, counts = np.unique(labels, return_counts=True)
        f.write(f"Total samples: {len(labels)}\n")
        f.write(f"Unique values (classes): {classes}\n")
        f.write(f"Counts of occurrences: {counts}\n")
        f.write(f"Class weights: {[count/len(labels) for count in counts]}\n")
