import pandas as pd

def label_data(input, output, label):
    df = pd.read_csv(input, sep=',')
    df['label'] = label
    df = df.drop(columns=['title', 'subject', 'date'])
    df.to_csv(output, index=False)

label_data('Fake.csv', 'fake_labeled.csv', 1)
label_data('True.csv', 'true_labeled.csv', 0)