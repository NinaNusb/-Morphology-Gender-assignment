import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt


def get_data(df):
    X = df['encoded_noun'].values.reshape(-1, 1)
    y = df['gender'].values
    return X, y


# read data
df = pd.read_csv('../data/cleaned_data.csv')

# encode the "noun" column to numerical values
le = LabelEncoder()
df["encoded_noun"] = le.fit_transform(df["noun"])

# separate data by language
datasets = [df[df["lang"] == lang] for lang in sorted(set(df["lang"]))]

# run perceptron algorithm for predicting gender on each dataset
scores = []
baseline_scores = []
for dataset in datasets:
    X, y = get_data(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = Perceptron(random_state=42)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))

    # get baseline scores
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    baseline_scores.append(dummy_clf.score(dummy_clf.predict(X_test), y_test))

print(scores)
print(baseline_scores)

# Plot the accuracies for each dataset
plt.bar(sorted(list(set(df["lang"]))), scores)
plt.title('Accuracy of Perceptron classifier on different datasets')
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.show()

########################################################################
# TODO 
