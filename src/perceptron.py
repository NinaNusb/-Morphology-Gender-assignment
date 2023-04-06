import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
datasets = [df[df["lang"] == lang] for lang in set(df["lang"])]

# run perceptron algorithm for predicting gender on each dataset
scores = []
baseline_scores = []
for dataset in datasets:
    X, y = get_data(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = Perceptron(early_stopping=True)
    clf.fit(X_train, y_train)
    scores.append(clf.score(X_test, y_test))


# Plot the accuracies for each dataset
plt.bar(list(set(df["lang"])), scores)
plt.title('Accuracy of Perceptron classifier on different datasets')
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.show()

########################################################################
# TODO 
# compare results with a baseline performance
# a source of inspiration:
# zero rule algorithm for classification
# source: https://machinelearningmastery.com/implement-baseline-machine-learning-algorithms-scratch-python/
def zero_rule_algorithm_classification(train, test):
 output_values = [row[-1] for row in train]
 prediction = max(set(output_values), key=output_values.count)
 predicted = [prediction for i in range(len(test))]
 return predicted