import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier


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

# run MLPClassifier for predicting gender on each dataset
accuracies = []
baseline_scores = []

for dataset in datasets:
    X, y = get_data(dataset)
    
    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create model object
    mlp = MLPClassifier(hidden_layer_sizes=(1500,),
                        random_state=42,
                        learning_rate_init=0.01)

    # Fit data onto the model
    mlp.fit(X_train,y_train)

    # make prediction on test set
    accuracies.append(accuracy_score(y_test, mlp.predict(X_test)))

    # get baseline scores
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    baseline_scores.append(dummy_clf.score(dummy_clf.predict(X_test), y_test))


# Plot the accuracies for each dataset
languages = sorted(list(set(df["lang"])))
x = np.arange(len(languages))  # the label locations
width = 0.4  # the width of the bars

plt.bar(x, accuracies, width, color="DarkSlateGray", label="MLP performance")      # plot of model performance
plt.bar(x+width, baseline_scores, width, color="#80b3b3", label="baseline performance")     # plot of baseline performance
plt.title('Accuracy of MLP classifier on different datasets')
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.xticks(x+width/2, languages)
plt.legend()
plt.show()
