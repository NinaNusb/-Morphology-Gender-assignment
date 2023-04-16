import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_recall_fscore_support
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
all_scores = [] 
# baseline_scores = []

for dataset in datasets:
    X, y = get_data(dataset)
    
    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    perceptron = Perceptron(random_state=42)
    
    # tune hyperparameters
    parameters = {"penalty": ("l1", "l2"), "class_weight": (None, "balanced")}
    gs_clf = GridSearchCV(perceptron, parameters)
    gs_clf.fit(X_scaled, y_train)

    # Train the model on the full training set with the best parameters
    best_clf = Perceptron(random_state=42, **gs_clf.best_params_)
    best_clf.fit(X_scaled, y_train)

    # predict scores with different evaluation metrics
    # scores.append(best_clf.score(X_test, y_test))
    y_pred = best_clf.predict(X_test)
    (precision, recall, fscore, support) = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
    scores = {}
    scores["lang"] = max(dataset["lang"])
    scores["accuracy on train"] = best_clf.score(X_scaled, y_train)
    scores["accuracy"] = best_clf.score(X_test, y_test)
    scores["precision"] = precision
    scores["recall"] = recall
    scores["fscore"] = fscore
    scores["support"] = support
    scores["max_cv_score"] = max(cross_val_score(best_clf, X_scaled, y_train, cv=5))

    # visualize results in a pandas DataFrame
    # results_df = pd.DataFrame(gs_clf.cv_results_)
    # columns_to_display = ['params', 'mean_test_score', 'rank_test_score']
    # print(results_df[columns_to_display].to_markdown(index=False))

    # get baseline scores
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    # baseline_scores.append(dummy_clf.score(dummy_clf.predict(X_test), y_test))
    scores["baseline_score"] = dummy_clf.score(dummy_clf.predict(X_test), y_test)

    all_scores.append(scores)

df = pd.DataFrame(all_scores)
print(df)

# Plot the accuracies for each dataset
# plt.bar(sorted(list(set(df["lang"]))), all_scores)
# plt.title('Accuracy of Perceptron classifier on different datasets')
# plt.xlabel('Dataset')
# plt.ylabel('Accuracy')
# plt.show()

########################################################################
# TODO : improve scores