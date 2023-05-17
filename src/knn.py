import pandas as pd
import sklearn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier

def predict_gender_kNN(X_train, y_train, k, new_input):
    knn= KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    predicted_gender = knn.predict(new_input)
    return predicted_gender

def get_data(df):
    X = df[['noun']].values
    y = df['gender'].values
    return X, y

def encode_nouns(df):
    # encode the "noun" column to numerical values
    le = LabelEncoder()
    df['noun_encoded'] = le.fit_transform(df['noun'])

    # get the features (X) and target (y)
    X = df[['noun_encoded']].values
    y = df['gender'].values
    
    return X, y

def tune_k(X_train, y_train, k_range=range(1, 31)):
    cv_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        cv_scores.append(scores.mean())
    optimal_k = k_range[cv_scores.index(max(cv_scores))]
    return optimal_k

df = pd.read_csv('../data/cleaned_data.csv')
# French data
df_french = df[df['lang'] == 'French']
X_french, y_french = encode_nouns(df_french)
X_train_fr, X_test_fr, y_train_fr, y_test_fr = train_test_split(X_french, y_french, test_size=0.2, random_state=42)
french_predicted_gender = predict_gender_kNN(X_train_fr, y_train_fr, k=3, new_input=X_test_fr)

# German data
df_german = df[df['lang'] == 'German']
X_ger, y_ger = encode_nouns(df_german)
X_train_ger, X_test_ger, y_train_ger, y_test_ger = train_test_split(X_ger, y_ger, test_size=0.2, random_state=42)
german_predicted_gender = predict_gender_kNN(X_train_ger, y_train_ger, k=3, new_input=X_test_ger)

# Polish data
df_polish = df[df['lang'] == 'Polish']
X_polish, y_polish = encode_nouns(df_polish)
X_train_polish, X_test_polish, y_train_polish, y_test_polish = train_test_split(X_polish, y_polish, test_size=0.2, random_state=42)
polish_predicted_gender = predict_gender_kNN(X_train_polish, y_train_polish, k=3, new_input=X_test_polish)

# Spanish data
df_spanish = df[df['lang'] == 'Spanish']
X_sp, y_sp = encode_nouns(df_spanish)
X_train_sp, X_test_sp, y_train_sp, y_test_sp = train_test_split(X_sp, y_sp, test_size=0.2, random_state=42)
spanish_predicted_gender = predict_gender_kNN(X_train_sp, y_train_sp, k=3, new_input=X_test_sp)

accuracy_fr = accuracy_score(y_test_fr, french_predicted_gender)
accuracy_ger = accuracy_score(y_test_ger, german_predicted_gender)
accuracy_pol = accuracy_score(y_test_polish, polish_predicted_gender)
accuracy_sp = accuracy_score(y_test_sp, spanish_predicted_gender)

#############################################################
# Define the datasets and their names
datasets = [df_french, df_german, df_polish, df_spanish]
dataset_names = ['French', 'German', 'Polish', 'Spanish']

# Define the range of k values to try
k_values = range(1, 11)

# Create an empty list to store the accuracies for each dataset
accuracies = []
baseline_scores = []

# Loop over the datasets
for dataset in datasets:
    # Get the data
    X, y = encode_nouns(dataset)
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Get the accuracies for different k values using cross-validation
    cv_scores = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5)
        cv_scores.append(scores.mean())
    # Get the best k value
    best_k = k_values[cv_scores.index(max(cv_scores))]
    # Train a kNN classifier with the best k value
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    # Evaluate the accuracy on the test set
    accuracy = knn.score(X_test, y_test)
    accuracies.append(accuracy)

    # get baseline scores
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)
    baseline_scores.append(dummy_clf.score(dummy_clf.predict(X_test), y_test))


# Plot the accuracies for each dataset
languages = sorted(list(set(df["lang"])))
x = np.arange(len(languages))  # the label locations
width = 0.4  # the width of the bars

plt.bar(x, accuracies, width, color="DarkSlateGray", label="kNN performance")      # plot of model performance
plt.bar(x+width, baseline_scores, width, color="#80b3b3", label="baseline performance")     # plot of baseline performance
plt.title('Accuracy of kNN classifier on different datasets')
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.xticks(x+width/2, languages)
plt.legend()
plt.show()

# Plot the accuracies for each dataset
# plt.bar(dataset_names, accuracies)