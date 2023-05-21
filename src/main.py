import pandas as pd
from experiments import experiment_1, experiment_2, experiment_3
from data_cleaning import filter_dataframe
from data_transformation import transform, reduce
from visualization import exp_3_visual, exp_1_visual


# Step 1: read the raw data
path = '../data/raw_scraped_data.json'
raw_df = pd.read_json(path) # first dataframe of raw data

# Step 2: filter the data
filtered = filter_dataframe(raw_df) # filter the raw dataframe

# Step 3: transform the data
trans_df = transform(filtered)

# Step 4: Reduce all languages/genders to even amount
reduced_df = reduce(trans_df)

# Step 5: run experiments

# Experiment 1
exp_1 = experiment_1(reduced_df)
# save results
perceptron_results = exp_1['Perceptron']
per_df = pd.DataFrame.from_dict(perceptron_results, orient="index", columns=['Polish', 'German', 'French', 'Spanish'])
per_iterative_df = per_df.applymap(lambda x: x[0]) # grab first element in the tuple (score, langauage)
per_iterative_df.to_csv('../data/perceptron_per_language_results.csv') # save as csv

knn_results = exp_1['KNN']
knn_df = pd.DataFrame.from_dict(knn_results, orient="index", columns=['Polish', 'German', 'French', 'Spanish'])
knn_iterative_df = knn_df.applymap(lambda x: x[0]) # grab first element in the tuple (score, langauage)
knn_iterative_df.to_csv('../data/knn_per_language_results.csv') # save to csv


# Experiment 2
exp_2 = experiment_2(reduced_df, 3)

# save results
knn_shuffle_res = exp_2['KNN']
knn_shuffle_df = pd.DataFrame(knn_shuffle_res)
knn_shuffle_df.to_csv('../data/knn_res_lang_shuffle_3_encoding.csv') # save as csv

percp_shuffle_res = exp_2['Perceptron']
percep_shuffle_df = pd.DataFrame(percp_shuffle_res)
percep_shuffle_df.to_csv('../data/percp_res_lang_shuffle_3_encoding.csv') # save as csv

# Experiment 3
exp_3 = experiment_3(reduced_df, 3)

# save results
whole_df = pd.DataFrame(exp_3)
whole_df.to_csv('../data/whole_3_encoding.csv') # save as csv

# Step 6: visiualize results

exp_1_visual()

print(exp_2)

exp_3_visual(exp_3)

