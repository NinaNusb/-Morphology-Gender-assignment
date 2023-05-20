import pandas as pd
from experiments import experiment_1, experiment_2, experiment_3
from data_cleaning import filter_dataframe
from data_transformation import transform, reduce


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
exp_1= experiment_1(reduced_df)

# Experiment 2
exp_2 = experiment_2(reduced_df, 4)


# Experiment 3
exp_3 = experiment_3(reduced_df, 4)


# Step 6: visiualize results
#TODO
print(exp_3)