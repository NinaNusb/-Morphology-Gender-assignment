import pandas as pd
from data_cleaning import clean_df
from sklearn.model_selection import train_test_split


# read raw data into dataframe
df = pd.read_json('../../data/raw_scraped_data.json')

# clean dataframe
filtered = clean_df(df)





# change to X and Y train
train, test = train_test_split(filtered, test_size=0.2)


print(train)