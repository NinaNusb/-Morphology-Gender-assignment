import pandas as pd
from data_cleaning import clean
from sklearn.model_selection import train_test_split



df = pd.read_json('../../data/raw_scraped_data.json')

# sub dfs for each language
spanish_df = df[df['lang'] == 'Spanish']
polish_df = df[df['lang'] == 'Polish']
french_df = df[df['lang'] == 'French']
german_df = df[df['lang'] == 'German']


x = clean(french_df)

# transform df into just noun and gender
new = x.drop('lang', axis=1)

print(new)
# change to X and Y train
train, test = train_test_split(new, test_size=0.2)


print(train)