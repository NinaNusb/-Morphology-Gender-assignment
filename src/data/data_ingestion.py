import pandas as pd




df = pd.read_json('../../data/raw_scraped_data.json')
spanish_df = df[df['lang'] == 'Spanish']
polish_df = df[df['lang'] == 'Polish']
french_df = df[df['lang'] == 'French']
german_df = df[df['lang'] == 'German']


def remove_uppercase(df):
    return df.loc[df['noun'].str.isupper() == False]

def remove_hypenated_nouns(df):
    return df.loc[df['noun'].str.contains('-') == False]

def remove_nouns_with_space(df):
    return df.loc[df['noun'].str.contains(' ') == False]

def filter_unwanted_nouns(df):
    pass