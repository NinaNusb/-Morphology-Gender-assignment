import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple


def encode(df: pd.DataFrame, n=0)-> pd.DataFrame:
    """
    Takes in a clean dataframe, reduces it by 3000 examples per gender,
    per langauge. Takes last 3 letters of each noun in df and one hot
    encodes them, appending results to df

    Returns:
        pd.Dataframe
    """
    to_be_encoded = df['noun'].str[-n:] # grab n amount of letters start from the end to encode only
    ohe = OneHotEncoder(sparse_output=False) # initialize the encoder
    transformed = ohe.fit_transform(to_be_encoded.to_numpy().reshape(-1, 1)) # encode
    transformed_df = pd.DataFrame(transformed) # convert to a dataframe
    df.reset_index(inplace=True, drop=True) # reset indexes
    return pd.concat([df, transformed_df], axis=1) # create new dataframe of reduced df and transformed df


def get_X_y(df: pd.DataFrame)-> Tuple[pd.DataFrame, pd.Series]:
    """
    Takes the 5th to the nth column of a dataframe, which represents a one-hot encoding per row, per word
    and a pd.Series based off gender and returns them both, to be used as X and y respectively.
    """
    return df.iloc[:, 4:], df['gender']


def distribution(df: pd.DataFrame)-> pd.DataFrame:
    """
    Groups values by gender and language, counts and gives a total for each
    Ex:

    lang      French German Polish Spanish
    gender                                
    feminine    8400   4781   5570    3707
    masculine   8424   3722   5491    5590
    neuter         -   3160   5571       -
    """
    return df.groupby(['gender','lang']).size().unstack()

def transform(df):
    max_length = df['noun'].str.len().max() # 38

    def add_filler(word):
        """ preprends n amount #'s to a word 
        based on a max_length"""
        if len(word) < max_length: # if word len() is less than the max len()
            diff = max_length - len(word) # we subtract the current word len() by the max len(): diff
            return '#' * diff + word # we prepend n(diff) amount of '#'s to the word and then return it
        return word # if len() of word is NOT less then the max len, then just return it

    # apply function to every value in column 'noun'
    df['noun'] = df['noun'].apply(add_filler)
    return df

def reduce(df):
    grouped = distribution(df) # get distribution of languages and gender
    lowest_value = int(grouped.min().min()) # select lowest count
    return df.groupby(['lang', 'gender'])['noun', 'gender', 'lang'].sample(n=lowest_value) # reduce each language and gender by lowest_value
