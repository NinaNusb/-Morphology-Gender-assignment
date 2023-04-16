import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple


def transform(df: pd.DataFrame)-> pd.DataFrame:
    """
    Takes in a clean dataframe, reduces it by 3000 examples per gender,
    per langauge. Takes last 3 letters of each noun in df and one hot
    encodes them, appending results to df

    Returns:
        pd.Dataframe
    """
    lowest = distribution(df).min()[0]
    reduced_df = df.groupby(['lang', 'gender'])['noun', 'lemma', 'gender', 'lang'].sample(n=lowest)
    to_be_encoded = reduced_df['noun'].str[-4:]
    ohe = OneHotEncoder(sparse=False)
    transformed = ohe.fit_transform(to_be_encoded.to_numpy().reshape(-1, 1))
    transformed_df = pd.DataFrame(transformed)
    reduced_df.reset_index(inplace=True, drop=True)
    return pd.concat([reduced_df, transformed_df], axis=1)



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
    return df.groupby('gender')['lang'].value_counts().unstack(fill_value='-')