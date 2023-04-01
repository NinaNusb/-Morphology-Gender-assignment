import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def transform_clean_df(df: pd.DataFrame)-> pd.DataFrame:
    """
    Takes in a clean dataframe, reduces it by 3000 examples per gender,
    per langauge. Takes last 3 letters of each noun in df and one hot
    encodes them, appending results to df

    Returns:
        pd.Dataframe
    """
    reduced_df = df.groupby(['lang', 'gender'])['noun', 'lemma', 'gender', 'lang'].sample(n=3000)
    to_be_encoded = reduced_df['noun'].str[-3:]
    ohe = OneHotEncoder(sparse=False)
    transformed = ohe.fit_transform(to_be_encoded.to_numpy().reshape(-1, 1))
    transformed_df = pd.DataFrame(transformed)
    reduced_df.reset_index(inplace=True, drop=True)
    return pd.concat([reduced_df, transformed_df], axis=1)



