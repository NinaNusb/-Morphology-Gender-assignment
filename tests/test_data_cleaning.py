import pandas as pd
from typing import List, Tuple, NamedTuple
from spacy.language import Language
from collections import namedtuple
from src.data_cleaning import split_df, sub_df_and_model



df = pd.read_json('../../data/dummy_raw.json')



def test_split_df_len():
    res = split_df(df)
    assert len(res) == 4


def test_sub_df_and_model_len():
    res = sub_df_and_model(df)
    assert len(res) == 4