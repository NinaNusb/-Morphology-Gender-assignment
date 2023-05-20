import spacy
import pandas as pd
from typing import List, Tuple
from spacy.language import Language
from collections import namedtuple


def filter_dataframe(df: pd.DataFrame)-> pd.DataFrame:
    """
    applies a filter to a df, then splits the df into smaller sub dfs 
    based on 'langauage' and further cleans them by dropping
    andy duplicates found

    returns:
          pd.DataFrame
    """
     
    filtered = df[(~df['noun'].str.contains('-| |\.|1|2|3|4|5|6|7|8|9|0')) & (~df['noun'].str.isupper())]
    dfs = df_lang(filtered) # split by language

    temp = []
    for sub_df in dfs: # for every language
        if sub_df.lang != 'German': # as long as not German
            temp.append(sub_df.df[~sub_df.df['noun'].str.istitle()]) # remove capitalized nouns
        else: # otherwise, for German, pass into SpaCy and filter out Proper nouns
            words = pd.Series(sub_df.df['noun']).tolist()
            nlp = spacy.load("de_core_news_sm") # language model
            text = " ".join(words) 
            nlp.max_length = len(text) # increase the max 
            doc = nlp(text) # create a doc 
            tokens = [token.text for token in doc if token.pos_ != 'PROPN']
            temp.append(sub_df.df[sub_df.df['noun'].isin(tokens)]) 
    return pd.concat(temp) # return new df made of each language df that was filtered


def df_lang(df: pd.DataFrame)-> List[Tuple[str, pd.DataFrame]]:
    """
    Splits main df by 'lang' column, and creates new sub
    dataFrames

    returns:
        list: namedtuple (lang, df)
    """
    Sub_df = namedtuple('Sub_df', ['lang', 'df'])
    languages = df['lang'].unique()
    dataframes = [df[df['lang'] == lang] for lang in languages]
    return [Sub_df(lang, sub_df) for lang, sub_df in zip(languages, dataframes)]