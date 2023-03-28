import spacy
import pandas as pd
from typing import List, Tuple
from spacy.language import Language
 


base = '../.../'
filename = '_cleaned_data.json'



def split_df(df: pd.DataFrame)-> List[Tuple[str, pd.DataFrame]]:
    """
    Splits main df by 'lang' column, and creates new sub
    dataFrames

    returns:
        list: tuples (language, sub dataFrame)
    """
    languages = [lang for lang in df['lang'].unique()]
    sub_dfs = [df[df['lang'] == lang] for lang in languages]
    return [(lang, sub_df) for lang, sub_df in zip(languages, sub_dfs)]


def sub_df_and_model(df: pd.DataFrame)-> List[Tuple[pd.DataFrame, Language]]:
    """
    maps sub dataFrames to its appropriate SpaCy language model

    """
    sub_dfs = split_df(df) # list of all sub DataFrames based on language
    d = {'Spanish': 'es', 'French': 'fr', 'German': 'de', 'Polish': 'pl'}
    # list of tuples (language, spacy language model)
    models = [(lang, spacy.load(lang + '_core_news_sm')) for lang in d.values()]
    return [(sub_df[1], model[1]) for sub_df, model in zip(sub_dfs, models)]



def clean_dfs(df: pd.DataFrame):
    """ 
    checks the POS tagging of every noun in a sub dataFrame via SpaCy

    returns:
        dfs(list): sub dataFrames
    """
    dfs_and_models = sub_df_and_model(df)
    dfs = []
    for tup in dfs_and_models: # fore every tuple (sub DataFrame, spacy model)
        res = []
        sub_df, nlp = tup # unpack tuple
        words = pd.Series(sub_df['noun']).tolist() # list of all str values in 'noun'
        for word in words: # for each word in list
            doc = nlp(word) # create a doc via appropriate spacy model for a particular language
            if doc[0].pos_ != 'PROPN': # since just one word is in doc, get its POS, if NOT a Proper Noun
                res.append(word) # then append it to a list
        dfs.append(sub_df[sub_df['noun'].isin(res)]) # create a new sub dataFrame IF 'noun' is in our list of non-proper nouns
    return dfs # return the new list of sub dataFrames


def sub_to_json(sub_dfs: List[pd.DataFrame]):
    """
    takes a list of sub dataFrames and creates a json for each
    the name of the file is modified by the particular language the sub DataFrame
    represents
    """
    for sub in sub_dfs:
        lang = sub.iloc[0][2] # get value at first row, 3 column ('lang'): Ex: Spanish, French, etc
        sub.to_json(base + lang + filename, orient='split')