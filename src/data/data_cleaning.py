import spacy
import pandas as pd
from typing import List, Tuple
from spacy.language import Language
from collections import namedtuple



def split_df(df: pd.DataFrame)-> List[Tuple[str, pd.DataFrame]]:
    """
    Splits main df by 'lang' column, and creates new sub
    dataFrames

    returns:
        list: namedtuple (lang, df)
    """
    Sub_df = namedtuple('Sub_df', ['lang', 'df'])
    languages = df['lang'].unique().tolist()
    dataframes = [df[df['lang'] == lang] for lang in languages]
    return [Sub_df(lang, sub_df) for lang, sub_df in zip(languages, dataframes)]


def sub_df_and_model(df: pd.DataFrame)-> List[Tuple[pd.DataFrame, Language]]:
    """
    
    """
    Model = namedtuple('Model', ['lang', 'nlp'])
    Df_nlp = namedtuple('Df_and_Model', ['df', 'nlp'])
    sub_dfs = split_df(df) # list of all sub DataFrames based on language
    d = {'Spanish': 'es', 'French': 'fr', 'German': 'de', 'Polish': 'pl'}
    models = [Model(lang, spacy.load(lang + '_core_news_sm')) for lang in d.values()]
    return [Df_nlp(sub_df.df, model.nlp) for sub_df, model in zip(sub_dfs, models)]


def good_pos_list(tup: Tuple[List[str], Language])-> List[str]:
    """
    takes in a namedtuple, uses the list of nouns stored in tup.words
    and passes each word into SpaCy POS tagger, appending only the nouns
    NOT labeled as Proper Nouns

    returns:
        list: nouns (str)
    """
    nlp = tup.nlp
    text = " ".join(tup.words) # all nouns from list into a str
    nlp.max_length = len(text) # increase the length the parser can handle
    doc = nlp(text) 
    return [token.text for token in doc if token.pos_ != 'PROPN']


def clean_df(df: pd.DataFrame)-> List[pd.DataFrame]:
    """
    takes in a DataFrame, creates sub dataframes based on each unique language,
    then takes each word found in each sub dataframe and passes it into SpaCy
    POS tagger and filters out nouns NOT labeled as Proper Nouns, utlimately
    return a list of sub dataframes complelety populated by nouns in each
    given language.

    returns:
        res(list): list of sub dataframes per language
    """
    Data = namedtuple('Data', ['words', 'nlp'])
    df_and_nlp = sub_df_and_model(df) # sub dataFrames and spacy nlp models
    res = [] # empty list to hold results
    for tup in df_and_nlp: # for every tuple (sub DataFrame, spacy model)
        data = Data(pd.Series(tup.df['noun']).tolist(), tup.nlp) # create a Data namedtuple (list of nouns, specific language model)
        res.append(tup.df[tup.df['noun'].isin(good_pos_list(data))]) # append a sub DataFrame per language with nouns verified as non-proper nouns via SpaCy
    return res # return the new list of sub dataFrames


def sub_to_json(sub_dfs: List[pd.DataFrame])-> None:
    """
    takes a list of sub dataFrames and creates a json for each
    the name of the file is modified by the particular language the sub DataFrame
    represents
    """
    base = '../../data/'
    filename = '_cleaned_data.json'
    for df in sub_dfs:
        lang = df['lang'].unique().tolist()[0] # get only value in column 'lang'
        df.to_json(base + lang + filename, orient='split')

def initial_filter(df):
    return df[(~df['noun'].str.contains('-| |\.|1|2|3|4|5|6|7|8|9|0')) & (~df['noun'].str.isupper())]