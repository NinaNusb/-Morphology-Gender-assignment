import spacy
import pandas as pd
from typing import List, Tuple
from spacy.language import Language
from collections import namedtuple


def split_df(df: pd.DataFrame)-> List[pd.DataFrame]:
    """
    Splits main df by 'lang' column, and creates new sub
    dataFrames

    returns:
        list: pd.DataFrame
    """
    languages = df['lang'].unique().tolist()
    return [df[df['lang'] == lang] for lang in languages]


def initial_filter(df: pd.DataFrame)-> pd.DataFrame:
     """
     applies a filter to a df, then splits the df into smaller sub dfs 
     based on 'langauage' and further cleans them by dropping
     andy duplicates found

     returns:
          pd.DataFrame
     """
     filtered = df[(~df['noun'].str.contains('-| |\.|1|2|3|4|5|6|7|8|9|0')) & (~df['noun'].str.isupper()) & (df['noun'].str.len > 3)]
     dfs = split_df(filtered)
     return pd.concat([df.drop_duplicates(subset="noun", keep=False) for df in dfs])


def df_lang(df: pd.DataFrame)-> List[Tuple[str, pd.DataFrame]]:
    """
    Splits main df by 'lang' column, and creates new sub
    dataFrames

    returns:
        list: namedtuple (lang, df)
    """
    Sub_df = namedtuple('Sub_df', ['lang', 'df'])
    languages = df['lang'].unique()
    dataframes = split_df(df)
    return [Sub_df(lang, sub_df) for lang, sub_df in zip(languages, dataframes)]


def sub_df_and_model(df: pd.DataFrame)-> List[Tuple[pd.DataFrame, Language]]:
    """
    takes in a DataFrame which it then loads sub dataframes and their respect
    SpaCy nlp model based on language. 
    returns:
        list: namedtuple (sub dataframe, nlp model)
    """
    Model = namedtuple('Model', ['lang', 'nlp'])
    Df_nlp = namedtuple('Df_and_Model', ['df', 'nlp'])
    sub_dfs = df_lang(df) 
    d = {'Spanish': 'es', 'French': 'fr', 'German': 'de', 'Polish': 'pl'}
    models = [Model(lang, spacy.load(lang + '_core_news_sm')) for lang in d.values()]
    return [Df_nlp(sub_df.df, model.nlp) for sub_df, model in zip(sub_dfs, models)]


def add_lemma(tup: Tuple[List[str], Language])-> List[str]:
    """
    takes in a namedtuple, uses the list of nouns stored in tup.words
    and passes each word into SpaCy POS tagger, appending only the nouns
    NOT labeled as Proper Nouns

    returns:
        list: nouns (str)
    """
    nlp = tup.nlp 
    text = " ".join(tup.words) 
    nlp.max_length = len(text) 
    doc = nlp(text) 
    data = [(token.text, token.lemma_) for token in doc if token.pos_ != 'PROPN']
    words = [word[0] for word in data]
    lemma_df = pd.DataFrame(data, columns=['noun', 'lemma'])
    filtered_df = tup.df[tup.df['noun'].isin(words)]
    return pd.merge(lemma_df, filtered_df, on=['noun'], how='inner')


def clean_df(df: pd.DataFrame)-> pd.DataFrame:
    """
    takes in a DataFrame, creates sub dataframes based on each unique language,
    then takes each word found in each sub dataframe and passes it into SpaCy
    POS tagger and filters out nouns NOT labeled as Proper Nouns, utlimately
    return a list of sub dataframes complelety populated by nouns in each
    given language.

    returns:
        res(list): list of sub dataframes per language
    """
    Data = namedtuple('Data', ['words', 'nlp', 'df'])
    df_and_nlp = sub_df_and_model(df)
    res = [] 
    for tup in df_and_nlp:
        data = Data(pd.Series(df['noun']).tolist(), tup.nlp, df)
        res.append(add_lemma(data))
    return pd.concat(res)


def raw_json_to_clean_df(path: str)-> pd.DataFrame:
    """
    reads in raw data json file , applies filters, 
    and returns a final and clean df to be used on an NLP model
    
    returns:
        pd.DataFrame
    """
    filtered_df = initial_filter(pd.read_json(path))
    return clean_df(filtered_df)