import sys
import pandas as pd
import numpy as np
import sqlalchemy
from langdetect import detect, DetectorFactory
import os

###### UTILS #####
def get_encoding(frame, cat_cols):

    '''
    Encodes the columns that are categorical in the dataframe.
    Values in the format '[category]-[value]'
    Changes the column values directly in the 'frame' object. 

    :param frame: Unencoded Disaster Response DataFrame containing the columns to be encoded

    :param cat_cols: List of columns to be encoded
    '''

    for col in frame[cat_cols].columns:

        frame[col] = frame[col].str.split('-').str[1]

        frame[col] = frame[col].astype(int)


def extract_category_names(expanded_cat_frame):
    '''
    Extracts categories names' from the Disaster Response Dataset.
    Values in the format: '[category]-[value]'

    :param expanded_cat_frame: DataFrame containing only the categorical columns

    :return categories_name: List of categories contained in the dataframe
    '''

    for i, value in enumerate(expanded_cat_frame.iloc[0]):

        cat_name = value.split('-')[0]

        if i == 0:

            categories_name = [cat_name]

        else:

            categories_name.append(cat_name)

    return categories_name

DetectorFactory.seed = 123

def infer_lang(txt):

    '''
    Infers text language
    
    :param txt: text for language to be infered

    :return lang: infered language
    '''

    try:

        lang = detect(txt)

    except:

        lang = None

    return lang

remove_double_spacing = lambda x: ' '.join(x.split())

def preprocess_text(df, txt_col):
    '''
    Preprocessing steps for message text

    :param df: DataFrame containing messages' text

    :param txt_col: Column name containing messages' text

    :return df: DataFrame with cleaned messages column
    '''

    re_non_alnum_pattern = r'[^\w\s]'

    re_url_pattern = r'http(s)?:\/\/[\w.-_]+\s'

    cleaned_txt_col_name = f'{txt_col}_clean'

    # Puts an 'url_placeholder' string inplace of URLs
    df[cleaned_txt_col_name] = df[txt_col].str.replace(re_url_pattern, 'url_placeholder ', regex = True)

    df[cleaned_txt_col_name] = df[cleaned_txt_col_name].str.replace(re_non_alnum_pattern,' ',regex = True)

    df[cleaned_txt_col_name] = df[cleaned_txt_col_name].str.lower()

    df[cleaned_txt_col_name] = df[cleaned_txt_col_name].apply(remove_double_spacing)

    df[cleaned_txt_col_name] = df[cleaned_txt_col_name].str.strip()

    return df


####### MAIN PIPELINE ########
def load_data(messages_filepath = 'disaster_messages.csv', categories_filepath = 'disaster_categories.csv'):
    '''
    Loads datasets individually and merges them.

    :param messages_filepath: Path to messages DataFrame

    :param categories_filepath: Path to categories DataFrame

    :return df_raw: Raw original Dataset merged
    '''

    raw_messages = pd.read_csv(messages_filepath)

    raw_categories = pd.read_csv(categories_filepath)

    expanded_categories =  raw_categories['categories'].str.split(';', expand = True)

    # Extracting categories
    categories_name = extract_category_names(expanded_categories)

    expanded_categories.columns = categories_name

    # Encoding columns
    get_encoding(expanded_categories, expanded_categories.columns)

    raw_categories = pd.concat([raw_categories['id'], expanded_categories], axis = 1)

    df_raw = raw_messages.merge(raw_categories,
                                on = 'id',
                                how = 'left')

    return df_raw


def clean_data(df):

    re_not_text = r'^\W+$'

    df.drop_duplicates(inplace = True)

    df['related'].replace(2,1, inplace = True)

    # Removes messages without any alphanumerical characters
    df.drop(df[df['message'].str.match(re_not_text)].index, inplace = True)

    df['infered_language'] = df['message'].apply(infer_lang)

    df = preprocess_text(df, 'message')

    return df


def save_data(df, database_filename = 'DisasterResponse.db'):

    conn = sqlalchemy.create_engine(os.path.join('sqlite:///', database_filename))

    df.to_sql('ANALYTICAL_TABLE', conn, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()