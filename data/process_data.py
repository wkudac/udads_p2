import sys

import numpy as np
import pandas as pd

from sqlalchemy import create_engine, inspect
from sqlalchemy import delete
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import MetaData

# command to start the script  
# python process_data.py disaster_messages.csv disaster_categories.csv DB_disaster_msg.db 

# Constants 
# Name of the messages table
DB_TABLE_NAME = 'Messages' 

def load_data(messages_filepath, categories_filepath):
    """
    read the data from 2 csv files
    input:
    - messages_filepath: file path to the messages file
    - categories_filepath: file path to the categories file
    """
    try:
        # load the 2 data sets
        messages = pd.read_csv(messages_filepath)
        print('Dataset Messages: ', messages.shape)
        categories = pd.read_csv(categories_filepath)
        print('Dateset Categories: ', categories.shape)
    except:
        print("Unable to load files. Check Parameters")
        return pd.DataFrame(), pd.DataFrame()
    
    return messages, categories


def clean_data(p_msg, p_cat):
    """
    make some data preparation like extracting the column names out 
    of the categories file, merge the messages and categories file
    together  
    """
    # select the first row of the categories dataframe
    row = p_cat.iloc[0]
    
    # get the column names for the different categories
    cat_colnames = row.categories.split(';')
    cat_colnames = [col[:-2] for col in cat_colnames]
    print('No. of Column Names for Categories: ', len(cat_colnames))
    
    # split the category values at ';' into the different columns
    p_cat[cat_colnames] = p_cat['categories'].str.split(';',expand=True)
    
    # extract the number for every category values
    for col in p_cat[cat_colnames]:
        # set each value to be the last character of the string
        p_cat[col] = p_cat[col].str.slice(-1).astype(int)
        
    # merge the messages and the categories together based on the id
    df = pd.merge(p_msg, p_cat, how='left', left_on=['id'], right_on=['id'])
    
    # dropping redundant columns
    df.drop(['categories'], axis=1, inplace=True)
    df['related'].replace(to_replace={2: 1}, inplace = True)
    print('Dataset Merged: ', df.shape)

    # rows with a NaN id
    df_nan = df[df['id'].isnull()]
    print('Rows with NaN ids: ', df_nan.shape)
    
    # drop rows with duplicate Id's
    df_dupl = df[df.duplicated(subset=['id'], keep='first')]
    print('Rows with duplicate ids: ', df_dupl.shape)
    df.drop_duplicates(subset=['id'], inplace=True)
    print('Rows after dropping duplicate ids: ', df.shape)
    return df


def save_data(df, database_filename):
    """
    save the data frame data to the database 
    """
    # DB Engine + DB DB_Disaster_Msg.db
    db_engine_url = 'sqlite:///data/' + database_filename
    print('DB File: ', db_engine_url)
    db_engine = create_engine(db_engine_url)
    db_inspector = inspect(db_engine)
    
    # delete the db table if already created before 
    tbl_name = DB_TABLE_NAME
    if db_inspector.has_table(tbl_name):
        db_metadata = MetaData()
        db_metadata.reflect(bind=db_engine)
        db_metadata.drop_all(db_engine, [tbl_name], checkfirst=True)
    
    # write the data frame into the db 
    df.to_sql('Messages', db_engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df_msg, df_cat = load_data(messages_filepath, categories_filepath)
        if df_msg.shape[0] == 0 or df_cat.shape[0] == 0:
            print('No data found')
            return None 
        
        print('Cleaning data...')
        df = clean_data(df_msg, df_cat)
        
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
