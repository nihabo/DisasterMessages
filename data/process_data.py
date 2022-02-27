# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """ Loads the Data out of the two files and stores it in one dataframe
    args:
        messages_filepath : string - contains the path for the messages file
        categories_filepath: string - contains the path for the categories file
        
    return:
        df: pd.DataFrame - contains the merged data for messages & categories
    """
    #load messages data
    messages = pd.read_csv("messages.csv")
    messages.head()
    
    # load categories dataset
    categories = pd.read_csv("categories.csv")
    categories.head()
    
    # merge datasets
    df = pd.merge(messages, categories, on="id")
    
    return df


def clean_data(df):
    """ Cleans & Refactors the given data by creating one column per Category
          and setting their value correspondingly either to 0 or 1
    args:
        df: pd.DataFrame - Data to be cleaned up
        
    return:
        df: pd.DataFrame - Cleaned & Refactored Data
    """
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand = True)
    
    # extract the new column names out of the texts within the first row
    categories.columns = categories.iloc[0].apply(lambda x: x[:-2])

    # transform the texts in the category columns to numeric values 0 or 1
    for column in categories:
        # value can be determined from last character of the current value
        categories[column] = categories[column].apply(lambda x: x[-1:]).apply(int)
    
    # drop the original categories column from `df` and replace with the 36 new
    df = df.drop("categories", axis = 1)  
    df = pd.merge(df, categories, left_index = True, right_index = True)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """ Saves the data of DataFrame df to a sqlite database
    args:
        df: pd.DataFrame - Data to stored
        database_filename: string - Name of the database where data should be stored to
    """
    
    # setup engine to connect to database
    engine = create_engine(f'sqlite:///{database_filename}')
    # store data to database
    df.to_sql('Message_Category', engine, index=False)  


def main():
     """ Executing all steps to Load, Clean and Save the data
         all necessary filepaths are given by userinput
     """
  
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