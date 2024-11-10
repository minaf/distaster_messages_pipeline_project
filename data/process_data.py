import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    Load data from csv files and merge to a single pandas dataframe
    Input: 
    messages_filepath - filepath to messages csv file
    categories_filepath - filepath to categories sv file
    Returns:
    df - dataframe merging categories and messages
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, left_on='id', right_on='id')
    return df

def clean_data(df):
    '''
    clean_data
    Clean data from the dataframe df
    Input: 
    df - dataframe 
    Returns:
    df - cleaned dataframe 
    '''
    #  Split categories into separate category columns
    categories = df['categories'].str.split(";", expand = True)
    # use the first row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    row = categories.iloc[0]
    category_colnames = row.str.split("-", expand = True)[0].tolist()
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    
    # Replace categories column in df with new category columns
    df = df.drop(columns = ['categories'])
    df = pd.concat([df, categories], axis = 1, join = 'inner')
    
    # Remove duplicates
    df = df[~df.duplicated()]
    # Remove non-binary categories
    df_tmp = df.iloc[:, 4:40].applymap(lambda x: 1 if x==0 or x==1 else 0)
    df.drop(df_tmp[df_tmp.sum(axis=1)!=36].index.tolist(), inplace = True)
    return df


def save_data(df, database_filename):
    '''
    save_data
    Save data from the dataframe df in the database_filename
    Input: 
    df - dataframe 
    Returns:
    database_filename - the filename to save df 
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster', engine, if_exists ='replace', index = False)


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