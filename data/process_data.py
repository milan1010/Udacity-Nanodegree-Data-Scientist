import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Reads messages and categories data and merges them into one dataframe"""
    
    # Reads data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge dataframes
    df = pd.merge(messages, categories, on='id', how='outer')
    
    return df


def clean_data(df):
    """ Cleans the data for further use in ML pipeline """
   
    # separates categories into separate columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[0:len(x)-2])
    categories.columns = category_colnames
    
    for column in categories:
        
        # extracts the last character of every string
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].apply(pd.to_numeric)
    
    # Replaces the original categories column with new one, removes duplicate entries
    categories['related'] = categories['related'].replace(2, 1)
    df= df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates(keep='first')
    
    return df


def save_data(df, database_filename):
    """Saves the data to a sqlite database"""
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False, if_exists='replace')
  


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
