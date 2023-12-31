import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''loads data from csvs and merges into a single df'''
    #load
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merge
    df = messages.merge(categories, on='id')
    
    return df

def clean_data(df):
    '''cleans data for ML use'''
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # extract a list of new column names for categories
    category_colnames = list(row.apply(lambda row: row[0:-2]))
    # rename the columns of `categories`
    categories.columns = category_colnames
    #convert category variables to 0 and 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda row: row[-1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        #ensure 0 or 1 - note: some entries under "related" were listed as 2 instead of 1
        categories[column] = categories[column].apply(lambda row: 0 if row == 0 else 1)
    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat(objs=(df, categories), axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    '''save dataset into SQLlite db''' 
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse_table', engine, index=False, if_exists='replace')

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