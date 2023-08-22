# import libraries
import pandas as pd
from sqlalchemy import create_engine

# load datasets
messages = pd.read_csv('messages.csv')
categories = pd.read_csv('categories.csv')

# merge datasets
df = messages.merge(categories, on='id')

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

# drop the original categories column from `df`
df.drop(columns=['categories'], inplace=True)

# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat(objs=(df, categories))

# drop duplicates
df.drop_duplicates(inplace=True)

# save dataset into SQLlite db 
engine = create_engine('sqlite:///apw_messages.db')
df.to_sql('messages', engine, index=False)