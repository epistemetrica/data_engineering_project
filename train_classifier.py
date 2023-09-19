#import libraries 
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle

#download NLP references
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data():
    '''loads data from the sqlite db created by process_data.py'''
    engine = create_engine('sqlite:///apw_messages.db')
    df = pd.read_sql_table(con=engine, table_name='messages')
    #split df into features and outcomes
    X = df.message
    y = df[['related', 'request', 'offer',
        'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
        'security', 'military', 'child_alone', 'water', 'food', 'shelter',
        'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
        'infrastructure_related', 'transport', 'buildings', 'electricity',
        'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
        'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
        'other_weather', 'direct_report']]
    
    return X, y

def tokenize(text):
    '''normalizes the string variable "text" and returns a list of the words'''
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    tokens = word_tokenize(text)
    lemmetizer = WordNetLemmatizer()

    words = []
    for token in tokens:
        word = lemmetizer.lemmatize(token).strip()
        words.append(word)

    return words

def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'clf__estimator__max_features': ['sqrt', 'log2'],
        'clf__estimator__criterion': ['gini', 'entropy']
    }
    cv = GridSearchCV(pipeline, param_grid= parameters)

def main():
    #load data
    X, y = load_data()
    #split data
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    #instantiate
    model = build_model()
    #train
    model.fit(X_train, y_train)

    return model

if __name__ == '__main__':
    main()