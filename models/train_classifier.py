import sys

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

def load_data(database_filepath):
    '''loads data from the sqlite db created by process_data.py'''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table(con=engine, table_name='DisasterResponse_table')
    #split df into features and outcomes
    X = df.message
    Y = df[['related', 'request', 'offer',
        'aid_related', 'medical_help', 'medical_products', 'search_and_rescue',
        'security', 'military', 'child_alone', 'water', 'food', 'shelter',
        'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid',
        'infrastructure_related', 'transport', 'buildings', 'electricity',
        'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
        'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
        'other_weather', 'direct_report']]
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    '''normalizes the string variable "text" and returns a list of the words'''
    #normalize
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    #tokenize
    tokens = word_tokenize(text)
    #lemmetize
    lemmetizer = WordNetLemmatizer()
    words = []
    for token in tokens:
        word = lemmetizer.lemmatize(token).strip()
        words.append(word)

    return words


def build_model():
    '''ML Pipeline for message data, grid-searching among options for max features and estimator criteria'''
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

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''Evaluates model, displaying accuracy and other metrics with sklearn's Classification Report'''
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print("Classification Report for {}: \n {}".format(category_names[i],
                                                           classification_report(Y_test.iloc[:,i],
                                                                                 Y_pred[:,i])))

def save_model(model, model_filepath):
    '''saves trained model to pickle file'''
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()