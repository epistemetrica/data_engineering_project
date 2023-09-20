# Disaster Response Pipeline - Adam Wilson
This project builds a pipeline to categorize messages sent during a disaster event so that the appropriate authorities get relevant messages, and constitutes the Data Engineering project portion of the [Udacity Data Scientist Nanodegree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025). 

The project is graded according to [this rubric](https://learn.udacity.com/nanodegrees/nd025/rubric/1565).

## Project Outline

There are 4 parts to this project:
1. An ETL Pipeline
    - File: process_data.py
    - Cleans data and loads into a SQLite database
2. A Machine Learning Pipeline
    - File: train_classifier.py
    - Loads data from SQLite db
    - Processes text, trains and tunes ML model using GridSearchCV
    - Outputs results and exports model as a pickle file
3. A Flask Web App
    - This app is mostly provided by and hosted by Udacity
4. The [github repository](https://github.com/epistemetrica/data_engineering_project)

## Data

The data consists of two csv files provided by Udacity: categories.csv and messages.csv

## Libraries

The following libraries are used:
- json
- plotly
- pandas
- numpy
- nltk
- sys
- re
- pickle
- WordNetLemmetizer from nltk.stem
- work_tokenizer from nltk.tokenize
- Flask
- render_template, request, and jsonify from Flask
- Bar from plotly.graph_objs
- joblib (note the current version of joblib is a direct import; previous versions are imported from sklearn.externals)
- create_engine from sqlalchemy
- RandomForestClassifier from sklearn.ensemble
- CountVectorizer and TfidfTransformer from sklearn.feature_extraction.text
- Pipeline from sklearn.pipeline
- MultiOutputClassifier from sklearn.multioutput
- train_test_split and GridSearchCV from sklearn.model_selection
- classification_report from sklearn.metrics

## NOTES FOR REVIEWER

I used this repo to work out the code, but finalized all code files in the Udacity Project IDE. The three primary files ran for me in that IDE, but will not run with the files in this repo due to differences in file names. Please let me know if you have questions. Thanks!  