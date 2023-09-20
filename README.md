# Disaster Response Pipeline - Adam Wilson
This project builds a pipeline to categorize messages sent during a disaster event so that the appropriate authorities get relevant messages, and constitutes the Data Engineering project portion of the [Udacity Data Scientist Nanodegree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025). 

The project is graded according to [this rubric](https://learn.udacity.com/nanodegrees/nd025/rubric/1565).

## Project Outline

There are 4 parts to this project:
1. An ETL Pipeline
    - File: data/process_data.py
    - Cleans data and loads into a SQLite database
2. A Machine Learning Pipeline
    - File: models/train_classifier.py
    - Loads data from SQLite db
    - Processes text, trains and tunes ML model using GridSearchCV
    - Outputs results and exports model as a pickle file
3. A Flask Web App
    - File: app/run.py
    - This app is mostly provided by and hosted by Udacity
4. The [github repository](https://github.com/epistemetrica/data_engineering_project), including this README.md file. 

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

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage. NOTE: The Udacity Project IDE includes a 'preview' button; I do not know how to launch the web app locally, so I will assume the reviewer has access to this 'preview' button in their environment.  