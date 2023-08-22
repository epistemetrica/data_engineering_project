# Disaster Response Pipeline - Adam Wilson
This project builds a pipeline to categorize messages sent during a disaster event so that the appropriate authorities get relevant messages, and constitutes the Data Engineering project portion of the [Udacity Data Scientist Nanodegree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025). 

The project is graded according to [this rubric](https://learn.udacity.com/nanodegrees/nd025/rubric/1565).

## Project Outline

There are 4 parts to this project:
1. An ETL Pipeline
    - File: process_data.py
    - Cleans data and loads into a SQLite database hosted by Udacity
2. A Machine Learning Pipeline
    - File: train_classifier.py
    - Loads data from SQLite db
    - Processes text, trains and tunes ML model using GridSearchCV
    - Outputs results and exports model as a pickle file
3. A Flask Web App
    - This app is mostly provided by and hosted by Udacity
4. The [github repository](https://github.com/epistemetrica/data_engineering_project)

