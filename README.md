# Disaster Response Pipeline Project

Aim of this project is to classify tweets during a disaster into several categories such as water, fire, request so that disaster management team can better handle the situation.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`
    

3. Go to http://0.0.0.0:3001/git 

### Files:
app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py # extracts, transforms and loads the data to be fed into the machine learning pipeline.
|- InsertDatabaseName.db # database to save clean data to
models
|- train_classifier.py # loads the data and applies ML algorithm to train a model for classification of tweets.
|- classifier.pkl # saved model
