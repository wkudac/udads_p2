# Disaster Response 

## Introduction

The application uses Supervised Machine Learning techniques to categorize a text snippet in 36 predefined categories. 

The base of all is a prepared csv file where a text is already labelled in categories. The text is extracted from messages about disasters in the world. 

## Steps
There are 3 major parts in the project: 
- ETL (Extract, Transform, Load)
  After reading the data from the csv files, the data is prepared in some kind and then stored again in a Sqlite database. 
- Text Processing with Machine Learning 
  The data from the Sqlite database from the 1st part, is used to build and train a prepared Natural Language Processing (NLP) model. 
  The trained model is saved as a special file which can later be used to propose the category of new text messages.
- Web App based on Bootstrap
  A web app created with the Bootstrap web framework offers the possibility to make a proposal for a category for a new message text. 
  Some dashboards offer additional statistics about the used data.

## Project Motivation
The web app should allow employees in crisis management centres to categorize incoming messages about disastera quicker which then can be used to trigger the activation of specialized helper organizations. 
That is definitely not possible with the current app version, but just to set a goal. 

## File Descriptions
2 Files are given:
- disaster_message.csv
  Contains the text messages about a disaster with a given ID and a genre assignment
- disaster_categories.csv 
  Contains for every ID used in the disaster_messages.csv file an assignment to 36 categories used to classify the problem. 
  The categories which are relevant for the message are labeled with 1 else with 0. 
  
## How is the Source Code organized
- Folder Data
  Contains the csv files and the python script process_data.py to load the data from the csv files and 
  dump the prepared and cleaned data in a Sqlite database.
- Folder Model
  With the python script train_classifier.py the data is read out of the Sqlite database and used to train a NLP model. 
  The model is then exported to a pickle file.
- Folder App
  The web application run.py is the Flask based web application which reads data from the Sqlite database for reporting and
  the NLP model stored in the pickle file to make predictions about the category out of new disaster messages entered in the web app. 
  
## How to run everthing
- Start the ETL proces 
  In folder DATA start the python script PROCESS_DATA.PY to create the Sqlite database.
  -- Switch to folder DATA 
  -- Execute "python process_data.py disaster_messages.csv disaster_categories.csv DB_disaster_Msg.db"
     Parameters: 
     1. Name of the messages csv file
     2. Name of the categories csv file
     3. Name of the Sqllite database
- Start the ML process
  In folder MODEL start the python script TRAIN_CLASSIFIER.PY 
  -- Switch to folder MODEL
  -- Execute "python train_classifier.py DB_disaster_msg.db ML_disaster_msg.pkl
     Parameters:
     1. Name of the Sqlite database created in the ETL step. 
     2. Name of the ML model pickle file to which the ML model is exported

- Start the Flask based web application 
  In folder APP start the python script 
  - Switch to folder APP
  - Execute "python run.py" 
    Caution: run.py needs no parameters. The script reads the database file and the pickle model file 
    with the given names: "../data/DB_disaster_msg.db" and "../model/ML_disaster_msg.pkl
    So the parameters for the ETL and ML steps should be exactly as specified above.

Licence, Acknowledgement
The data is provided by company "Figure Eight - 

Stackoverflow Survey 2017 (data link)
World Bank Indicator Data Gdp 2017 (data link)
Continents and Countries (data link)
Please check the license agreements and the detailed description of the data by using the links added.

Without their kind willingness to make the data publically available the project wouldn’t be possible. Many thanks.
