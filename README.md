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

Project Motivation
Despite of the lower return rate of the gender, the gender feature will be part of the study. The main focus lies on the gender and the country issues whether common assumptions about the developer world could be confirmed by the survey of a developer community. Besides of the Stackoverflow survey data, data about the gross domestic product (gdp) from the world bank for the countries and a file from which the continent of a country is extracted are used to enrich the data set.

What is the influence of the gdp on the participation on the survey?
Is there really a gender gap in the community and how big is the gap?
How is the country and continent distribution of the community members?
For the beginning only descriptive statistics techniques are used to exploit the data. It would be interesting to know how the development of the community over the years will happen regarding the addressed topics.

File Descriptions
There is one Jupyter Notebook which contains all the coding ("stov_survey.ipynb"). The data used (s.below) is copied locally in a folder called data.
A more verbal description can be found in the following blog post.

Licence, Acknowledgement
The following data sources are used:

Stackoverflow Survey 2017 (data link)
World Bank Indicator Data Gdp 2017 (data link)
Continents and Countries (data link)
Please check the license agreements and the detailed description of the data by using the links added.

Without their kind willingness to make the data publically available the project wouldn’t be possible. Many thanks.
