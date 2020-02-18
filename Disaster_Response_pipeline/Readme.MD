Data Science Nanodegree Project 5: Disaster Response Pipeline Webapp

Description
When natural disasters happen, people use twitter to try and get help. However, not all tweets are natural disaster tweets and there isn't a simple way to use key words in order to identify these kinds of texts.

Towards this end, the task is to create a machine learning webapp that is able to take tweets and discern whether they are relevant for a disaster response team or not using data that has already been labeled.

There are three steps:

Create an ETL which cleans the Data
Create a ML pipeline which performs feature extraction and trains a model
Take model and embed it into a webapp
Repo Layout
This repo is split into subdirectories:

data-processing_and_model-training - Contains notebooks, python files and other such files for ML Engineering purposes
webapp - Contains webapp for deployment, see README inside this directory for more instructions
Packages Used
sys
pandas
sqlalchemy
joblib
re
nltk
sklearn
json
plotly
flask
Acknowle
