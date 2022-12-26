# Disaster Response ML Pipeline
## Introduction
This project aimed the end-to-end development of a Machine Learning web application for disaster-related messages labeling.

It encompasses the steps starting at an ETL and going up to a web app that allows the user to catch a glimpse of the training data and using the developed model to predict whichever messages they want to predict.  

Currently, the app is not deployed. To access it, it needs to be run locally from the ```app``` folder

## Problem definition
The problem at hand was an NLP-Multilabel classification problem. Given a certain message, the correct categories to which the message maps out (i.e. "belongs") to need to be defined.

To train the model, there is pre-labeled data available.

## Libraries used
For the Data Science + ML Stack, the main libraries used were:  
- Pandas, Numpy, Sklearn, SQLAlchemy + SQLite for storing data

For the Web app:
- Flask + Bootstrap + Jinja + HTML

## Running the project
To run each part of the project, one can simply go into the directories and run the wanted scripts from the command line. Keep in mind that some scripts, if chosen to be tweaked, will replace information used by other parts of the project. This might lead to different results.

Results can only be guaranteed if the project is run using the requirements defined.

### Running the ETL to SQLite 3 (cmd)
``` 
cd data

.../data> python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

```

### Running the model trainer (cmd)
``` 
cd models

.../models> python train_classifier.py ..data/DisasterResponse.db [NAME_OF_CLASSIFIER].pkl

```

**On this step, since the script contains a [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) it can take a while to run depending on your local setup.**

### Running the app (cmd)
To run the app with the current model:

``` 
cd app

.../app> python run.py

```

When the app starts up, an IP adress that contains the rendering of the app will show up in the prompt. Just click or copy-paste in browser to see the running app. 

## Folder structure
```
├───app -> Contains the Web app and HTML templates
│   └───templates -> HTML templates that are rendered
├───data -> Contains ETL scripts and raw/refined data
├───models -> Result files and script related to modelling
└───notebooks -> Contains notebooks used in ETL and modelling exploration
```
## Acknowledgements
This project is part of the [Udacity Data Science Nanodegree Program](https://www.udacity.com/).  
The data used was sourced by Udacity.

