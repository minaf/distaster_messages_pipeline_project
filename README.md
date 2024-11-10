# Disaster Response Pipeline Project
We have developed a machine learning pipeline using a dataset of real messages sent during disaster events. The pipeline categorizes these messages and then sends the information to appropriate disaster relief agency. The project also includes a web app, where emergency workers can input new messages and receive classification results across multiple categories.

The project is divided into three folders: _data_, _model_, and _app_. In the folder _data_, we have csv files of data and the file to process data (_process_data.py_). The folder _model_ consists of the file to train (_train_classifier.py_). The folder _app_ contains everything for the web application to run. To chnage the layout of the visualization in the application use the file _run.py_. 
Notebooks _ETL Pipeline Preparation.ipynb_ and _ML Pipeline Preparation.ipynb_ are used to exploratory data analysis and building ML pipeline. The files in the folders _data_ and _model_ are based on these notebooks.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### 
This project is based on the Udacity course.
