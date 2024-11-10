import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''
    load_data
    Load data from the database given its name
    Input: 
    database_filepath - filepath to read the dataframe
    Returns:
    df - dataframe read from the database
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster', engine)
    X = df['message']
    Y = df.iloc[:,4:40]
    return X, Y, Y.columns.tolist()


def tokenize(text):
    '''
    tokenize
    Separate text into clean words 
    Input: 
    text - string input
    Returns:
    clean_tokens - a list of clear tokens
    '''
    words = word_tokenize(text)
    tokens = [w for w in words if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return clean_tokens



def build_model():
    '''
    build_model
    Build pipeline to be able to fit the model

    Returns:
    pipeline - pipeline consisting of transforming tokens and choosing the method
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('mrf',  MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {'mrf__estimator__n_estimators': [10, 20]}
    cv = GridSearchCV(pipeline, parameters)
    return cv




def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model
    Evaluate model using test data
    Input: 
    model - ML model
    X_test - input features in test data
    y_test - expected output
    category_names - names of all outputs
    '''
    y_pred = model.predict(X_test)
    for ind, column in enumerate(category_names):
        print(classification_report(np.array(Y_test[column]), y_pred[:,ind]))


def save_model(model, model_filepath):
    '''
    save_model
    Save a created model to be used later
    Input: 
    model - ML model to be saved
    model_filepath - the file location
    '''
    pickle.dump(model, open(model_filepath, "wb"))


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