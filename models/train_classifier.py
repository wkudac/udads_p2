import sys
import numpy as np
import pandas as pd

from sqlalchemy import create_engine
import pickle

# from utils.util_stat import DfStat 
from datetime import datetime
from pytz import timezone

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble  import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline 
from sklearn.multioutput import MultiOutputClassifier

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize 
from nltk.stem.wordnet import WordNetLemmatizer
import re 

GC_UTC = timezone('UTC')

def load_data(database_filepath):
    """
    load the data from the database 
    """
    # DB Engine
    db_engine_url = 'sqlite:///../data/' + database_filepath
    db_engine = create_engine(db_engine_url)
    df = pd.read_sql("SELECT * FROM Messages", db_engine)
    X = df.message.values
    Y = df[df.columns[4:]]
    col_names = df.columns[4:]
    return X, Y, col_names


def tokenize(text):
    """
    Prepare the NLP modelling by tokenize the text phrases 
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens: 
        clean_tok = lemmatizer.lemmatize(re.sub(r"[^a-zA-Z0-9]", " ", tok.lower()).strip())
        clean_tokens.append(clean_tok) 
    return clean_tokens


def build_model():
    """
    build the NLP model by create a pipeline which defines all the 
    used estimators 
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(
            AdaBoostClassifier(random_state=42),
            n_jobs=-1)),
    ])
    #     parameters = {
    #         'tfidf__use_idf': (True, False),
    #         'clf__estimator__n_estimators': [50,100,200,400],
    #         'clf__estimator__learning_rate': [0.5, 1, 2, 10, 20]
    #     }
    parameters = {
        'tfidf__use_idf': [False],
        'clf__estimator__n_estimators': [50],
        'clf__estimator__learning_rate': [0.5]
    }
    model = GridSearchCV(pipeline, param_grid=parameters,verbose=5)
    return model 

def evaluate_model(model, X_test, Y_test, category_names):
    """
    print some evaluation metrics of the model training
    """
    # evaluate all steps on test set
    predicted = model.predict(X_test)
    print('Accuracy\n', (predicted == Y_test).mean())
    print('cv_report\n', classification_report(Y_test, predicted, 
                                       target_names=Y_test.columns, 
                                       zero_division=False))

def save_model(model, model_filepath):
    """
    save the model as a pickle file 
    """
    filename = model_filepath
    outfile = open(filename,'wb')
    pickle.dump(model, outfile)
    outfile.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        tmst =  datetime.now(GC_UTC)
        print('Loading data...{}\n    DATABASE: {}'.format(tmst, database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        tmst =  datetime.now(GC_UTC)
        print('Building model...', tmst)
        model = build_model()
        
        tmst =  datetime.now(GC_UTC)
        print('Training model...', tmst) 
        model.fit(X_train, Y_train)
        
        tmst =  datetime.now(GC_UTC)
        print('Evaluating model...', tmst) 
        evaluate_model(model, X_test, Y_test, category_names)

        tmst =  datetime.now(GC_UTC)
        print('Saving model... {}\n    MODEL: {}'.format(tmst,model_filepath))
        save_model(model, model_filepath)

        tmst =  datetime.now(GC_UTC)
        print('Trained model saved!', tmst) 

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
