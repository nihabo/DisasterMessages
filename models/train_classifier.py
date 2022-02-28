import sys
import pandas as pd
# imports required for loading & saving
from sqlalchemy import create_engine
import pickle
# imports required for Tokenizing
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
# imports required to build the ML model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
# imports required for model evaluation
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# to check runtime
from datetime import datetime


def load_data(database_filepath):
    """ Loads the Data from the database and stores it in feature/target dataframe
    args:
        database_filepath : string - contains the path for the database e.g. example.db
    return:
        X: pd.DataFrame - Feature Variables
        Y: pd.DataFrame - Target Variables
        category_names: list(string) - Names of all Target Categories
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM Message_Category', engine)
    X = df["message"]
    Y = df.drop(["id","message", "original", "genre"], axis = 1)
    
    return X, Y, Y.columns


def tokenize(text):
    """ Converts continous message text to single (normalized) word tokens
    args:
        text : string - Complete Message text
    return:
        tokenized: list(list(string)) - List of sentences which are again split to list of words
    """
    tokenized = sent_tokenize(text)
    noun_lemmatizer = WordNetLemmatizer()
    verb_lemmatizer = WordNetLemmatizer(pos = 'v')
    
    for sent in tokenized:
        sent = word_tokenize(sent)
        sent = [noun_lemmatizer(word.lower()) for word in sent]
        sent = [verb_lemmatizer(word) for word in sent]
        
    return tokenized


def build_model():
    """ Builds up the Machine Learning model as GridSearch by using a pipeline
    return:
        cv: GridSearchCV - Machine Learning model containing parameters which can be used for                               GridSearch
    """
    pipeline = Pipeline([
        ("vect", CountVectorizer(tokenize)),
        ("tfidf", TfidfTransformer()),
        ("clf", MultiOutputClassifier(RandomForestClassifier()))
    
    ])
    
    parameters = {'vect__stop_words': [None, "english"],
             #'vect__binary': [True, False],
             'tfidf__smooth_idf': [True, False],
             'tfidf__use_idf': [True,False]}#,
             #'clf__estimator': [RandomForestClassifier(), BaggingClassifier()]}

    cv = GridSearchCV(pipeline, parameters, verbose = 5, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """ Predicts Category values for Test-Features and compares to actual target values
    args:
        model: GridSearchCV - trained machine learning model
        X_test: pd.DataFrame - containing Features of Testdata
        Y_test: pd.DataFrame - containing actual Category values for Testdata
        category_names: list(string) - names of all target categories
    """
    Y_pred = model.predict(X_test)
    
    f = open("results" + datetime.now().strftime("%d%m%Y_%H%M%S"), "w")
    f.write(f"Best Parameters: {model.best_params_} \n")
    
    print("Best Parameters: ",  model.best_params_)
    
    for i, col in enumerate(category_names):
        print(classification_report(Y_test[col], Y_pred[:,i], target_names=[col + "_0", col + "_1", col + "_other"]))
        f.write(f"{classification_report(Y_test[col], Y_pred[:,i], target_names=[col + '_0', col + '_1', col + '_other'])} \n")
        
    f.close()


def save_model(model, model_filepath):
    """ save trained model using pickle
    args:
        model : GridSearchCV - trained machine learning model
        model_filepath: string - path where file should be stored to including filename 
                                e.g. file.sav
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ Executing all necessary steps to train, evaluate and store the ML model
        all necessary filepaths are provided by userinput
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print(datetime.now().strftime("%H:%M:%S"), 'Building model...')
        model = build_model()
        
        print(datetime.now().strftime("%H:%M:%S"), 'Training model...')
        model.fit(X_train, Y_train)
        
        print(datetime.now().strftime("%H:%M:%S"), 'Evaluating model...')
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