################### IMPORTS ###################
# General
import sys
import pandas as pd
import sqlalchemy
import os
import pickle


# NLP
import nltk

from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re

# ML

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV


###################### SCRIPT ########################

en_stopwords = stopwords.words('english')

en_stopwords = [re.sub('\W','', word) for word in en_stopwords]

def load_data(database_filepath):
    '''
    Load database from SQLite

    :param database_filepath: Path to the cleaned database in SQLite

    :return X: np.array with features to use for prediction
    :return y: np.array with features to be predicted
    :return cat_names: names of the categories that are predicted
    '''

    conn = sqlalchemy.create_engine('sqlite:///' + os.path.abspath(database_filepath))

    df = pd.read_sql_table('ANALYTICAL_TABLE', conn)

    df.set_index('index', inplace = True)

    df.drop(columns = 'child_alone', inplace = True)

    cat_names = df.columns[4:-2].to_list()

    language_filter = df['infered_language'] == 'en'

    df = df[language_filter]

    X = df[['message_clean','genre']].values

    y = df[cat_names]
    
    return X, y, cat_names


def tokenizer(text):
    '''
    Tokenizes, Lemmatizes and Stemms a string.

    :param text: String to be tokenized

    :return tokens: List of tokens that are in the string
    '''

    tokens = text.split(' ')

    wn = WordNetLemmatizer()

    ps = PorterStemmer()

    tokens = [token for token in tokens if token not in en_stopwords]

    tokens = [wn.lemmatize(token) for token in tokens]

    tokens = [ps.stem(token) for token in tokens]

    return tokens

def build_model():
    '''
    Instantiate a GridSearchCV to be tuned.

    :return cv: GridSearchCV object with the respective parameters
    '''

    full_ct = ColumnTransformer([('tfidf',TfidfVectorizer(analyzer = 'word', tokenizer = tokenizer, ngram_range= (1,2)), 0),
                                    ('onehot',OneHotEncoder(),[1])])

    pipe = Pipeline([('ColumnTransformer', full_ct),
                    ('clf', MultiOutputClassifier(LogisticRegression(random_state=123,solver = 'saga',max_iter=200, n_jobs=-1),n_jobs=-1))])

    hamming_score = make_scorer(hamming_loss, greater_is_better=False)

    params = {'ColumnTransformer__tfidf__ngram_range':[(1,1),(1,2)],
                'clf__estimator__C':[1,5]}

    # Using cv = 2 for processing time reasons
    cv = GridSearchCV(pipe, param_grid=params,scoring = hamming_score, cv = 2 ,n_jobs = -1)

    return cv
        

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluates and prints to terminal a model result based on f1-score, recall, precision and hamming loss (overall single labels matched correctly)

    :param model: Instance of a model
    :param X_test: Test features to be used as predictors
    :param Y_test: Test features to be predicted
    :param category_names: list of strings with the name of the categories
    '''

    Y_pred = model.predict(X_test)

    print(classification_report(Y_test, Y_pred, zero_division=0, target_names=category_names))
    print('Overall Hamming Loss:', hamming_loss(Y_test, Y_pred))
    

def save_model(model, model_filepath):

    '''
    Saves the model as a .pkl file according to the model_filepath

    :param model: Instance of the model to be saved
    :param model_filepath: path to save the .pkl to
    '''
    
    with open(model_filepath,'wb') as file:

        pickle.dump(model, file)

def main():
    '''
    Runs the training pipeline. Loads the data, instantiates the model, tune its hyperparameters and with the best estimator, evaluates the model and saves it.
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        cv = build_model()
        
        print('Training model...')
        cv.fit(X_train, Y_train)

        print('Best model hyperparameters:')
        print(cv.best_params_)
    
        model = cv.best_estimator_
        
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