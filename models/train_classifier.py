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


###################### SCRIPT ########################

en_stopwords = stopwords.words('english')

en_stopwords = [re.sub('\W','', word) for word in en_stopwords]

def load_data(database_filepath):

    conn = sqlalchemy.create_engine('sqlite:///' + os.path.abspath(database_filepath))

    df = pd.read_sql('SELECT * FROM ANALYTICAL_TABLE', conn)

    df.set_index('index', inplace = True)

    df.drop(columns = 'child_alone', inplace = True)

    cat_names = df.columns[4:-2].to_list()

    language_filter = df['infered_language'] == 'en'

    df = df[language_filter]

    X = df[['message_cleaned','genre']]

    y = df[cat_names]
    
    return X, y, cat_names


def tokenizer(text):

    tokens = text.split(' ')

    wn = WordNetLemmatizer()

    ps = PorterStemmer()

    tokens = [token for token in tokens if token not in en_stopwords]

    tokens = [wn.lemmatize(token) for token in tokens]

    tokens = [ps.stem(token) for token in tokens]

    return tokens

def build_model():

    full_ct = ColumnTransformer([('tfidf',TfidfVectorizer(analyzer = 'word', tokenizer = tokenizer, ngram_range= (1,2)), 'message_cleaned'),
                                    ('onehot',OneHotEncoder(),['genre'])])

    model = Pipeline([('ColumnTransformer', full_ct),
                    ('clf', MultiOutputClassifier(LogisticRegression(random_state=123,solver = 'saga',max_iter=200, C=5, n_jobs=-1),n_jobs=-1))])


    return model
        

def evaluate_model(model, X_test, Y_test, category_names):

    Y_pred = model.predict(X_test)

    print(classification_report(Y_test, Y_pred, zero_division=0, target_names=category_names))
    print('Overall Hamming Loss:', hamming_loss(Y_test, Y_pred))
    

def save_model(model, model_filepath):
    
    with open(model_filepath,'wb') as file:

        pickle.dump(model, file)

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