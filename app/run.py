import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import re

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

en_stopwords = stopwords.words('english')

en_stopwords = [re.sub('\W','', word) for word in en_stopwords]

app = Flask(__name__)

def preprocess_text(text):
  
    re_non_alnum_pattern = r'[^\w\s]'

    re_url_pattern = r'http(s)?:\/\/[\w.-_]+\s'

    re_twitter_pat = r'(@\S+)|(#\S+)'

    cleaned_text = text.lower()

    cleaned_text = re.sub(re_twitter_pat, ' ', cleaned_text)
    
    cleaned_text = re.sub(re_url_pattern, ' ', cleaned_text)

    cleaned_text = re.sub(re_non_alnum_pattern, ' ', cleaned_text)

    # removes double spacing
    cleaned_text = ' '.join(cleaned_text.split(' '))

    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

def tokenizer(text):

    tokens = text.split(' ')

    wn = WordNetLemmatizer()

    ps = PorterStemmer()

    tokens = [token for token in tokens if token not in en_stopwords]

    tokens = [wn.lemmatize(token) for token in tokens]

    tokens = [ps.stem(token) for token in tokens]

    return tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')

phi_corr_data = pd.read_sql_table('CAT_PHI_CORR', engine)

df = pd.read_sql_table('ANALYTICAL_TABLE', engine)

cat_cols = df.columns[5:-3].to_list()

cat_cols.remove('child_alone')

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    frequent_cats = df[cat_cols].sum()\
                            .sort_values(ascending = False)\
                            .head(10)

    list_category_name = frequent_cats.index.to_list()

    list_category_count = frequent_cats.values

    plot_corrs = phi_corr_data.nlargest(10, columns = 'value').reset_index()

    plot_corrs['label'] = plot_corrs['index']+' & '+plot_corrs['variable']

    plot_corrs_names = plot_corrs['label'].values

    plot_corrs_phi = plot_corrs['value'].values

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=plot_corrs_names,
                    y=plot_corrs_phi
                )
            ],

            'layout': {
                'title': "Most closely related categories by Pearson's Rho",
                'yaxis': {
                    'title': "Pearson's Phi Coefficient"
                },
                'xaxis': {
                    'tickangle':25,
                    'automargin':True
                }
            }
        },
        {
            'data': [
                Bar(
                    x=list_category_name,
                    y=list_category_count
                )
            ],

            'layout': {
                'title': 'Count of occurance of Top 10 most common categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category Name"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    query = preprocess_text(query)

    genre = request.args.get('genre','')

    # use model to predict classification for query
    classification_labels = model.predict([[query,genre]])[0]
    classification_results = dict(zip(df.columns[5:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()