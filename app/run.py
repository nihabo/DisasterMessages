import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """ predefined tokenizer method, required to load the files with joblib
    args:
        text : string - Complete Message text
    return:
        tokenized: list(list(string)) - List of sentences which are again split to list of words
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table("Message_Category", engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """ called when /index page is opened, used to create the visuals to be displayed
    return:
        rendertemplate - required to display the graphs
    """
    
    # extract data needed for visuals
  
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    
    # prepare data for Graph 2: Distribution of Requests/Offers in different genres
    df["request_offer"] = ["request" if value == 1 else "offer" for value in df["request"]]
    
    count = []
    for genre in genre_names:
        df_genre = df[df["genre"] == genre]
        count.append(df_genre.groupby("request_offer").count()["message"])
    labels = ["request", "offer"]
    

    
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
                {
                    'x' : labels,
                    'y' : count[i],
                    'name' : genre_names[i],
                    'type' : 'bar'
                } for i in range(len(genre_names))
                
            ],

            'layout': {
                'title': 'Distribution of Requests and offers in different genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Request vs. Offers"
                },
                'barmode': 'stack'
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
    """ called when user presses "go" and wants to classify a sentence given by user input
    return:
        rendertemplate - required to show the visuals for the classification result
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """ main method called every time this .py file is executed, starts the app
    """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()