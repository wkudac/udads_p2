import json
import plotly
import matplotlib as plt 
import numpy as np
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
import plotly.graph_objs as go 
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DB_disaster_msg.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/ML_disaster_msg.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    sample_counts = df[df.columns[4:]].sum().sort_values(ascending=False)
    sample_names = list(sample_counts.index) 
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    plot_1 =  {
        'data': [{
            'x': genre_names,
            'y': genre_counts,
            'type': 'bar'
        }], 
        'layout': {
            'title': 'Genre',
            'xaxis': {
                'title': 'Genre'                    
            },
            'yaxis': {
                'title': 'No. of Occureces'
            }
        }
    }
    plot_2 =  {
        'data': [{
            'labels': genre_names,
            'values': genre_counts,
            'type': 'pie'
        }], 
        'layout': {
            'title': 'Genre'
        }
    }
    plot_3 =  {
        'data': [{
            'x': sample_counts,
            'y': sample_names,
            'type': 'bar',
            'orientation': 'h'
        }], 
        'layout': {
            'title': 'Samples Category in Modelbuilding',
            'xaxis': {
                'title': 'No. of Occurrences'                    
            },
            'yaxis': {
                'title': 'Category'
            },
            'autosize': False,
            'height': 800, 
            'margin': dict(l=250, r=10, t=80, b=80),
            'textfont': dict(family='Courier New', size=5)
        }
    }
    plots = [plot_1, plot_2, plot_3] 

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(plots)]
    graphJSON = json.dumps(plots, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    # classification_results = dict(zip(df.columns[4:], classification_labels))
    classification_results = pd.DataFrame(zip(df.columns[4:], classification_labels))
    classification_probabilities = model.predict_proba([query])
    dic_result = prep_probs(classification_results, classification_probabilities)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=dic_result
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)

def prep_probs(p_res_class, p_res_probs):
    """ 
    Prepare probability metrics 
    """
    df_res = pd.DataFrame() 
    for row in p_res_probs:  
        prob0 = '{}'.format(round(row[0][0], 2))
        if len(row[0]) > 1:
            prob1 = '{}'.format(round(row[0][1], 2))
        else:
            prob1 = '0'
        new_row = {'Prob_0': prob0, 'Prob_1': prob1} 
        df_res = df_res.append(new_row, ignore_index=True) 
    df_res = p_res_class.join(df_res) 
    df_res.columns = ['Name', 'Label', 'Prob_0', 'Prob_1' ]
    df_res['Probs'] = df_res['Prob_1']
    # df_res.loc[df_res['Label'] == 0, 'Probs'] = df_res['Prob_0']
    # df_res.loc[df_res['Label'] == 1, 'Probs'] = df_res['Prob_1']
    df_res['SortValue'] = df_res.apply(lambda x:'%s_%s_%s' % (x['Label'],x['Probs'],x['Name']),axis=1)
    df_res.sort_values(by=['SortValue'], ascending=False, inplace=True) 

    return df_res

if __name__ == '__main__':
    main()
