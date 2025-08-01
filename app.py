import matplotlib
import requests
from requests_oauthlib import OAuth1
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, callback_context
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.cluster import KMeans
import nltk
from scipy.stats import gaussian_kde
from nltk.metrics import BigramAssocMeasures
from wordcloud import WordCloud
import base64
from io import BytesIO
import re
from datetime import datetime, timedelta
from typing import Iterable
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import json
import os
from utils import get_queries, get_secrets, query_netsuite


def fetch_data():
    secrets = get_secrets()
    queries = get_queries()
    print("Using OAuth1 for NetSuite API")
    print(f"secrets: {secrets}")
    print(f"queries: {queries}")

    all_items = []
    offset = 0
    limit = 1000
    max_records = 100000


    while True:
        json_result = query_netsuite(queries['get_eos_notes'], limit, offset)

        items = json_result.get('items', [])
        all_items.extend(items)


        print(f"Fetched batch of {len(items)} records (total: {len(all_items)})")


        if not json_result.get('hasMore', False) or len(all_items) >= max_records:
            break


        offset += limit

    df = pd.DataFrame(all_items)
    print(f"Fetched {len(df)} total records from NetSuite")

    return df



df = fetch_data()
print(f"Fetched {len(df)} records from NetSuite")


required_fields = ['custrecord_eos_date', 'custrecord_eos_memo', 'custrecord_eos_prod_line']
for field in required_fields:
    if field not in df.columns:
        print(f"WARNING: Required field '{field}' is missing from the API response")
    else:
        print(f"Field '{field}' is present. Sample value:", df[field].iloc[0] if not df.empty else "No data")

df['custrecord_eos_memo'] = df['custrecord_eos_memo'].fillna('')
df['DATE'] = pd.to_datetime(df['custrecord_eos_date'])
df['MONTH'] = df['DATE'].dt.strftime('%Y-%m')
df['YEAR'] = df['DATE'].dt.year
df['WEEKDAY'] = df['DATE'].dt.day_name()

nltk_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)

nltk.data.path.append(nltk_data_path)


def ensure_nltk_resource(resource_name):
    if resource_name == 'vader_lexicon':
        resource_path = f"sentiment/{resource_name}"
    else:
        resource_path = f"corpora/{resource_name}"

    try:
        nltk.data.find(resource_path)
        print(f"NLTK resource already available: {resource_name}")
        return True
    except LookupError:
        print(f"NLTK resource not found: {resource_name}. Attempting to download...")
        try:
            download_result = nltk.download(resource_name, download_dir=nltk_data_path, quiet=False)
            if download_result:
                print(f"Successfully downloaded NLTK resource: {resource_name}")
                return True
            else:
                print(f"Warning: Failed to download NLTK resource: {resource_name}")
                return False
        except Exception as e:
            print(f"Error downloading NLTK resource {resource_name}: {str(e)}")
            return False


print(f"Checking NLTK resources in {nltk_data_path}")
required_resources = ['vader_lexicon', 'stopwords', 'punkt']
missing_resources = []

for resource in required_resources:
    if not ensure_nltk_resource(resource):
        missing_resources.append(resource)
        print(f"Warning: Could not access resource {resource}, will use fallbacks if needed")

if 'vader_lexicon' not in missing_resources:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    sentiment_analyzer = SentimentIntensityAnalyzer()
else:
    class SimpleSentimentAnalyzer:
        def polarity_scores(self, text):
            positive_words = ['good', 'great', 'excellent', 'perfect', 'resolved', 'fixed', 'improved', 'success']
            negative_words = ['issue', 'problem', 'error', 'fail', 'down', 'jam', 'stuck', 'break', 'stopped']

            text = text.lower()
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)

            total = pos_count + neg_count
            if total == 0:
                compound = 0
            else:
                compound = (pos_count - neg_count) / total

            return {'compound': compound, 'pos': pos_count, 'neg': neg_count, 'neu': 1}


    sentiment_analyzer = SimpleSentimentAnalyzer()
    print("Using simple fallback sentiment analyzer")

if 'stopwords' not in missing_resources:
    from nltk.corpus import stopwords

    nltk_stopwords = set(stopwords.words('english'))
else:
    nltk_stopwords = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
        'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
        'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
        'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
        'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
        'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
        'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
        'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's',
        't', 'can', 'will', 'just', 'don', 'should', 'now'
    }
    print("Using fallback stopwords list")

if 'punkt' not in missing_resources:
    from nltk.tokenize import word_tokenize

    tokenize_function = word_tokenize
else:
    def tokenize_function(text):
        return text.lower().split()


    print("Using simple fallback tokenizer")



DOMAIN_STOP = {
    'line', 'shift', 'run', 'case', 'production', 'machine', 'ran',
    'cans', 'cases', 'product', 'products', 'packaging', 'package'
}
STOP_WORDS = nltk_stopwords | DOMAIN_STOP


df['SENT_SCORE'] = df['custrecord_eos_memo'].apply(
    lambda x: sentiment_analyzer.polarity_scores(str(x))['compound'] if isinstance(x, str) else 0
)

product_lines = sorted(df['custrecord_eos_prod_line'].dropna().unique())


def extract_ngrams(texts, ngram_range=(1, 1), top_n=20):
    vectorizer = CountVectorizer(
        lowercase=True,
        token_pattern=r'\b[a-zA-Z]+\b',
        stop_words=list(STOP_WORDS),
        ngram_range=ngram_range
    )
    X = vectorizer.fit_transform(texts)
    counts = np.asarray(X.sum(axis=0)).ravel()
    features = vectorizer.get_feature_names_out()
    return pd.DataFrame({'phrase': features, 'count': counts}) \
        .sort_values('count', ascending=False) \
        .head(top_n)


def extract_tfidf_terms(texts, ngram_range=(1, 1), top_n=20):
    if len(texts) == 0:
        return pd.DataFrame(columns=['phrase', 'tfidf'])

    try:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]+\b',
            stop_words=list(STOP_WORDS),
            ngram_range=ngram_range
        )
        X = vectorizer.fit_transform(texts)
        scores = np.asarray(X.sum(axis=0)).ravel()
        features = vectorizer.get_feature_names_out()

        scores = np.round(scores, 2)

        return pd.DataFrame({'phrase': features, 'tfidf': scores}) \
            .sort_values('tfidf', ascending=False) \
            .head(top_n)
    except Exception as e:
        print(f"tf-idf computation failed: {e}")
        return pd.DataFrame(columns=['phrase', 'tfidf'])


def extract_lda_topics(texts, n_topics=5, n_top_words=5):
    vectorizer = CountVectorizer(stop_words=list(STOP_WORDS))
    X = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=0,
        learning_method='online'
    )
    lda.fit(X)

    terms = vectorizer.get_feature_names_out()
    topics = []
    for i, comp in enumerate(lda.components_):
        top_word_indices = comp.argsort()[:-n_top_words - 1:-1]
        top_terms = [terms[idx] for idx in top_word_indices]
        topics.append({
            'topic': i + 1,
            'top_terms': ', '.join(top_terms)
        })

    return pd.DataFrame(topics)


def extract_clusters(texts: Iterable[str], n_clusters: int = 5) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(stop_words=list(STOP_WORDS))
    X = vectorizer.fit_transform(texts)

    if X.shape[0] < n_clusters:
        if X.shape[0] == 0:
            return pd.DataFrame(columns=['cluster', 'count'])
        n_clusters = max(2, X.shape[0])

    n_samples = X.shape[0]
    n_clusters = min(n_samples, n_clusters)
    model = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = model.fit_predict(X)

    counts = np.unique(labels, return_counts=True)[1]
    return pd.DataFrame({
        'cluster': np.unique(labels),
        'count': counts
    }).sort_values('count', ascending=False)


def create_wordcloud(texts):
    if not isinstance(texts, (pd.Series, list)) or len(texts) == 0 or not any(texts):
        return px.scatter(title="No data available for wordcloud")

    text = ' '.join([str(t) for t in texts if isinstance(t, str) and t]).strip()
    if not text:
        return px.scatter(title="No valid text available for wordcloud")

    wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=STOP_WORDS,
        max_words=100,
        collocations=True
    ).generate(text)

    img = wc.to_array()
    fig_wc = px.imshow(img, title="Word Cloud Visualization")
    fig_wc.update_xaxes(visible=False)
    fig_wc.update_yaxes(visible=False)
    return fig_wc


def generate_pca_scatter(texts, dates=None):
    if len(texts) < 3:
        return px.scatter(title="Not enough data for PCA visualization")

    vectorizer = TfidfVectorizer(stop_words=list(STOP_WORDS))
    try:
        X = vectorizer.fit_transform(texts)

        pca = PCA(n_components=2)
        components = pca.fit_transform(X.toarray())

        var_explained = pca.explained_variance_ratio_

        # Create a dataframe with PCA components
        pca_df = pd.DataFrame({
            'pca_x': components[:, 0],
            'pca_y': components[:, 1]
        })

        # Add date information if available
        if dates is not None and len(dates) == len(pca_df):
            pca_df['day'] = pd.Series(dates).dt.strftime('%Y-%m-%d')

            fig = px.scatter(
                pca_df,
                x='pca_x',
                y='pca_y',
                text='day',  # Add day labels to points
                labels={
                    'pca_x': f'PC1 ({var_explained[0]:.2%} variance)',
                    'pca_y': f'PC2 ({var_explained[1]:.2%} variance)'
                },
                title="PCA of Document Similarity"
            )
            # Make the text visible by default
            fig.update_traces(textposition='top center')
        else:
            fig = px.scatter(
                pca_df,
                x='pca_x',
                y='pca_y',
                labels={
                    'pca_x': f'PC1 ({var_explained[0]:.2%} variance)',
                    'pca_y': f'PC2 ({var_explained[1]:.2%} variance)'
                },
                title="PCA of Document Similarity"
            )

        return fig
    except Exception as e:
        print(f"PCA computation failed: {e}")
        return px.scatter(title="PCA computation failed")

def extract_sentiment_keywords(texts, positive_terms, negative_terms):
    pos_counts = {}
    neg_counts = {}

    for text in texts:
        if not isinstance(text, str): continue

        for term in positive_terms:
            if re.search(r'\b' + term + r'\b', text, re.IGNORECASE):
                pos_counts[term] = pos_counts.get(term, 0) + 1

        for term in negative_terms:
            if re.search(r'\b' + term + r'\b', text, re.IGNORECASE):
                neg_counts[term] = neg_counts.get(term, 0) + 1

    return {
        'positive': pd.DataFrame(
            {'term': list(pos_counts.keys()), 'count': list(pos_counts.values())}
        ).sort_values('count', ascending=False),
        'negative': pd.DataFrame(
            {'term': list(neg_counts.keys()), 'count': list(neg_counts.values())}
        ).sort_values('count', ascending=False)
    }


POSITIVE_TERMS = ['good', 'great', 'excellent', 'perfect', 'resolved', 'fixed', 'improved', 'success']
NEGATIVE_TERMS = ['issue', 'problem', 'error', 'fail', 'down', 'jam', 'stuck', 'break', 'stopped']

line_metrics = {}
for product_line in product_lines:
    texts = df[df['custrecord_eos_prod_line'] == product_line]['custrecord_eos_memo']
    line_metrics[product_line] = {
        'bigrams': extract_ngrams(texts, (2, 2)),
        'tfidf_bigrams': extract_tfidf_terms(texts, (2, 2)),
        'lda_topics': extract_lda_topics(texts),
        'clusters': extract_clusters(texts),
        'sentiment': extract_sentiment_keywords(texts, POSITIVE_TERMS, NEGATIVE_TERMS)
    }

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True
)


app.enable_dev_tools()


def serve_layout():
    navbar = dbc.NavbarSimple(
        brand="End-of-Shift Dashboard",
        color="dark",
        dark=True,
        fluid=True,
    )

    controls = dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id="line-filter",
                options=[{"label": l, "value": l} for l in product_lines],
                placeholder="Select Product Line",
                value=product_lines[0] if product_lines else None,
            ),
            md=4,
        ),
        dbc.Col(
            dcc.DatePickerRange(
                id="date-range",
                start_date=df["DATE"].min(),
                end_date=df["DATE"].max(),
                display_format="YYYY-MM-DD",
            ),
            md=8,
        ),
    ], className="my-4")

    kpi_row = dbc.Row(id="kpi-row", className="mb-4")

    # Charts
    charts = dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Bigram Summary"),
                dbc.CardBody(dcc.Graph(id="bigram_summary")),
            ]),
            md=6,
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("TF-IDF Bigrams"),
                dbc.CardBody(dcc.Graph(id="tfidf_bigrams")),
            ]),
            md=6,
        ),
    ], className="mb-4")

    charts2 = dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("LDA Topics"),
                dbc.CardBody(dcc.Graph(id="lda_topics")),
            ]),
            md=6,
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Time Series Trend"),
                dbc.CardBody(dcc.Graph(id="timeseries_chart")),
            ]),
            md=6,
        ),
    ], className="mb-4")

    charts3 = dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Downtime Analysis"),
                dbc.CardBody(dcc.Graph(id="downtime-pie-chart")),
            ]),
            md=12,  # Changed to full width
        ),
    ], className="mb-4")

    charts4 = dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Sentiment Analysis"),
                dbc.CardBody(dcc.Graph(id="sentiment_chart")),
            ]),
            md=12,
        ),
    ], className="mb-4")

    charts5 = dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("PCA Document Similarity"),
                dbc.CardBody(dcc.Graph(id="pca_scatter")),
            ]),
            md=12,
        ),
    ], className="mb-4")

    charts6 = dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Sentiment Distribution"),
                dbc.CardBody(dcc.Graph(id="sentiment-dist-chart")),
            ]),
            md=6,
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Sentiment Trend"),
                dbc.CardBody(dcc.Graph(id="sentiment-trend-chart")),
            ]),
            md=6,
        ),
    ], className="mb-4")

    charts7 = dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Extreme Sentiment Dates"),
                dbc.CardBody(dcc.Graph(id="extremes-chart")),
            ]),
            md=12,
        ),
    ], className="mb-4")

    # Add hidden placeholder for cluster_summary to keep callback structure intact
    hidden_components = html.Div(dcc.Graph(id="cluster_summary", style={"display": "none"}))

    return dbc.Container([
        navbar,
        controls,
        kpi_row,
        charts,
        charts2,
        charts3,
        charts4,
        charts5,
        charts6,
        charts7,
        hidden_components
    ], fluid=True)


app.layout = serve_layout




def sentiment_distribution(texts):
    scores = [sentiment_analyzer.polarity_scores(t)['compound'] for t in texts]
    return pd.DataFrame({'score': scores})


@app.callback(
    Output('kpi-row', 'children'),
    Output('bigram_summary', 'figure'),
    Output('tfidf_bigrams', 'figure'),
    Output('lda_topics', 'figure'),
    Output('cluster_summary', 'figure'),
    Output('downtime-pie-chart', 'figure'),
    Output('timeseries_chart', 'figure'),
    Output('sentiment_chart', 'figure'),
    Output('pca_scatter', 'figure'),
    Output('sentiment-dist-chart', 'figure'),
    Output('sentiment-trend-chart', 'figure'),
    Output('extremes-chart', 'figure'),
    Input('line-filter', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date')
)
def update_dashboard(selected_line, start_date, end_date):
    m = line_metrics[selected_line]

    line_df = df[df['custrecord_eos_prod_line'] == selected_line]
    if start_date and end_date:
        line_df = line_df[(line_df['DATE'] >= start_date) & (line_df['DATE'] <= end_date)]

    def create_table(df, title):
        return go.Figure(data=[go.Table(
            header=dict(values=list(df.columns), fill_color='lightgrey', align='left'),
            cells=dict(values=[df[col] for col in df.columns], align='left')
        )]).update_layout(title=title, height=400)

    total_memos = len(line_df)
    avg_sent = line_df['SENT_SCORE'].mean()
    downtime_memos = line_df['custrecord_eos_memo'] \
        .str.contains('down|jam|stuck|break|stopped|failure|error', case=False).sum()
    kpis = [
        html.Div([html.H4('Total Memos'), html.P(total_memos)]),
        html.Div([html.H4('Avg Sentiment'), html.P(f"{avg_sent:.2f}")]),
        html.Div([html.H4('Downtime Memos'), html.P(downtime_memos)])
    ]

    def top_extremes(df_line):
        df_line = df_line.copy()
        df_line['score'] = df_line['custrecord_eos_memo'] \
            .apply(lambda t: sentiment_analyzer.polarity_scores(t)['compound'])
        top_pos = df_line.nlargest(5, 'score')[['DATE', 'custrecord_eos_memo', 'score']]
        top_neg = df_line.nsmallest(5, 'score')[['DATE', 'custrecord_eos_memo', 'score']]
        return top_pos, top_neg

    fig_bg = create_table(m['bigrams'], f"{selected_line} — Top Bigrams")
    fig_tb = create_table(m['tfidf_bigrams'], f"{selected_line} — Top TF-IDF Bigrams")
    fig_ld = create_table(m['lda_topics'], f"{selected_line} — LDA Topics")

    # Create an empty figure instead of the cluster sizes table
    fig_cl = go.Figure()
    fig_cl.update_layout(title="Cluster Summary")

    time_df = line_df.groupby(pd.Grouper(key='DATE', freq='W')).size().reset_index(name='count')
    fig_ts = px.line(
        time_df,
        x='DATE',
        y='count',
        title=f"{selected_line} — Weekly Comment Trends",
        labels={'DATE': 'Week', 'count': 'Number of Comments'}
    )

    pos_df = m['sentiment']['positive'].head(10)
    neg_df = m['sentiment']['negative'].head(10)

    sentiment_df = pd.DataFrame({
        'Term': list(pos_df['term']) + list(neg_df['term']),
        'Count': list(pos_df['count']) + list(neg_df['count']),
        'Sentiment': ['Positive'] * len(pos_df) + ['Negative'] * len(neg_df)
    })

    fig_sent = px.bar(
        sentiment_df,
        x='Term',
        y='Count',
        color='Sentiment',
        title=f"{selected_line} — Sentiment Analysis",
        color_discrete_map={'Positive': 'green', 'Negative': 'red'}
    )

    scores = np.array([
        sentiment_analyzer.polarity_scores(txt)['compound']
        for txt in line_df['custrecord_eos_memo']
    ])

    if scores.size == 0:
        fig_dist = go.Figure()
    else:
        kde = gaussian_kde(scores)
        x_grid = np.linspace(scores.min(), scores.max(), 200)
        y_kde = kde(x_grid)

        fig_dist = go.Figure([
            go.Histogram(
                x=scores,
                nbinsx=20,
                histnorm='probability density',
                name='Histogram'
            ),
            go.Scatter(
                x=x_grid,
                y=y_kde,
                mode='lines',
                name='KDE'
            )
        ])
        fig_dist.update_layout(
            title='Sentiment Score Distribution (Density + KDE)',
            xaxis_title='VADER Compound Score',
            yaxis_title='Probability Density',
            bargap=0.2
        )

    trend_df = (
        line_df.set_index('DATE')['SENT_SCORE']
        .resample('W').mean()  # Changed from 'M' to 'W' for weekly
        .reset_index(name='avg_score')
    )
    fig_trend = px.line(
        trend_df, x='DATE', y='avg_score', markers=True,
        title='Weekly Avg Sentiment',  # Updated title
        labels={'avg_score': 'Avg VADER Score'}
    )

    top_pos, top_neg = top_extremes(line_df)

    top_pos['DATE_STR'] = top_pos['DATE'].dt.strftime('%Y-%m-%d')
    top_neg['DATE_STR'] = top_neg['DATE'].dt.strftime('%Y-%m-%d')

    fig_ex = go.Figure()
    fig_ex.add_trace(go.Scatter(
        x=top_pos['DATE'], y=top_pos['score'],
        mode='markers+text',
        name='Top Positive',
        text=top_pos['DATE_STR'],
        textposition='top center',
        marker=dict(color='green', size=10)
    ))
    fig_ex.add_trace(go.Scatter(
        x=top_neg['DATE'], y=top_neg['score'],
        mode='markers+text',
        name='Top Negative',
        text=top_neg['DATE_STR'],
        textposition='bottom center',
        marker=dict(color='red', size=10)
    ))
    fig_ex.update_layout(
        title='Top 5 Positive & Negative Memo Dates',
        xaxis_title='Date',
        yaxis_title='VADER Compound Score'
    )

    fig_pca = generate_pca_scatter(line_df['custrecord_eos_memo'], line_df['DATE'])


    downtime_pattern = 'down|jam|stuck|break|stopped|failure|error'
    line_df['is_downtime'] = line_df['custrecord_eos_memo'].str.contains(downtime_pattern, case=False)
    downtime_count = line_df['is_downtime'].sum()
    total_count = len(line_df)
    other = total_count - downtime_memos

    fig_pie = px.pie(
        values=[downtime_memos, other],
        names=['Downtime', 'Other'],
        hole=0.4,
        title='Downtime vs Other Memos',
        color_discrete_sequence=['red', 'lightblue']
    )
    fig_pie.update_traces(textinfo='percent+value', textposition='inside')

    return [
        kpis,fig_bg, fig_tb,
        fig_ld, fig_cl, fig_pie,
        fig_ts, fig_sent, fig_pca,
        fig_dist, fig_trend, fig_ex
    ]

if __name__ == '__main__':
    app.run(debug=True)
