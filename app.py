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

import json
import os


with open(os.path.join(os.path.dirname(__file__), "secrets.json")) as f:
    _secrets = json.load(f)


CONSUMER_KEY    = _secrets["ConsumerKey"]
CONSUMER_SECRET = _secrets["ConsumerSecret"]
TOKEN           = _secrets["ProdTokenID"]
TOKEN_SECRET    = _secrets["ProdTokenSecret"]


def fetch_data():

    url = (
        "https://7501774.suitetalk.api.netsuite.com"
        "/services/rest/query/v1/suiteql?limit=1000&offset=0"
    )
    query = {"q": "SELECT * FROM customrecord_end_of_shift ORDER BY created DESC"}

    auth = OAuth1(
        CONSUMER_KEY,
        client_secret=CONSUMER_SECRET,
        resource_owner_key=TOKEN,
        resource_owner_secret=TOKEN_SECRET,
        signature_method='HMAC-SHA256'
    )
    headers = {
        'Content-Type': 'application/json',
        'Accept':       'application/json',
        'Prefer':       'transient'
    }

    resp = requests.post(url, headers=headers, json=query, auth=auth)
    if resp.status_code != 200:
        raise Exception(f"SuiteQL API Error {resp.status_code}: {resp.text}")

    items = resp.json().get('items', [])
    return pd.DataFrame.from_records(items)



df = fetch_data()
df['CUSTRECORD_EOS_MEMO'] = df['CUSTRECORD_EOS_MEMO'].fillna('')
df['DATE']   = pd.to_datetime(df['CUSTRECORD_EOS_DATE'])
df['MONTH']  = df['DATE'].dt.strftime('%Y-%m')
df['YEAR']   = df['DATE'].dt.year
df['WEEKDAY']= df['DATE'].dt.day_name()

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

from nltk.metrics import BigramAssocMeasures
from wordcloud import WordCloud
import base64
from io import BytesIO
import re
from datetime import datetime, timedelta
from typing import Iterable
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

DOMAIN_STOP = {
    'line', 'shift', 'run', 'case', 'production', 'machine', 'ran',
    'cans', 'cases', 'product', 'products', 'packaging', 'package'
}
STOP_WORDS = nltk_stopwords | DOMAIN_STOP

df = fetch_data()
df['CUSTRECORD_EOS_MEMO'] = df['CUSTRECORD_EOS_MEMO'].fillna('')
df['DATE']   = pd.to_datetime(df['CUSTRECORD_EOS_DATE'])
df['MONTH']  = df['DATE'].dt.strftime('%Y-%m')
df['YEAR']   = df['DATE'].dt.year
df['WEEKDAY']= df['DATE'].dt.day_name()


df['SENT_SCORE'] = df['CUSTRECORD_EOS_MEMO'].apply(
    lambda x: sentiment_analyzer.polarity_scores(str(x))['compound'] if isinstance(x, str) else 0
)

product_lines = sorted(df['BINNUMBER'].dropna().unique())


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


def generate_pca_scatter(texts):
    if len(texts) < 3:
        return px.scatter(title="Not enough data for PCA visualization")

    vectorizer = TfidfVectorizer(stop_words=list(STOP_WORDS))
    try:
        X = vectorizer.fit_transform(texts)

        pca = PCA(n_components=2)
        components = pca.fit_transform(X.toarray())

        var_explained = pca.explained_variance_ratio_

        fig = px.scatter(
            x=components[:, 0],
            y=components[:, 1],
            labels={
                'x': f'PC1 ({var_explained[0]:.2%} variance)',
                'y': f'PC2 ({var_explained[1]:.2%} variance)'
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
    texts = df[df['BINNUMBER'] == product_line]['CUSTRECORD_EOS_MEMO']
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
    df = fetch_data()

    navbar = dbc.NavbarSimple(
        brand="End-of-Shift Dashboard",
        color="dark",
        dark=True,
        fluid=True,
    )

    controls = dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id="line-filter",  # Changed from "line-dropdown"
                options=[{"label": l, "value": l} for l in product_lines],  # Changed to use product_lines
                placeholder="Select Product Line",
                value=product_lines[0] if product_lines else None,  # Set default value
            ),
            md=4,
        ),
        dbc.Col(
            dcc.DatePickerRange(
                id="date-range",  # Changed from "date-picker"
                start_date=df["DATE"].min(),  # Changed from "date"
                end_date=df["DATE"].max(),  # Changed from "date"
                display_format="YYYY-MM-DD",
            ),
            md=8,
        ),
    ], className="my-4")

    # KPI row
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
                dbc.CardHeader("Cluster Summary"),
                dbc.CardBody(dcc.Graph(id="cluster_summary")),
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

    charts4 = dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Sentiment Analysis"),
                dbc.CardBody(dcc.Graph(id="sentiment_chart")),
            ]),
            md=6,
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Weekday Analysis"),
                dbc.CardBody(dcc.Graph(id="weekday_chart")),
            ]),
            md=6,
        ),
    ], className="mb-4")

    charts5 = dbc.Row([
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Word Cloud"),
                dbc.CardBody(dcc.Graph(id="wordcloud_chart")),
            ]),
            md=6,
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("PCA Document Similarity"),
                dbc.CardBody(dcc.Graph(id="pca_scatter")),
            ]),
            md=6,
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
            md=6,
        ),
        dbc.Col(
            dbc.Card([
                dbc.CardHeader("Monthly Volume"),
                dbc.CardBody(dcc.Graph(id="monthly-volume-chart")),
            ]),
            md=6,
        ),
    ], className="mb-4")

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
        charts7
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
    Output('weekday_chart', 'figure'),
    Output('wordcloud_chart', 'figure'),
    Output('pca_scatter', 'figure'),
    Output('sentiment-dist-chart', 'figure'),
    Output('sentiment-trend-chart', 'figure'),
    Output('extremes-chart', 'figure'),
    Output('monthly-volume-chart', 'figure'),
    Input('line-filter', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date')
)

def update_dashboard(selected_line, start_date, end_date):
    m = line_metrics[selected_line]

    line_df = df[df['BINNUMBER'] == selected_line]
    if start_date and end_date:
        line_df = line_df[(line_df['DATE'] >= start_date) & (line_df['DATE'] <= end_date)]

    def create_table(df, title):
        return go.Figure(data=[go.Table(
            header=dict(values=list(df.columns), fill_color='lightgrey', align='left'),
            cells=dict(values=[df[col] for col in df.columns], align='left')
        )]).update_layout(title=title, height=400)

    total_memos = len(line_df)
    avg_sent = line_df['SENT_SCORE'].mean()
    downtime_memos = line_df['CUSTRECORD_EOS_MEMO'] \
        .str.contains('down|jam|stuck|break|stopped|failure|error', case=False).sum()
    kpis = [
        html.Div([html.H4('Total Memos'), html.P(total_memos)]),
        html.Div([html.H4('Avg Sentiment'), html.P(f"{avg_sent:.2f}")]),
        html.Div([html.H4('Downtime Memos'), html.P(downtime_memos)])
    ]

    def top_extremes(df_line):
        df_line = df_line.copy()
        df_line['score'] = df_line['CUSTRECORD_EOS_MEMO'] \
            .apply(lambda t: sentiment_analyzer.polarity_scores(t)['compound'])
        top_pos = df_line.nlargest(5, 'score')[['DATE', 'CUSTRECORD_EOS_MEMO', 'score']]
        top_neg = df_line.nsmallest(5, 'score')[['DATE', 'CUSTRECORD_EOS_MEMO', 'score']]
        return top_pos, top_neg

    fig_bg = create_table(m['bigrams'], f"{selected_line} — Top Bigrams")
    fig_tb = create_table(m['tfidf_bigrams'], f"{selected_line} — Top TF-IDF Bigrams")
    fig_ld = create_table(m['lda_topics'], f"{selected_line} — LDA Topics")
    fig_cl = create_table(m['clusters'], f"{selected_line} — Cluster Sizes")

    time_df = line_df.groupby(pd.Grouper(key='DATE', freq='W')).size().reset_index(name='count')
    fig_ts = px.line(
        time_df,
        x='DATE',
        y='count',
        title=f"{selected_line} — Weekly Comment Trends",
        labels={'DATE': 'Week', 'count': 'Number of Comments'}
    )

    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_df = line_df.groupby(['WEEKDAY', line_df['DATE'].dt.isocalendar().week]).size().reset_index(name='count')

    weekday_stats = weekday_df.groupby('WEEKDAY').agg(
        mean_count=('count', 'mean'),
        std_err=('count', lambda x: stats.sem(x, nan_policy='omit') if len(x) > 1 else 0)
    ).reset_index()

    weekday_stats['WEEKDAY'] = pd.Categorical(weekday_stats['WEEKDAY'], categories=weekday_order)
    weekday_stats = weekday_stats.sort_values('WEEKDAY')

    fig_wd = px.bar(
        weekday_stats,
        x='WEEKDAY',
        y='mean_count',
        error_y='std_err',
        title=f"{selected_line} — Comments by Day of Week (with Statistical Significance)",
        labels={'WEEKDAY': 'Day', 'mean_count': 'Average Number of Comments'}
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

    fig_wc = create_wordcloud(line_df['CUSTRECORD_EOS_MEMO'])

    # ── NEW DISTRIBUTION BLOCK (Density + KDE) ──────────────────────
    # extract raw compound scores
    scores = np.array([
        sentiment_analyzer.polarity_scores(txt)['compound']
        for txt in line_df['CUSTRECORD_EOS_MEMO']
    ])

    # handle empty case
    if scores.size == 0:
        fig_dist = go.Figure()
    else:
        # KDE smoothing
        kde = gaussian_kde(scores)
        x_grid = np.linspace(scores.min(), scores.max(), 200)
        y_kde = kde(x_grid)

        # build density histogram + KDE
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
    # ────────────────────────────────────────────────────────────────

    trend_df = (
        line_df.set_index('DATE')['SENT_SCORE']
        .resample('M').mean()
        .reset_index(name='avg_score')
    )
    fig_trend = px.line(
        trend_df, x='DATE', y='avg_score', markers=True,
        title='Monthly Avg Sentiment',
        labels={'avg_score': 'Avg VADER Score'}
    )

    # ── NEW TOP EXTREMES BLOCK (Dates-only) ─────────────────────────
    top_pos, top_neg = top_extremes(line_df)

    # add short date labels
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
    # ────────────────────────────────────────────────────────────────

    vol_df = (
        line_df.set_index('DATE')
        .resample('M').size()
        .reset_index(name='count')
    )
    fig_vol = px.line(
        vol_df, x='DATE', y='count', markers=True,
        title='Monthly Memo Volume',
        labels={'count': 'Number of Memos'}
    )

    fig_pca = generate_pca_scatter(line_df['CUSTRECORD_EOS_MEMO'])

    downtime_pattern = 'down|jam|stuck|break|stopped|failure|error'
    line_df['is_downtime'] = line_df['CUSTRECORD_EOS_MEMO'].str.contains(downtime_pattern, case=False)
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
        kpis,
        fig_bg,
        fig_tb,
        fig_ld,
        fig_cl,
        fig_pie,
        fig_ts,
        fig_sent,
        fig_wd,
        fig_wc,
        fig_pca,
        fig_dist,
        fig_trend,
        fig_ex,
        fig_vol
    ]


if __name__ == '__main__':
    app.run(debug=True)
