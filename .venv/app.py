import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from rake_nltk import Rake

DOMAIN_STOP = {'line', 'shift', 'run', 'case', 'production', 'machine', 'ran', 'cans', 'cases'}
STOP_WORDS = set(stopwords.words('english')) | DOMAIN_STOP

df = pd.read_csv('Data/CB_EOS_Data_Scraper.csv')
df['CUSTRECORD_EOS_MEMO'] = df['CUSTRECORD_EOS_MEMO'].fillna('')
df['DATE'] = pd.to_datetime(df['CUSTRECORD_EOS_DATE'])

product_lines = sorted(df['BINNUMBER'].dropna().unique())


def get_top_ngrams(texts, ngram_range=(1, 1), top_n=20):
    vect = CountVectorizer(
        lowercase=True,
        token_pattern=r'\b[a-zA-Z]+\b',
        stop_words=list(STOP_WORDS),
        ngram_range=ngram_range,
        min_df=1
    )
    X = vect.fit_transform(texts)
    counts = np.asarray(X.sum(axis=0)).ravel()
    feats = vect.get_feature_names_out()
    return pd.DataFrame({'phrase': feats, 'count': counts}) \
        .sort_values('count', ascending=False) \
        .head(top_n)


def get_top_tfidf(texts, ngram_range=(1, 1), top_n=20):
    doc_count = len(texts)
    if doc_count == 0:
        return pd.DataFrame(columns=['phrase', 'tfidf'])
    try:
        vect = TfidfVectorizer(
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]+\b',
            stop_words=list(STOP_WORDS),
            ngram_range=ngram_range,
            min_df=1
        )

        X = vect.fit_transform(texts)
        scores = np.asarray(X.sum(axis=0)).ravel()
        feats = vect.get_feature_names_out()

        return pd.DataFrame({'phrase': feats, 'tfidf': scores}) \
            .sort_values('tfidf', ascending=False) \
            .head(top_n)
    except Exception as e:
        print(f"Warning: TF-IDF calculation failed with error: {e}")
        return pd.DataFrame(columns=['phrase', 'tfidf'])


def get_top_pmi_bigrams(texts, top_n=20, freq_filter=5):
    tokens = [t for doc in texts for t in doc.lower().split() if t.isalpha() and t not in STOP_WORDS]
    finder = BigramCollocationFinder.from_words(tokens)
    finder.apply_freq_filter(freq_filter)
    scored = finder.score_ngrams(BigramAssocMeasures.pmi)
    dfb = pd.DataFrame(scored, columns=['bigram', 'pmi'])
    dfb['bigram'] = dfb['bigram'].str.join(' ')
    return dfb.head(top_n)


def get_top_rake_phrases(texts, top_n=20):
    rake = Rake(stopwords=STOP_WORDS)
    rake.extract_keywords_from_sentences(texts)
    phrases = rake.get_ranked_phrases()[:top_n]
    return pd.DataFrame({'rake_phrase': phrases})


def perform_lda(texts, n_topics=5, n_top_words=5):
    vect = CountVectorizer(stop_words=list(STOP_WORDS))
    X = vect.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(X)
    terms = vect.get_feature_names_out()
    rows = []
    for i, comp in enumerate(lda.components_):
        top = [terms[idx] for idx in comp.argsort()[:-n_top_words - 1:-1]]
        rows.append({'topic': i + 1, 'top_terms': ', '.join(top)})
    return pd.DataFrame(rows)


def perform_clustering(texts, n_clusters=5):
    vect = TfidfVectorizer(stop_words=list(STOP_WORDS))
    X = vect.fit_transform(texts)

    n_samples = X.shape[0]

    if n_samples < n_clusters:
        if n_samples == 0:

            return pd.DataFrame(columns=['cluster', 'count'])
        else:

            n_clusters = n_samples

    km = KMeans(n_clusters=n_clusters, random_state=0)
    labels = km.fit_predict(X)
    sizes = pd.DataFrame({
        'cluster': np.unique(labels),
        'count': np.unique(labels, return_counts=True)[1]
    })
    return sizes


metrics = {}
for line in product_lines:
    sub = df[df['BINNUMBER'] == line]['CUSTRECORD_EOS_MEMO']
    metrics[line] = {
        'unigrams': get_top_ngrams(sub),
        'bigrams': get_top_ngrams(sub, (2, 2)),
        'tf_uni': get_top_tfidf(sub),
        'tf_bi': get_top_tfidf(sub, (2, 2)),
        'pmi': get_top_pmi_bigrams(sub),
        'rake': get_top_rake_phrases(sub),
        'lda': perform_lda(sub),
        'clusters': perform_clustering(sub)
    }

app = dash.Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.H2('EOS Reporting Dashboard'),
        html.Label('Product Line'),
        dcc.Dropdown(product_lines, product_lines[0], id='line-filter'),
    ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
    html.Div([
        *[dcc.Loading(dcc.Graph(id=id_), type='circle') for id_ in [
            'unigrams-table', 'bigrams-table', 'tfidf-uni-table', 'tfidf-bi-table',
            'pmi-table', 'rake-table', 'lda-table', 'clusters-table',
            'downtime-jams-chart'
        ]]
    ], style={'width': '75%', 'display': 'inline-block', 'padding': '20px'})
])


@app.callback(
    Output('unigrams-table', 'figure'),
    Output('bigrams-table', 'figure'),
    Output('tfidf-uni-table', 'figure'),
    Output('tfidf-bi-table', 'figure'),
    Output('pmi-table', 'figure'),
    Output('rake-table', 'figure'),
    Output('lda-table', 'figure'),
    Output('clusters-table', 'figure'),
    Output('downtime-jams-chart', 'figure'),
    Input('line-filter', 'value')
)
def update(line):
    m = metrics[line]

    def mk_tbl(df, title):
        return go.Figure(data=[go.Table(
            header=dict(values=list(df.columns), fill_color='lightgrey'),
            cells=dict(values=[df[col] for col in df.columns])
        )]).update_layout(title=title)

    fig_ug = mk_tbl(m['unigrams'], f"{line} — Top 20 Unigrams")
    fig_bg = mk_tbl(m['bigrams'], f"{line} — Top 20 Bigrams")
    fig_tu = mk_tbl(m['tf_uni'], f"{line} — Top 20 TF–IDF Unigrams")
    fig_tb = mk_tbl(m['tf_bi'], f"{line} — Top 20 TF–IDF Bigrams")
    fig_pm = mk_tbl(m['pmi'], f"{line} — Top 20 PMI Bigrams")
    fig_rk = mk_tbl(m['rake'], f"{line} — Top 20 RAKE Phrases")
    fig_ld = mk_tbl(m['lda'], f"{line} — LDA Topics")
    fig_cl = mk_tbl(m['clusters'], f"{line} — K-Means Cluster Sizes")

    sub = df[df['BINNUMBER'] == line]
    x = sub['CUSTRECORD_EOS_MEMO'].str.contains('downtime', case=False) \
        .groupby(sub['DATE'].dt.date).sum().reset_index(name='downtime')
    y = sub['CUSTRECORD_EOS_MEMO'].str.contains('jams', case=False) \
        .groupby(sub['DATE'].dt.date).sum().reset_index(name='jams')
    both = pd.merge(x, y, on='DATE')
    both['DATE'] = pd.to_datetime(both['DATE'])
    dfm = both.melt(id_vars='DATE', value_vars=['downtime', 'jams'],
                    var_name='keyword', value_name='count')
    fig_cp = px.bar(dfm, x='DATE', y='count', color='keyword', barmode='group',
                    title=f"{line}: downtime vs jams mentions") \
        .update_layout(xaxis_title='Date', yaxis_title='Count')

    return fig_ug, fig_bg, fig_tu, fig_tb, fig_pm, fig_rk, fig_ld, fig_cl, fig_cp


if __name__ == '__main__':
    app.run(debug=True)