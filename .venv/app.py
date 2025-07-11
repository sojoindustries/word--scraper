import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from rake_nltk import Rake

# Define domain-specific and general stop words
DOMAIN_STOP = {'line', 'shift', 'run', 'case', 'production', 'machine', 'ran', 'cans', 'cases'}
STOP_WORDS = set(stopwords.words('english')) | DOMAIN_STOP

# Load and preprocess dataset
df = pd.read_csv('Data/CB_EOS_Data_Scraper.csv')
df['CUSTRECORD_EOS_MEMO'] = df['CUSTRECORD_EOS_MEMO'].fillna('')
df['DATE'] = pd.to_datetime(df['CUSTRECORD_EOS_DATE'])

# Get list of product lines
product_lines = sorted(df['BINNUMBER'].dropna().unique())

# Extract top unigrams or bigrams based on count
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

# Extract top terms using TF-IDF
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
            .sort_values('tfidf', ascending=False).head(top_n)
    except Exception as e:
        # This might fail if the documents are empty or too uniform
        print(f"TF-IDF computation failed: {e}")
        return pd.DataFrame(columns=['phrase', 'tfidf'])

# Identify bigrams with high pointwise mutual information (PMI)
def extract_pmi_bigrams(texts, top_n=20, freq_filter=5):
    tokens = [t for doc in texts for t in doc.lower().split() if t.isalpha() and t not in STOP_WORDS]
    finder = BigramCollocationFinder.from_words(tokens)
    finder.apply_freq_filter(freq_filter)
    scored = finder.score_ngrams(BigramAssocMeasures.pmi)
    df_bigrams = pd.DataFrame(scored, columns=['bigram', 'pmi'])
    df_bigrams['bigram'] = df_bigrams['bigram'].str.join(' ')
    return df_bigrams.head(top_n)

# Use RAKE to extract key phrases from text
def extract_rake_phrases(texts, top_n=20):
    rake = Rake(stopwords=STOP_WORDS)
    rake.extract_keywords_from_sentences(texts)
    phrases = rake.get_ranked_phrases()[:top_n]
    return pd.DataFrame({'rake_phrase': phrases})

# Perform Latent Dirichlet Allocation (LDA) topic modeling
def extract_lda_topics(texts, n_topics=5, n_top_words=5):
    vectorizer = CountVectorizer(stop_words=list(STOP_WORDS))
    X = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    lda.fit(X)
    terms = vectorizer.get_feature_names_out()
    topics = []
    for i, comp in enumerate(lda.components_):
        top_terms = [terms[idx] for idx in comp.argsort()[:-n_top_words - 1:-1]]
        topics.append({'topic': i + 1, 'top_terms': ', '.join(top_terms)})
    return pd.DataFrame(topics)

# Cluster similar documents using K-Means
def extract_clusters(texts, n_clusters=5):
    vectorizer = TfidfVectorizer(stop_words=list(STOP_WORDS))
    X = vectorizer.fit_transform(texts)
    if X.shape[0] < n_clusters:
        if X.shape[0] == 0:
            return pd.DataFrame(columns=['cluster', 'count'])
        n_clusters = X.shape[0]
    model = KMeans(n_clusters=n_clusters, random_state=0)
    labels = model.fit_predict(X)
    label_counts = pd.DataFrame({'cluster': np.unique(labels), 'count': np.unique(labels, return_counts=True)[1]})
    return label_counts

# Compute NLP metrics for each product line
line_metrics = {}
for product_line in product_lines:
    filtered_df = df[df['BINNUMBER'] == product_line]['CUSTRECORD_EOS_MEMO']
    line_metrics[product_line] = {
        'unigrams': extract_ngrams(filtered_df),
        'bigrams': extract_ngrams(filtered_df, (2, 2)),
        'tfidf_unigrams': extract_tfidf_terms(filtered_df),
        'tfidf_bigrams': extract_tfidf_terms(filtered_df, (2, 2)),
        'pmi_bigrams': extract_pmi_bigrams(filtered_df),
        'rake_phrases': extract_rake_phrases(filtered_df),
        'lda_topics': extract_lda_topics(filtered_df),
        'clusters': extract_clusters(filtered_df)
    }

# Initialize Dash app
app = dash.Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.H2('EOS Reporting Dashboard'),
        html.Label('Select Product Line'),
        dcc.Dropdown(product_lines, product_lines[0], id='line-filter'),
    ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),

    html.Div([
        *[dcc.Loading(dcc.Graph(id=graph_id), type='circle') for graph_id in [
            'unigram_summary', 'bigram_summary', 'tfidf_unigrams', 'tfidf_bigrams',
            'pmi_bigrams', 'rake_phrases', 'lda_topics', 'cluster_summary', 'downtime_jams_chart']]
    ], style={'width': '75%', 'display': 'inline-block', 'padding': '20px'})
])

@app.callback(
    Output('unigram_summary', 'figure'),
    Output('bigram_summary', 'figure'),
    Output('tfidf_unigrams', 'figure'),
    Output('tfidf_bigrams', 'figure'),
    Output('pmi_bigrams', 'figure'),
    Output('rake_phrases', 'figure'),
    Output('lda_topics', 'figure'),
    Output('cluster_summary', 'figure'),
    Output('downtime_jams_chart', 'figure'),
    Input('line-filter', 'value')
)
def update_dashboard(selected_line):
    metrics = line_metrics[selected_line]

    # Helper to build a simple Plotly table
    def create_table(df, title):
        return go.Figure(data=[go.Table(
            header=dict(values=list(df.columns), fill_color='lightgrey'),
            cells=dict(values=[df[col] for col in df.columns])
        )]).update_layout(title=title)

    fig_ug = create_table(metrics['unigrams'], f"{selected_line} — Top Unigrams")
    fig_bg = create_table(metrics['bigrams'], f"{selected_line} — Top Bigrams")
    fig_tu = create_table(metrics['tfidf_unigrams'], f"{selected_line} — Top TF-IDF Unigrams")
    fig_tb = create_table(metrics['tfidf_bigrams'], f"{selected_line} — Top TF-IDF Bigrams")
    fig_pm = create_table(metrics['pmi_bigrams'], f"{selected_line} — Top PMI Bigrams")
    fig_rk = create_table(metrics['rake_phrases'], f"{selected_line} — Top RAKE Phrases")
    fig_ld = create_table(metrics['lda_topics'], f"{selected_line} — LDA Topics")
    fig_cl = create_table(metrics['clusters'], f"{selected_line} — Cluster Sizes")

    # Plot keyword trends over time
    sub_df = df[df['BINNUMBER'] == selected_line]
    downtime = sub_df['CUSTRECORD_EOS_MEMO'].str.contains('downtime', case=False) \
        .groupby(sub_df['DATE'].dt.date).sum().reset_index(name='downtime')
    jams = sub_df['CUSTRECORD_EOS_MEMO'].str.contains('jams', case=False) \
        .groupby(sub_df['DATE'].dt.date).sum().reset_index(name='jams')
    trend_df = pd.merge(downtime, jams, on='DATE')
    trend_df['DATE'] = pd.to_datetime(trend_df['DATE'])
    melted = trend_df.melt(id_vars='DATE', value_vars=['downtime', 'jams'],
                           var_name='keyword', value_name='count')
    fig_cp = px.bar(melted, x='DATE', y='count', color='keyword', barmode='group',
                    title=f"{selected_line}: Downtime vs Jams Mentions") \
        .update_layout(xaxis_title='Date', yaxis_title='Count')

    return fig_ug, fig_bg, fig_tu, fig_tb, fig_pm, fig_rk, fig_ld, fig_cl, fig_cp

if __name__ == '__main__':
    app.run(debug=True)
