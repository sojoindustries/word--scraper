import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, callback_context
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from rake_nltk import Rake
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import os
import re
from datetime import datetime, timedelta
from typing import Iterable
from scipy import stats

# ------------------------
# Define stopword sets
# ------------------------
# These common domain terms appear too frequently to be informative
DOMAIN_STOP = {
    'line', 'shift', 'run', 'case', 'production', 'machine', 'ran',
    'cans', 'cases', 'product', 'products', 'packaging', 'package'
}
# Combine our domain-specific list with NLTK's English stopwords
STOP_WORDS = set(stopwords.words('english')) | DOMAIN_STOP

# ------------------------
# Load and preprocess data
# ------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = "/Users/cole/PycharmProjects/EOSDashboard/Data/CB_EOS_Data_Scraper.csv"
df = pd.read_csv(file_path)

# Clean and transform data
df['CUSTRECORD_EOS_MEMO'] = df['CUSTRECORD_EOS_MEMO'].fillna('')  # remove NaNs
df['DATE'] = pd.to_datetime(df['CUSTRECORD_EOS_DATE'])  # Convert dates to datetime objects
df['MONTH'] = df['DATE'].dt.strftime('%Y-%m')  # Extract month for time-based analysis
df['YEAR'] = df['DATE'].dt.year  # Extract year for time-based analysis
df['WEEKDAY'] = df['DATE'].dt.day_name()  # Extract day name for day-of-week analysis

# Extract unique product lines for dropdown filter
product_lines = sorted(df['BINNUMBER'].dropna().unique())


# ------------------------
# Text analysis functions
# ------------------------
def extract_ngrams(texts, ngram_range=(1, 1), top_n=20):
    """
    Return the top_n most frequent ngrams in texts.

    Parameters:
    -----------
    texts : iterable
        Collection of text documents
    ngram_range : tuple, default=(1, 1)
        The lower and upper boundary of the range of n-values for n-grams to be extracted
    top_n : int, default=20
        Number of top ngrams to return

    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns 'phrase' and 'count'
    """
    vectorizer = CountVectorizer(
        lowercase=True,
        token_pattern=r'\b[a-zA-Z]+\b',  # Only consider alphabetic words
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
    """
    Compute TF-IDF scores and return top_n terms.

    TF-IDF (Term Frequency-Inverse Document Frequency) highlights terms that are
    important to a document in a collection of documents.

    Parameters:
    -----------
    texts : iterable
        Collection of text documents
    ngram_range : tuple, default=(1, 1)
        The lower and upper boundary of the range of n-values for n-grams
    top_n : int, default=20
        Number of top terms to return

    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns 'phrase' and 'tfidf'
    """
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


def extract_pmi_bigrams(texts, top_n=20, freq_filter=5):
    """
    Compute Pointwise Mutual Information (PMI) for bigrams.

    PMI measures how much more often words co-occur than would be expected by chance.
    Higher PMI indicates stronger association between words.

    Parameters:
    -----------
    texts : iterable
        Collection of text documents
    top_n : int, default=20
        Number of top bigrams to return
    freq_filter : int, default=5
        Minimum frequency threshold for bigrams

    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns 'bigram' and 'pmi'
    """
    # Extract tokens that are alphabetic and not in stopwords
    tokens = [t for doc in texts for t in doc.lower().split()
              if t.isalpha() and t not in STOP_WORDS]

    finder = BigramCollocationFinder.from_words(tokens)
    finder.apply_freq_filter(freq_filter)  # Ignore rare bigrams
    scored = finder.score_ngrams(BigramAssocMeasures.pmi)

    df_bigrams = pd.DataFrame(scored, columns=['bigram', 'pmi'])
    df_bigrams['bigram'] = df_bigrams['bigram'].str.join(' ')  # Join tuple words with space
    return df_bigrams.head(top_n)


def extract_rake_phrases(texts, top_n=20):
    """
    Rapid Automatic Keyword Extraction (RAKE) for multi-word phrases.

    RAKE is designed to extract keywords from text by analyzing the frequency of
    word appearance and co-occurrences.

    Parameters:
    -----------
    texts : iterable
        Collection of text documents
    top_n : int, default=20
        Number of top phrases to return

    Returns:
    --------
    pandas.DataFrame
        DataFrame with column 'rake_phrase'
    """
    rake = Rake(stopwords=STOP_WORDS)
    rake.extract_keywords_from_sentences(texts)
    phrases = rake.get_ranked_phrases()[:top_n]
    return pd.DataFrame({'rake_phrase': phrases})


def extract_lda_topics(texts, n_topics=5, n_top_words=5):
    """
    Latent Dirichlet Allocation to identify dominant topics.

    LDA is a generative statistical model that explains sets of observations
    through unobserved groups that explain why some parts of the data are similar.

    Parameters:
    -----------
    texts : iterable
        Collection of text documents
    n_topics : int, default=5
        Number of topics to extract
    n_top_words : int, default=5
        Number of top words to include for each topic

    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns 'topic' and 'top_terms'
    """
    vectorizer = CountVectorizer(stop_words=list(STOP_WORDS))
    X = vectorizer.fit_transform(texts)

    # Train LDA model
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=0,
        learning_method='online'
    )
    lda.fit(X)

    # Extract topics
    terms = vectorizer.get_feature_names_out()
    topics = []
    for i, comp in enumerate(lda.components_):
        # Get indices of top words for this topic, sorted by importance
        top_word_indices = comp.argsort()[:-n_top_words - 1:-1]
        top_terms = [terms[idx] for idx in top_word_indices]
        topics.append({
            'topic': i + 1,
            'top_terms': ', '.join(top_terms)
        })

    return pd.DataFrame(topics)


def extract_clusters(texts: Iterable[str], n_clusters: int = 5) -> pd.DataFrame:
    """
    K-Means clustering of documents (memos). Returns cluster sizes.

    This function groups similar text documents together based on their content.

    Parameters:
    -----------
    texts : iterable
        Collection of text documents
    n_clusters : int, default=5
        Number of clusters to form

    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns 'cluster' and 'count'
    """
    vectorizer = TfidfVectorizer(stop_words=list(STOP_WORDS))
    X = vectorizer.fit_transform(texts)

    # Adjust number of clusters if necessary
    if X.shape[0] < n_clusters:
        if X.shape[0] == 0:
            return pd.DataFrame(columns=['cluster', 'count'])
        n_clusters = max(2, X.shape[0])

    # Perform clustering
    n_samples = X.shape[0]
    n_clusters = min(n_samples, n_clusters)  # Ensure n_clusters is not greater than n_samples
    model = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    labels = model.fit_predict(X)

    # Count documents in each cluster
    counts = np.unique(labels, return_counts=True)[1]
    return pd.DataFrame({
        'cluster': np.unique(labels),
        'count': counts
    }).sort_values('count', ascending=False)


def create_wordcloud(texts):
    """
    Generate a word cloud visualization from text data.

    Parameters:
    -----------
    texts : iterable
        Collection of text documents

    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure containing the wordcloud image
    """
    if not isinstance(texts, (pd.Series, list)) or len(texts) == 0 or not any(texts):
        return px.scatter(title="No data available for wordcloud")

    try:
        # Combine all texts into a single string
        text = ' '.join([str(t) for t in texts if isinstance(t, str) and t])

        if not text.strip():
            return px.scatter(title="No valid text available for wordcloud")

        # Generate wordcloud
        wc = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=STOP_WORDS,
            max_words=100,
            collocations=True
        ).generate(text)

        # Convert matplotlib figure to plotly
        fig = plt.figure(figsize=(10, 7))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')

        # Convert to image bytes
        img_data = BytesIO()
        plt.savefig(img_data, format='png', bbox_inches='tight', pad_inches=0)
        img_data.seek(0)

        # Create plotly figure from image
        img_str = base64.b64encode(img_data.read()).decode()

        # Create a plotly figure with the image
        fig = go.Figure()
        fig.add_layout_image(
            dict(
                source=f'data:image/png;base64,{img_str}',
                x=0,
                y=0,
                xref="paper",
                yref="paper",
                sizex=1,
                sizey=1,
                sizing="stretch",
                layer="below"
            )
        )
        fig.update_layout(
            title="Word Cloud Visualization",
            autosize=True,
            height=500,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )

        return fig

    except Exception as e:
        print(f"Wordcloud generation failed: {e}")
        return px.scatter(title=f"Wordcloud generation failed: {str(e)}")


def generate_pca_scatter(texts):
    """
    Create PCA visualization of document similarity.

    This reduces the high-dimensional text data to 2D for visualization.

    Parameters:
    -----------
    texts : iterable
        Collection of text documents

    Returns:
    --------
    plotly.graph_objects.Figure
        Scatter plot of documents in 2D PCA space
    """
    if len(texts) < 3:
        return px.scatter(title="Not enough data for PCA visualization")

    vectorizer = TfidfVectorizer(stop_words=list(STOP_WORDS))
    try:
        # Convert texts to TF-IDF vectors
        X = vectorizer.fit_transform(texts)

        # Apply PCA to reduce to 2 dimensions
        pca = PCA(n_components=2)
        components = pca.fit_transform(X.toarray())

        # Calculate variance explained by each component
        var_explained = pca.explained_variance_ratio_

        # Create scatter plot
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
    """
    Extract sentiment-specific keywords from texts.

    Parameters:
    -----------
    texts : iterable
        Collection of text documents
    positive_terms : list
        List of words indicating positive sentiment
    negative_terms : list
        List of words indicating negative sentiment

    Returns:
    --------
    dict
        Dictionary with positive and negative keyword counts
    """
    pos_counts = {}
    neg_counts = {}

    for text in texts:
        if not isinstance(text, str): continue

        # Count positive term occurrences
        for term in positive_terms:
            if re.search(r'\b' + term + r'\b', text, re.IGNORECASE):
                pos_counts[term] = pos_counts.get(term, 0) + 1

        # Count negative term occurrences
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


# Define sentiment terms
POSITIVE_TERMS = ['good', 'great', 'excellent', 'perfect', 'resolved', 'fixed', 'improved', 'success']
NEGATIVE_TERMS = ['issue', 'problem', 'error', 'fail', 'down', 'jam', 'stuck', 'break', 'stopped']

# ------------------------
# Pre-calculate metrics per product line
# ------------------------
line_metrics = {}
for product_line in product_lines:
    texts = df[df['BINNUMBER'] == product_line]['CUSTRECORD_EOS_MEMO']
    line_metrics[product_line] = {
        'unigrams': extract_ngrams(texts),
        'bigrams': extract_ngrams(texts, (2, 2)),
        'tfidf_unigrams': extract_tfidf_terms(texts),
        'tfidf_bigrams': extract_tfidf_terms(texts, (2, 2)),
        'pmi_bigrams': extract_pmi_bigrams(texts),
        'rake_phrases': extract_rake_phrases(texts),
        'lda_topics': extract_lda_topics(texts),
        'clusters': extract_clusters(texts),
        'sentiment': extract_sentiment_keywords(texts, POSITIVE_TERMS, NEGATIVE_TERMS)
    }

# ------------------------
# Build Dash app layout
# ------------------------
app = dash.Dash(__name__)
app.layout = html.Div([
    html.Div([
        html.H2('End of Shift Reporting Dashboard'),
        html.Label('Select Product Line:'),
        dcc.Dropdown(id='line-filter', options=product_lines, value=product_lines[0]),
        html.Hr(),
        html.Label('Date Range:'),
        dcc.DatePickerRange(
            id='date-range',
            min_date_allowed=df['DATE'].min(),
            max_date_allowed=df['DATE'].max(),
            start_date=df['DATE'].min(),
            end_date=df['DATE'].max()
        )
    ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),

    html.Div([
        html.H3('Text Analysis Results'),
        *[dcc.Loading(dcc.Graph(id=graph_id), type='circle') for graph_id in [
            'unigram_summary', 'bigram_summary', 'tfidf_unigrams', 'tfidf_bigrams',
            'pmi_bigrams', 'rake_phrases', 'lda_topics', 'cluster_summary',
            'downtime_jams_chart', 'timeseries_chart', 'sentiment_chart',
            'weekday_chart', 'wordcloud_chart', 'pca_scatter', 'downtime_issues_overtime'
        ]]
    ], style={'width': '75%', 'display': 'inline-block', 'padding': '20px'})
])


# ------------------------
# Callback to update all figures
# ------------------------
@app.callback(
    [Output(graph_id, 'figure') for graph_id in [
        'unigram_summary', 'bigram_summary', 'tfidf_unigrams', 'tfidf_bigrams',
        'pmi_bigrams', 'rake_phrases', 'lda_topics', 'cluster_summary',
        'downtime_jams_chart', 'timeseries_chart', 'sentiment_chart',
        'weekday_chart', 'wordcloud_chart', 'pca_scatter', 'downtime_issues_overtime'
    ]],
    [Input('line-filter', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_dashboard(selected_line, start_date, end_date):
    # Get pre-computed metrics for the selected line
    m = line_metrics[selected_line]

    # Filter dataframe by selected line and date range
    line_df = df[df['BINNUMBER'] == selected_line]
    if start_date and end_date:
        line_df = line_df[(line_df['DATE'] >= start_date) & (line_df['DATE'] <= end_date)]

    # Helper to build a table figure
    def create_table(df, title):
        return go.Figure(data=[go.Table(
            header=dict(values=list(df.columns), fill_color='lightgrey', align='left'),
            cells=dict(values=[df[col] for col in df.columns], align='left')
        )]).update_layout(title=title, height=400)

    # Create basic table figures
    fig_ug = create_table(m['unigrams'], f"{selected_line} — Top Unigrams")
    fig_bg = create_table(m['bigrams'], f"{selected_line} — Top Bigrams")
    fig_tu = create_table(m['tfidf_unigrams'], f"{selected_line} — Top TF-IDF Unigrams")
    fig_tb = create_table(m['tfidf_bigrams'], f"{selected_line} — Top TF-IDF Bigrams")
    fig_pm = create_table(m['pmi_bigrams'], f"{selected_line} — Top PMI Bigrams")
    fig_rk = create_table(m['rake_phrases'], f"{selected_line} — Top RAKE Phrases")
    fig_ld = create_table(m['lda_topics'], f"{selected_line} — LDA Topics")
    fig_cl = create_table(m['clusters'], f"{selected_line} — Cluster Sizes")

    # Create time-based chart (weekly trends)
    time_df = line_df.groupby(pd.Grouper(key='DATE', freq='W')).size().reset_index(name='count')
    fig_ts = px.line(
        time_df,
        x='DATE',
        y='count',
        title=f"{selected_line} — Weekly Comment Trends",
        labels={'DATE': 'Week', 'count': 'Number of Comments'}
    )

    # Create day-of-week analysis with error bars
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_df = line_df.groupby(['WEEKDAY', line_df['DATE'].dt.isocalendar().week]).size().reset_index(name='count')

    # Calculate mean and standard error for each day
    weekday_stats = weekday_df.groupby('WEEKDAY').agg(
        mean_count=('count', 'mean'),
        std_err=('count', lambda x: stats.sem(x, nan_policy='omit') if len(x) > 1 else 0)
    ).reset_index()

    # Order days correctly
    weekday_stats['WEEKDAY'] = pd.Categorical(weekday_stats['WEEKDAY'], categories=weekday_order)
    weekday_stats = weekday_stats.sort_values('WEEKDAY')

    # Create bar chart with error bars
    fig_wd = px.bar(
        weekday_stats,
        x='WEEKDAY',
        y='mean_count',
        error_y='std_err',
        title=f"{selected_line} — Comments by Day of Week (with Statistical Significance)",
        labels={'WEEKDAY': 'Day', 'mean_count': 'Average Number of Comments'}
    )

    # Create sentiment analysis chart
    pos_df = m['sentiment']['positive'].head(10)
    neg_df = m['sentiment']['negative'].head(10)

    # Combine positive and negative terms
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

    # Create PMI heatmap for top 10 bigrams
    top_bigrams = m['pmi_bigrams'].head(10)
    fig_hm = px.bar(
        top_bigrams,
        x='bigram',
        y='pmi',
        title=f"{selected_line} — Top Bigram PMI Scores",
        labels={'bigram': 'Bigram', 'pmi': 'PMI Score'}
    )

    # Create wordcloud
    fig_wc = create_wordcloud(line_df['CUSTRECORD_EOS_MEMO'])

    # Create PCA scatter plot
    fig_pca = generate_pca_scatter(line_df['CUSTRECORD_EOS_MEMO'])

    # Analyze downtime mentions
    downtime_pattern = 'down|jam|stuck|break|stopped|failure|error'
    line_df['is_downtime'] = line_df['CUSTRECORD_EOS_MEMO'].str.contains(downtime_pattern, case=False)
    downtime_count = line_df['is_downtime'].sum()
    total_count = len(line_df)

    fig_dj = px.pie(
        values=[downtime_count, total_count - downtime_count],
        names=['Downtime/Issue Mentions', 'Other'],
        title=f"{selected_line} — Downtime/Issue Mentions ({downtime_count}/{total_count} reports)",
        color_discrete_sequence=['red', 'lightblue']
    )

    # Create line graph of downtime/issues compared to other metrics over time
    # First, aggregate by week
    line_df['week'] = line_df['DATE'].dt.isocalendar().week
    line_df['year'] = line_df['DATE'].dt.year
    line_df['yearweek'] = line_df['year'].astype(str) + '-' + line_df['week'].astype(str)

    # Count downtime issues and other reports per week
    weekly_issues = line_df.groupby(['yearweek', 'DATE']).agg(
        downtime_issues=('is_downtime', 'sum'),
        other_reports=('is_downtime', lambda x: (~x).sum()),
        total_reports=('is_downtime', 'count')
    ).reset_index()

    # Sort by date for the line chart
    weekly_issues = weekly_issues.sort_values('DATE')

    # Create the time series chart comparing downtime to other issues
    fig_downtime_time = px.line(
        weekly_issues,
        x='DATE',
        y=['downtime_issues', 'other_reports'],
        title=f"{selected_line} — Downtime Issues vs Other Reports Over Time",
        labels={
            'DATE': 'Week',
            'value': 'Number of Reports',
            'variable': 'Report Type'
        },
        color_discrete_map={
            'downtime_issues': 'red',
            'other_reports': 'blue'
        }
    )

    fig_downtime_time.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Reports",
        legend_title="Report Type"
    )

    return [fig_ug, fig_bg, fig_tu, fig_tb, fig_pm, fig_rk, fig_ld, fig_cl,
            fig_dj, fig_ts, fig_sent, fig_wd, fig_wc, fig_pca, fig_downtime_time]


if __name__ == '__main__':
    app.run(debug=True)


