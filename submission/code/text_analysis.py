#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
text_analysis.py

Handles all Natural Language Processing (NLP) tasks, including text cleaning,
sentiment analysis, N-gram extraction, topic modeling (LDA), and word clouds.
"""

# -----------------------------------------------------------------------------
# 1. Import Packages
# -----------------------------------------------------------------------------
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from wordcloud import WordCloud


# -----------------------------------------------------------------------------
# 2. Global Variables and Configuration
# -----------------------------------------------------------------------------

# --- Color and Theme Definitions ---
PLOTLY_TEMPLATE = 'plotly_dark'

# --- NLP Parameters ---
LDA_TOPICS = 5
TOP_N_WORDS_PER_TOPIC = 10
TOP_N_NGRAMS = 20
GLOBAL_STOP_WORDS = ['movie', 'tv', 'show', 'netflix', 'series', 'film']


def _get_stop_words():
    """
    Private helper function to get the NLTK stopword list and add custom words.
    """
    try:
        stop_words = set(stopwords.words('english'))
        stop_words.update(GLOBAL_STOP_WORDS)
        return list(stop_words)
    except LookupError:
        print("NLTK stopwords not found. Downloading...")
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        stop_words.update(GLOBAL_STOP_WORDS)
        return list(stop_words)


# --- Global Constants ---
STOP_WORDS = _get_stop_words()


# -----------------------------------------------------------------------------
# 3. Setup and Text Cleaning
# -----------------------------------------------------------------------------

def install_requirements():
    """Installs required packages."""
    print("Installing text_analysis requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas",
                               "plotly", "scikit-learn", "nltk", "textblob",
                               "wordcloud", "matplotlib"])
        print("Packages installed successfully.")
    except Exception as e:
        print(f"Error installing packages: {e}")


def download_nltk_data():
    """Downloads necessary NLTK models (stopwords, punkt)."""
    print("Downloading NLTK data (stopwords, punkt)...")
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        print("NLTK data downloaded successfully.")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")


def clean_text(text):
    """
    Cleans a single string of text:
    - Lowercase
    - Remove emails and URLs
    - Remove non-alphabetic characters
    - Remove stop words and short words
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic chars
    text = ' '.join([word for word in text.split()
                     if word not in STOP_WORDS and len(word) > 2])
    return text


def preprocess_text_column(df, text_col='description'):
    """
    Applies text cleaning to an entire DataFrame column.
    Assumes NLTK data has already been downloaded.
    """
    print("Preprocessing text column...")
    df['clean_description'] = df[text_col].apply(clean_text)
    return df


# -----------------------------------------------------------------------------
# 4. Sentiment and N-gram Analysis
# -----------------------------------------------------------------------------

def get_sentiment(df, text_col='clean_description'):
    """Calculates sentiment polarity for a text column."""
    print("Calculating sentiment...")
    df['sentiment'] = df[text_col].apply(lambda t: TextBlob(t).sentiment.polarity)
    return df


def get_top_ngrams(corpus, n=2, top_k=TOP_N_NGRAMS):
    """Extracts top K N-grams from a text corpus."""
    print(f"Extracting top {top_k} {n}-grams...")
    vectorizer = CountVectorizer(ngram_range=(n, n),
                                 stop_words=STOP_WORDS,
                                 min_df=3)
    X_counts = vectorizer.fit_transform(corpus)
    frequencies = np.asarray(X_counts.sum(axis=0)).ravel()
    features = np.array(vectorizer.get_feature_names_out())

    # Get indices of top_k frequencies in descending order
    top_indices = frequencies.argsort()[::-1][:top_k]

    ngram_list = list(zip(features[top_indices], frequencies[top_indices]))
    return ngram_list


def plot_ngrams(ngram_list, title, template=PLOTLY_TEMPLATE):
    """Plots N-grams as a horizontal bar chart."""
    words = [w for w, _ in ngram_list][::-1]  # Reverse for ascending plot
    counts = [c for _, c in ngram_list][::-1]
    fig = px.bar(x=counts, y=words, orientation='h', title=title,
                 template=template)
    fig.update_layout(xaxis_title='Count', yaxis_title='N-gram', title_x=0.5)
    return fig


# -----------------------------------------------------------------------------
# 5. Topic Modeling (LDA)
# -----------------------------------------------------------------------------

def perform_lda_topic_modeling(corpus, n_topics=LDA_TOPICS):
    """Performs LDA topic modeling on a text corpus."""
    print(f"Performing LDA with {n_topics} topics...")
    vectorizer = CountVectorizer(max_df=0.9, min_df=10,
                                 max_features=5000,
                                 stop_words=STOP_WORDS)
    X_counts = vectorizer.fit_transform(corpus)

    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method='online',
        random_state=42,
        n_jobs=-1
    )
    lda_model.fit(X_counts)

    feature_names = vectorizer.get_feature_names_out()
    return lda_model, X_counts, vectorizer, feature_names


def get_lda_topics(model, feature_names, n_top_words=TOP_N_WORDS_PER_TOPIC):
    """Extracts the top words for each LDA topic."""
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[::-1][:n_top_words]
        topics[topic_idx] = [(feature_names[i], float(topic[i]))
                             for i in top_indices]
    return topics


def plot_topic_trends_over_time(df, n_topics=LDA_TOPICS,
                                template=PLOTLY_TEMPLATE):
    """
    Assigns topics to each document and plots their prevalence over time.
    Assumes 'clean_description' and 'year_added' columns exist.
    """
    print("Plotting topic trends over time...")
    lda_model, X_counts, vectorizer, feature_names = \
        perform_lda_topic_modeling(df['clean_description'], n_topics)

    # Create user-friendly topic names (top 3 words)
    topic_map = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        # Get top 3 words
        top_words = [feature_names[i] for i in topic.argsort()[::-1][:3]]
        topic_map[topic_idx] = f"Topic {topic_idx}: {', '.join(top_words)}"

    # Assign dominant topic
    topic_distributions = lda_model.transform(X_counts)
    df['dominant_topic_id'] = np.argmax(topic_distributions, axis=1)
    df['Topic'] = df['dominant_topic_id'].map(topic_map)

    # Aggregate by year
    topic_trends = df.groupby(['year_added', 'Topic']).size().reset_index(
        name='Count')
    total_counts_per_year = topic_trends.groupby(
        'year_added')['Count'].transform('sum')
    topic_trends['percent'] = topic_trends['Count'] / total_counts_per_year

    # Plot
    fig = px.bar(
        topic_trends,
        x='year_added',
        y='percent',
        color='Topic',
        title='<b>The "Netflix Genome": Changing Themes Over Time (2015-2020)</b>',
        labels={'year_added': 'Year Content Was Added',
                'percent': 'Percentage of Content'},
        template=template,
        barmode='stack',
        hover_data={'percent': ':.1%'}
    )
    fig.update_layout(
        yaxis=dict(tickformat=',.0%'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right",
                    x=1),
        title_x=0.5
    )
    return fig


# -----------------------------------------------------------------------------
# 6. Word Cloud Generation
# -----------------------------------------------------------------------------

def generate_wordcloud(text_corpus, save_path=None):
    """Generates and optionally saves a word cloud image."""
    print("Generating word cloud...")
    full_text = " ".join(text_corpus.tolist())

    wordcloud_obj = WordCloud(
        width=1400,
        height=700,
        background_color='white',
        stopwords=STOP_WORDS
    ).generate(full_text)

    plt.figure(figsize=(14, 7))
    plt.imshow(wordcloud_obj, interpolation='bilinear')
    plt.axis('off')
    plt.title('Global Wordcloud (Descriptions)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Word cloud saved to {save_path}")

    plt.show()


# -----------------------------------------------------------------------------
# 7. Main Execution Block
# -----------------------------------------------------------------------------

def main():
    """
    Main function to run demonstrations of the text_analysis module.
    Installs requirements, downloads NLTK data, and runs a small demo.
    """
    print("Running text_analysis.py as main script (demonstration)...")

    # --- Setup ---
    # Uncomment the line below if you need to install packages
    # install_requirements()
    download_nltk_data()

    # --- Create a dummy DataFrame for testing ---
    dummy_data = {
        'description': [
            "A young woman finds love in a new city.",
            "A detective solves a dark mystery.",
            "This documentary explores the life of a famous musician.",
            "A young man joins a team of superheroes to save the world.",
            "A detective drama about a mysterious crime."
        ],
        'year_added': [2019, 2020, 2019, 2021, 2020]
    }
    df_dummy = pd.DataFrame(dummy_data)

    # --- Run preprocessing ---
    df_dummy = preprocess_text_column(df_dummy)
    print("\nPreprocessed Data:\n", df_dummy)

    # --- Run N-gram analysis ---
    bigrams = get_top_ngrams(df_dummy['clean_description'], n=2, top_k=5)
    print("\nTop bigrams:", bigrams)

    # --- Run topic modeling ---
    print("\nGenerating topic modeling plot...")
    fig = plot_topic_trends_over_time(df_dummy, n_topics=2)
    fig.show()
    print("Topic modeling plot generated.")

    # --- Run Word Cloud ---
    print("\nGenerating word cloud...")
    generate_wordcloud(df_dummy['clean_description'])


if __name__ == '__main__':
    main()