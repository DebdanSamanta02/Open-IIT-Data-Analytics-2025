#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
statistical_analysis.py

Handles advanced statistical modeling, strategic analysis (HHI, Quadrant),
time-series decomposition, and network analysis.
"""
# -----------------------------------------------------------------------------
# 1. Import Packages and Install Requirements
# -----------------------------------------------------------------------------

import subprocess
import sys
from collections import Counter, defaultdict

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose


def install_requirements():
    """Installs required packages."""
    print("Installing statistical_analysis requirements...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pandas", "plotly",
             "scipy", "statsmodels", "networkx", "mlxtend"]
        )
        print("Packages installed successfully.")
    except Exception as e:
        print(f"Error installing packages: {e}")


# Uncomment the line below to run the installation
# install_requirements()


# -----------------------------------------------------------------------------
# 2. Global Variables and Configuration
# -----------------------------------------------------------------------------

# --- Color and Theme Definitions ---
NETFLIX_RED = '#E50914'
PLOTLY_TEMPLATE = 'plotly_dark'

# --- Analysis Parameters ---
EMERGING_HUB_MIN_TITLES = 25
QUADRANT_START_YEAR = 2016
QUADRANT_END_YEAR = 2021
APRIORI_MIN_SUPPORT = 0.01
RULES_MIN_LIFT = 1.1
NETWORK_MIN_COLLABS = 5


# -----------------------------------------------------------------------------
# 3. Time-Series Analysis
# -----------------------------------------------------------------------------

def plot_time_series_decomposition(df, template=PLOTLY_TEMPLATE):
    """Performs and plots seasonal decomposition on monthly additions."""
    print("Performing time-series decomposition...")

    # Resample daily additions to get total monthly additions
    monthly_additions = df.set_index('date_added').resample('MS').size()
    monthly_additions.name = 'titles_added'

    # We need at least 2 full periods (24 months)
    if len(monthly_additions) < 24:
        print("Not enough data for 12-month seasonal decomposition.")
        return None

    decomposition = seasonal_decompose(
        monthly_additions, model='additive', period=12
    )

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=('Observed', 'Trend', 'Seasonal', 'Residual')
    )
    fig.add_trace(go.Scatter(
        x=decomposition.observed.index, y=decomposition.observed, name='Observed'
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=decomposition.trend.index, y=decomposition.trend, name='Trend'
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=decomposition.seasonal.index, y=decomposition.seasonal, name='Seasonal'
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=decomposition.resid.index, y=decomposition.resid, name='Residual',
        mode='markers'
    ), row=4, col=1)

    fig.update_layout(
        title='Time Series Decomposition of Monthly Additions',
        height=800,
        template=template,
        title_x=0.5
    )
    return fig


# -----------------------------------------------------------------------------
# 4. Strategic Genre Analysis (HHI, Long Tail, Quadrant)
# -----------------------------------------------------------------------------

def calculate_hhi(df, genre_col='primary_genre'):
    """Calculates the Herfindahl-Hirschman Index (HHI) for genre concentration."""
    print("Calculating HHI for genre concentration...")
    genre_counts = df[genre_col].value_counts()
    total_titles = genre_counts.sum()
    genre_shares = genre_counts / total_titles
    hhi = ((genre_shares ** 2).sum() * 10000)

    print(f"Genre Concentration (HHI): {hhi:.2f}")
    if hhi > 2500:
        print("Interpretation: HIGH concentration. "
              "Over-reliance on a few genres.")
    elif hhi > 1500:
        print("Interpretation: MODERATE concentration.")
    else:
        print("Interpretation: LOW concentration. Well-diversified portfolio.")
    return hhi


def plot_long_tail(df, genre_col='primary_genre', template=PLOTLY_TEMPLATE):
    """Plots the long-tail distribution of genres."""
    print("Plotting genre long-tail distribution...")
    genre_counts = df[genre_col].value_counts().reset_index()
    genre_counts.columns = ['genre', 'count']
    genre_counts = genre_counts.sort_values('count', ascending=False)
    genre_counts['rank'] = range(1, len(genre_counts) + 1)

    fig = px.bar(
        genre_counts,
        x='rank',
        y='count',
        hover_name='genre',
        title='Genre Long-Tail Distribution',
        log_y=True,
        template=template
    )
    fig.update_layout(
        xaxis_title='Genre Rank',
        yaxis_title='Number of Titles (Log Scale)',
        title_x=0.5
    )
    return fig


def perform_quadrant_analysis(netflix_imdb_df,
                              start_year=QUADRANT_START_YEAR,
                              end_year=QUADRANT_END_YEAR):
    """
    Combines genre supply growth (CAGR) and quality (IMDb rating)
    for quadrant analysis.
    """
    print("Performing quadrant analysis...")

    # 1. Get genre counts per year
    genre_year = (
        netflix_imdb_df.explode('genre_list')
        .groupby(['genre_list', 'release_year'])
        .size()
        .reset_index(name='titles')
    )
    genre_pivot = genre_year.pivot(
        index='genre_list', columns='release_year', values='titles'
    ).fillna(0).astype(int)

    # 2. Calculate CAGR (Compound Annual Growth Rate)
    start_vals = genre_pivot.get(start_year,
                                 pd.Series(0, index=genre_pivot.index))
    end_vals = genre_pivot.get(end_year,
                               pd.Series(0, index=genre_pivot.index))

    n_years = end_year - start_year
    cagr = pd.Series(0.0, index=genre_pivot.index, dtype=float)

    # Vectorized calculation for CAGR
    if n_years > 0:
        valid_mask = (start_vals > 0) & (end_vals > 0)
        cagr.loc[valid_mask] = (
            (end_vals[valid_mask] / start_vals[valid_mask]) ** (1 / n_years) - 1
        )

    cagr.name = 'cagr_pct'
    cagr_df = cagr.reset_index().rename(columns={'genre_list': 'genre'})
    cagr_df['cagr_pct'] *= 100

    # 3. Get average rating per genre
    genre_avg_rating = (
        netflix_imdb_df.explode('genre_list')
        .groupby('genre_list')['averageRating']
        .mean()
        .reset_index()
    )
    genre_avg_rating = genre_avg_rating.rename(
        columns={'genre_list': 'genre', 'averageRating': 'avg_rating'}
    )

    # 4. Get total volume per genre
    genre_volume = (
        netflix_imdb_df.explode('genre_list')['genre_list']
        .value_counts()
        .reset_index()
    )
    genre_volume.columns = ['genre', 'volume']

    # 5. Combine metrics
    quadrant_df = cagr_df.merge(genre_avg_rating, on='genre', how='inner')
    quadrant_df = quadrant_df.merge(genre_volume, on='genre', how='inner')

    # Filter out very small genres
    quadrant_df = quadrant_df[quadrant_df['volume'] >= 50].copy()

    return quadrant_df


def plot_quadrant_analysis(quadrant_df, template=PLOTLY_TEMPLATE):
    """Plots the strategic genre quadrant."""
    print("Plotting quadrant analysis...")

    cagr_median = quadrant_df['cagr_pct'].median()
    rating_median = quadrant_df['avg_rating'].median()

    fig = px.scatter(
        quadrant_df,
        x='cagr_pct',
        y='avg_rating',
        size='volume',
        hover_name='genre',
        text='genre',
        size_max=60,
        title=f'Netflix Genre Quadrant Analysis '
              f'(Growth vs. Quality {QUADRANT_START_YEAR}-{QUADRANT_END_YEAR})',
        template=template,
        labels={'cagr_pct': 'Supply Growth (CAGR %)',
                'avg_rating': 'Average IMDb Rating'}
    )

    # Add Quadrant Lines
    fig.add_vline(x=cagr_median, line_dash="dash", line_color="grey")
    fig.add_hline(y=rating_median, line_dash="dash", line_color="grey")

    fig.update_traces(textposition='top center')
    fig.update_layout(title_x=0.5)
    return fig


# -----------------------------------------------------------------------------
# 5. Network and Association Analysis
# -----------------------------------------------------------------------------

def build_collaboration_network(df, min_collabs=NETWORK_MIN_COLLABS):
    """Builds a NetworkX graph of director-actor collaborations."""
    print("Building collaboration network...")
    G = nx.Graph()
    co_occurrence = defaultdict(int)

    for _, row in df.iterrows():
        directors = row.get('director_list', [])
        actors = row.get('cast_list', [])

        # Use top 10 actors to keep graph manageable
        actors = actors[:10]

        for director in directors:
            if director == 'Unknown Director':
                continue
            G.add_node(director, type='director')
            for actor in actors:
                if actor == 'Unknown Cast':
                    continue
                G.add_node(actor, type='actor')
                co_occurrence[(director, actor)] += 1

    # Add edges based on minimum collaborations
    for (director, actor), weight in co_occurrence.items():
        if weight >= min_collabs:
            G.add_edge(director, actor, weight=weight)

    print(f"Network built: {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges.")
    return G


def get_network_centrality(graph):
    """Calculates degree, betweenness, and eigenvector centrality."""
    print("Calculating network centrality...")
    centrality = pd.DataFrame({
        'degree': nx.degree_centrality(graph),
        'betweenness': nx.betweenness_centrality(graph),
        'eigenvector': nx.eigenvector_centrality(graph, max_iter=500)
    })
    return centrality.sort_values('degree', ascending=False)


def run_association_rules(genre_matrix, min_support=APRIORI_MIN_SUPPORT,
                          min_lift=RULES_MIN_LIFT):
    """Runs Apriori and Association Rules to find genre pairings."""
    print("Running Apriori algorithm for frequent itemsets...")
    genre_matrix_bool = genre_matrix.astype(bool)
    frequent_sets = apriori(
        genre_matrix_bool, min_support=min_support, use_colnames=True
    )

    print("Generating association rules...")
    rules = association_rules(frequent_sets, metric="lift",
                              min_threshold=min_lift)
    rules = rules.sort_values(['lift', 'confidence'], ascending=False)

    return rules


def main():
    """Main function to be run when script is executed directly."""
    print("statistical_analysis.py is a module and should be imported.")
    print("It does not produce output when run directly.")


if __name__ == '__main__':
    main()