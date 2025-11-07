#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualization_functions.py

Contains reusable functions for generating Plotly visualizations for the
Netflix analysis project.
"""

# -----------------------------------------------------------------------------
# 1. Import Packages and Install Requirements
# -----------------------------------------------------------------------------

import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pycountry
from plotly.subplots import make_subplots


def install_requirements():
    """Installs required packages."""
    print("Installing visualization_functions requirements...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pandas", "plotly"]
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
NETFLIX_BLACK = '#000000'
PLOTLY_TEMPLATE = 'plotly_dark'
COLOR_SCALE_REDS = 'Reds'
COLOR_SCALE_DIVERGING = 'RdBu_r'


# -----------------------------------------------------------------------------
# 3. Basic EDA Plotting Functions
# -----------------------------------------------------------------------------

def get_iso_alpha_3(country_name):
    """Converts a country name to its 3-letter ISO code."""
    # Handle common exceptions
    if country_name == 'United States':
        return 'USA'
    if country_name == 'United Kingdom':
        return 'GBR'
    if country_name == 'South Korea':
        return 'KOR'

    try:
        # Try to find the country by its name
        country = pycountry.countries.get(name=country_name)
        if country:
            return country.alpha_3
        else:
            # If not found, try a fuzzy search
            country = pycountry.countries.search_fuzzy(country_name)
            if country:
                return country[0].alpha_3
        return None  # Return None if no match
    except Exception:
        return None  # Return None on error


def plot_content_type_pie(df, template=PLOTLY_TEMPLATE):
    """Generates a pie chart for Movie vs. TV Show distribution."""
    type_counts = df['type'].value_counts()
    fig = px.pie(
        type_counts,
        values=type_counts.values,
        names=type_counts.index,
        title='Overall Composition: Movies vs. TV Shows',
        color_discrete_map={'Movie': NETFLIX_RED, 'TV Show': NETFLIX_BLACK},
        template=template
    )
    fig.update_layout(title_x=0.5)
    return fig


def plot_content_trend_lines(df, template=PLOTLY_TEMPLATE):
    """Generates a line chart for content added over time by type."""
    content_trends = df.groupby(
        ['year_added', 'type']
    ).size().reset_index(name='Count')
    content_trends = content_trends[content_trends['year_added'] >= 2010]

    fig = px.line(
        content_trends,
        x='year_added',
        y='Count',
        color='type',
        title='Content Volume Added to Netflix (2010-Present)',
        markers=True,
        color_discrete_map={'Movie': NETFLIX_RED, 'TV Show': '#b00000'},
        template=template
    )
    fig.update_layout(
        xaxis_title='Year Added',
        yaxis_title='Number of Titles Added',
        title_x=0.5
    )
    return fig


def plot_top_n_bar(df, column, title, n=15, template=PLOTLY_TEMPLATE,
                     color_scale=COLOR_SCALE_REDS):
    """Generates a horizontal bar chart for top N items in a column."""
    counts = df[column].value_counts().head(n).reset_index()
    counts.columns = [column, 'Count']

    fig = px.bar(
        counts,
        x='Count',
        y=column,
        orientation='h',
        title=f'Top {n} {title}',
        color='Count',
        color_continuous_scale=color_scale,
        template=template
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), title_x=0.5)
    return fig


def plot_choropleth_map(df, template=PLOTLY_TEMPLATE,
                        color_scale=COLOR_SCALE_REDS):
    """Generates a choropleth map of content production by country."""
    country_counts = df['primary_country'].value_counts().reset_index()
    country_counts.columns = ['Country', 'Count']

    # Convert country names to ISO-3 codes
    country_counts['iso_code'] = country_counts['Country'].apply(get_iso_alpha_3)

    # Optional: Check which countries failed to convert
    failed = country_counts[country_counts['iso_code'].isnull()]['Country'].tolist()
    if failed:
        print(f"Warning: Could not find ISO-3 codes for: {failed}")

    fig = px.choropleth(
        country_counts,
        locations="iso_code",
        locationmode="ISO-3",
        color="Count",
        hover_name="Country",
        color_continuous_scale=color_scale,
        title="Netflix Content Production by Country",
        template=template
    )
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type="natural earth"
        ),
        title_x=0.5
    )
    return fig


# -----------------------------------------------------------------------------
# 4. Time-Series and Lag Plotting Functions
# -----------------------------------------------------------------------------

def plot_lag_time_trend(df, template=PLOTLY_TEMPLATE):
    """Plots the average acquisition lag time over the years."""
    avg_lag_by_year = df[df['acquisition_lag'] >= 0].groupby(
        'year_added'
    )['acquisition_lag'].mean().reset_index()
    avg_lag_by_year = avg_lag_by_year[avg_lag_by_year['year_added'] >= 2010]

    fig = px.line(
        avg_lag_by_year,
        x='year_added',
        y='acquisition_lag',
        title='Netflix\'s "Acquisition Lag" Over Time',
        markers=True,
        labels={
            'acquisition_lag': 'Average Lag (Years)',
            'year_added': 'Year Added to Netflix'
        },
        template=template
    )
    fig.update_traces(line=dict(color=NETFLIX_RED))
    fig.update_layout(title_x=0.5)
    return fig


def plot_monthly_heatmap(df, template=PLOTLY_TEMPLATE):
    """Generates a heatmap of content additions by month and year."""
    month_year_data = df.groupby(
        ['year_added', 'month_added']
    ).size().reset_index(name='count')
    month_year_pivot = month_year_data.pivot_table(
        index='month_added',
        columns='year_added',
        values='count',
        fill_value=0
    )

    month_order = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
    month_year_pivot = month_year_pivot.reindex(
        [m for m in month_order if m in month_year_pivot.index]
    )

    fig = px.imshow(
        month_year_pivot,
        title='Heatmap of Additions by Month & Year',
        labels=dict(x="Year", y="Month", color="Titles Added"),
        aspect="auto",
        template=template,
        color_continuous_scale=COLOR_SCALE_REDS
    )
    fig.update_layout(title_x=0.5)
    return fig


# -----------------------------------------------------------------------------
# 5. Advanced & Relational Plotting Functions
# -----------------------------------------------------------------------------

def plot_sankey_collaborations(sankey_df, title='Netflix Co-Production Network',
                             template=PLOTLY_TEMPLATE):
    """
    Generates a Sankey diagram from a DataFrame with 'source', 'target',
    and 'value' columns.
    """
    all_nodes = pd.concat([sankey_df['source'], sankey_df['target']]).unique()
    node_labels = list(all_nodes)
    node_map = {country: index for index, country in enumerate(node_labels)}

    sankey_df['source_idx'] = sankey_df['source'].map(node_map)
    sankey_df['target_idx'] = sankey_df['target'].map(node_map)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=NETFLIX_RED
        ),
        link=dict(
            source=sankey_df['source_idx'],
            target=sankey_df['target_idx'],
            value=sankey_df['value']
        )
    )])
    fig.update_layout(
        title_text=title,
        font_size=10,
        template=template,
        title_x=0.5
    )
    return fig


def plot_bubble_chart(df, x_col, y_col, size_col, color_col, title,
                        template=PLOTLY_TEMPLATE):
    """Generates a configurable bubble chart."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        size=size_col,
        color=color_col,
        hover_name="country",  # Assuming 'country' is the hover label
        log_x=True,
        size_max=60,
        title=title,
        template=template
    )
    fig.update_layout(title_x=0.5)
    return fig


def plot_dual_axis_chart(df, bar_col, line_col, bar_name, line_name, title,
                           template=PLOTLY_TEMPLATE):
    """Generates a dual-axis bar and line chart."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add Bar Chart
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df[bar_col],
            name=bar_name,
            marker_color=NETFLIX_RED
        ),
        secondary_y=False,
    )

    # Add Line Chart
    fig.add_trace(
        go.Line(
            x=df.index,
            y=df[line_col],
            name=line_name,
            marker_color=NETFLIX_BLACK
        ),
        secondary_y=True,
    )

    fig.update_layout(title_text=title, template=template, title_x=0.5)
    fig.update_yaxes(title_text=bar_name, secondary_y=False)
    fig.update_yaxes(title_text=line_name, secondary_y=True)
    return fig


def plot_rating_distribution(df, n=15, template=PLOTLY_TEMPLATE,
                             color_scale=COLOR_SCALE_REDS):
    """Generates a bar chart for the distribution of maturity ratings."""
    print("Plotting distribution of maturity ratings...")
    rating_counts = df['rating'].value_counts().head(n).reset_index()
    rating_counts.columns = ['Rating', 'Count']

    fig = px.bar(
        rating_counts,
        x='Rating',
        y='Count',
        title=f'Top {n} Content Maturity Ratings on Netflix',
        color='Count',
        color_continuous_scale=color_scale,
        template=template
    )
    fig.update_layout(
        xaxis_title='Maturity Rating',
        yaxis_title='Number of Titles',
        title_x=0.5
    )
    return fig


def plot_rating_distribution_boxplot(df, template=PLOTLY_TEMPLATE):
    """
    Generates box plots comparing the distribution of user ratings
    (e.g., 'vote_average') between Movies and TV Shows.
    """
    print("Plotting box plot distribution of user ratings...")

    # Ensure 'vote_average' exists and is numeric
    if 'vote_average' not in df.columns or \
            not pd.api.types.is_numeric_dtype(df['vote_average']):
        print("Warning: 'vote_average' column not found or not numeric. "
              "Skipping box plot.")
        return None

    df_clean = df.dropna(subset=['type', 'vote_average'])

    fig = px.box(
        df_clean,
        x='type',
        y='vote_average',
        color='type',
        title='Distribution of User Ratings: Movies vs. TV Shows',
        labels={'type': 'Content Type',
                'vote_average': 'User Rating (e.g., IMDb)'},
        color_discrete_map={'Movie': NETFLIX_RED, 'TV Show': '#b00000'},
        template=template,
        points="all"  # Show all underlying data points as a "strip plot"
    )
    fig.update_layout(title_x=0.5, showlegend=False)
    return fig


def plot_treemap_top_contributors(df, column_to_explode, title, n=25,
                                  template=PLOTLY_TEMPLATE):
    """
    Generates a treemap for the top N contributors (e.g., directors, actors).

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_to_explode (str): The column name to explode
                                 (e.g., 'director_list', 'cast_list').
        title (str): The title for the chart.
        n (int): The number of top contributors to show.
    """
    print(f"Generating treemap for top {n} {column_to_explode}...")

    if column_to_explode not in df.columns:
        print(f"Warning: Column '{column_to_explode}' not found. "
              "Skipping treemap.")
        return None

    df_exploded = df.explode(column_to_explode)

    # Clean up common unknown values
    df_exploded = df_exploded[
        (df_exploded[column_to_explode].notna()) &
        (df_exploded[column_to_explode] != 'Unknown') &
        (df_exploded[column_to_explode] != 'Unknown Director') &
        (df_exploded[column_to_explode] != 'Unknown Cast')
    ]

    counts = df_exploded[column_to_explode].value_counts().head(n).reset_index()
    counts.columns = ['Contributor', 'Count']

    fig = px.treemap(
        counts,
        path=[px.Constant(title), 'Contributor'],  # Create a root node
        values='Count',
        title=f'Top {n} {title} by Number of Titles',
        color='Count',
        color_continuous_scale=COLOR_SCALE_REDS,
        template=template
    )
    fig.update_layout(title_x=0.5)
    return fig


def plot_popularity_vs_rating_scatter(df, template=PLOTLY_TEMPLATE):
    """
    Generates a scatter plot of 'popularity' vs. 'vote_average'
    with a trendline.
    """
    print("Generating popularity vs. rating scatter plot...")

    # Check for required numeric columns
    required_cols = ['popularity', 'vote_average', 'type']
    if not all(col in df.columns and
               pd.api.types.is_numeric_dtype(df[col])
               for col in required_cols if col != 'type'):
        print("Warning: 'popularity' or 'vote_average' not found or not numeric. "
              "Skipping scatter plot.")
        return None

    df_clean = df.dropna(subset=required_cols).copy()

    # Add a small amount of jitter to prevent overplotting
    df_clean['jittered_pop'] = \
        df_clean['popularity'] + np.random.rand(len(df_clean)) * 0.1
    df_clean['jittered_rating'] = \
        df_clean['vote_average'] + np.random.rand(len(df_clean)) * 0.1

    fig = px.scatter(
        df_clean,
        x='jittered_pop',
        y='jittered_rating',
        color='type',
        trendline="ols",  # Add an "Ordinary Least Squares" regression line
        title='Popularity vs. User Rating (with Trendline)',
        labels={'jittered_pop': 'Popularity Score (Jittered)',
                'jittered_rating': 'User Rating (Jittered)'},
        color_discrete_map={'Movie': NETFLIX_RED,
                            'TV Show': 'rgb(100, 100, 100)'},
        template=template,
        opacity=0.5  # Make points transparent to see dense areas
    )

    fig.update_layout(title_x=0.5)
    return fig


def prepare_moving_average_data(df):
    """
    Prepares time-series data for moving average analysis.
    This function resamples daily data to monthly and calculates MAs.
    """
    print("Preparing data for Moving Average plot...")

    # --- 1. Validate Input ---
    # This analysis requires a datetime 'date_added' and 'show_id'
    if 'date_added' not in df.columns or 'show_id' not in df.columns:
        print("Error: DataFrame must have 'date_added' and 'show_id' columns.")
        return None

    # Ensure date_added is a datetime object and drop any rows without a date
    df_clean = df.copy()
    df_clean['date_added'] = pd.to_datetime(df_clean['date_added'],
                                            errors='coerce')
    df_clean = df_clean.dropna(subset=['date_added'])

    # --- 2. Create Daily Counts ---
    # Group by day, count titles
    daily_additions = df_clean.groupby(
        df_clean['date_added'].dt.date
    )['show_id'].count().reset_index()
    daily_additions['date_added'] = pd.to_datetime(daily_additions['date_added'])

    if daily_additions.empty:
        print("Error: No valid data found after processing dates.")
        return None

    # --- 3. Create Full Date Range (Fill Missing Days) ---
    date_range = pd.date_range(daily_additions['date_added'].min(),
                               daily_additions['date_added'].max())
    daily_additions = daily_additions.set_index('date_added').reindex(
        date_range, fill_value=0
    ).reset_index()
    daily_additions.columns = ['date', 'titles_added']

    # --- 4. Resample to Monthly & Calculate MAs ---
    monthly_additions = daily_additions.set_index(
        'date'
    ).resample('MS')['titles_added'].sum().reset_index()
    monthly_additions.columns = ['month_year', 'titles_added']

    # Calculate 6-month and 12-month moving average
    monthly_additions['MA_6_month'] = \
        monthly_additions['titles_added'].rolling(window=6).mean()
    monthly_additions['MA_12_month'] = \
        monthly_additions['titles_added'].rolling(window=12).mean()

    return monthly_additions


def plot_moving_average_trend(monthly_additions_df, template=PLOTLY_TEMPLATE):
    """
    Plots a bar chart of monthly additions with 6-month and 12-month
    moving average lines.
    """
    print("Generating Moving Average plot...")

    if monthly_additions_df is None or monthly_additions_df.empty:
        print("...Skipped Moving Average plot (no data).")
        return None

    fig_ma = go.Figure()

    # Add the raw monthly additions as a bar chart
    fig_ma.add_trace(go.Bar(
        x=monthly_additions_df['month_year'],
        y=monthly_additions_df['titles_added'],
        name='Monthly Additions',
        marker_color='lightblue'
    ))

    # Add the moving average lines
    fig_ma.add_trace(go.Scatter(
        x=monthly_additions_df['month_year'],
        y=monthly_additions_df['MA_6_month'],
        name='6-Month Moving Average',
        line=dict(color='blue', width=2)
    ))
    fig_ma.add_trace(go.Scatter(
        x=monthly_additions_df['month_year'],
        y=monthly_additions_df['MA_12_month'],
        name='12-Month Moving Average',
        line=dict(color='red', width=2)
    ))

    fig_ma.update_layout(
        title='Monthly Additions with Moving Average (Trend Detection)',
        xaxis_title='Date',
        yaxis_title='Number of Titles Added',
        template=template,
        title_x=0.5  # Center the title
    )

    return fig_ma


def plot_content_age_strategy(df_age_plot, template=PLOTLY_TEMPLATE):
    """
    Plots a 100% stacked area chart (groupnorm='percent') to show
    the shift in content age strategy over time.
    """
    print("Generating Content Age Strategy plot...")

    if df_age_plot is None or df_age_plot.empty:
        print("...Skipped Content Age Strategy plot (no data).")
        return None

    fig = px.area(
        df_age_plot,
        x='year_added',
        y='count',
        color='content_age_category',
        title='Shift in Content Age Strategy (Proportional)',
        labels={
            'addition_year': 'Year Added',
            'content_age_category': 'Content Age',
            'count': 'Percentage of Titles'
        },
        # This normalizes each year's data to sum to 100%
        groupnorm='percent'
    )

    fig.update_layout(
        template=template,
        title_x=0.5
    )

    return fig


def main():
    """Runs when the script is executed directly."""
    print("visualization_functions.py is a module and should be imported.")
    print("It does not produce output when run directly.")


if __name__ == '__main__':
    main()