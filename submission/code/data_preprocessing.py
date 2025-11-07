#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_preprocessing.py

Handles all data loading, cleaning, merging, and feature engineering for the
Netflix analysis project.
"""
# -----------------------------------------------------------------------------
# 1. Import Packages and Install Requirements
# -----------------------------------------------------------------------------

import os
import subprocess
import sys

import pandas as pd


def install_requirements():
    """Installs required packages."""
    print("Installing data_preprocessing requirements...")
    try:
        # Removed 'rapidfuzz' as it's not imported or used in this file
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pandas"]
        )
        print("Packages installed successfully.")
    except Exception as e:
        print(f"Error installing packages: {e}")


# Uncomment the line below to run the installation
# install_requirements()


# -----------------------------------------------------------------------------
# 2. Global Variables and Configuration
# -----------------------------------------------------------------------------

# --- Directory and File Paths ---
DATA_DIR = 'data'
NETFLIX_2021_PATH = os.path.join(DATA_DIR, 'netflix_titles.csv')
NETFLIX_MOVIES_2025_PATH = os.path.join(
    DATA_DIR, 'netflix_movies_detailed_up_to_2025.csv'
)
NETFLIX_TV_2025_PATH = os.path.join(
    DATA_DIR, 'netflix_tv_shows_detailed_up_to_2025.csv'
)
IMDB_BASICS_PATH = os.path.join(DATA_DIR, 'title.basics.tsv')
IMDB_RATINGS_PATH = os.path.join(DATA_DIR, 'title.ratings.tsv')


# -----------------------------------------------------------------------------
# 3. Data Loading Functions
# -----------------------------------------------------------------------------

def load_netflix_datasets():
    """Loads the 2021 and 2025 Netflix datasets from CSV files."""
    print("Loading Netflix datasets...")
    try:
        df_2021 = pd.read_csv(NETFLIX_2021_PATH)
        df_movies_2025 = pd.read_csv(NETFLIX_MOVIES_2025_PATH)
        df_tv_2025 = pd.read_csv(NETFLIX_TV_2025_PATH)
        print("Netflix datasets loaded successfully.")
        return df_2021, df_movies_2025, df_tv_2025
    except FileNotFoundError as e:
        print(f"Error: Dataset file not found. {e}")
        print("Please make sure the files are in the 'data/' directory.")
        return None, None, None


def load_and_prep_imdb():
    """Loads and merges the IMDb basics and ratings datasets."""
    print("Loading IMDb datasets (basics and ratings)...")
    try:
        basics = pd.read_csv(
            IMDB_BASICS_PATH,
            sep='\t',
            na_values='\\N',
            low_memory=False,
            dtype={'tconst': str, 'startYear': str, 'runtimeMinutes': str}
        )
        ratings = pd.read_csv(
            IMDB_RATINGS_PATH,
            sep='\t',
            na_values='\\N',
            low_memory=False,
            dtype={'tconst': str, 'averageRating': float, 'numVotes': float}
        )

        # Merge basics and ratings
        imdb_df = basics.merge(ratings, on='tconst', how='inner')
        imdb_df['startYear'] = pd.to_numeric(imdb_df['startYear'],
                                             errors='coerce')

        print(f"IMDb data loaded and merged. Total records: {len(imdb_df)}")
        return imdb_df

    except FileNotFoundError as e:
        print(f"Error: IMDb file not found. {e}")
        print("Please make sure 'title.basics.tsv' and 'title.ratings.tsv' "
              "are in 'data/'.")
        return None


# -----------------------------------------------------------------------------
# 4. Data Cleaning and Parsing Functions
# -----------------------------------------------------------------------------

def parse_multi_value_column(series, fill_na_value='Unknown'):
    """Splits a comma-separated string series into a list of strings."""
    series = series.fillna(fill_na_value).astype(str)
    return series.apply(lambda x: [i.strip() for i in x.split(',')])


def clean_2021_dataset(df):
    """Cleans the 2021 Netflix dataset."""
    print("Cleaning 2021 dataset...")
    df['country'] = df['country'].fillna('Unknown')
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')
    df['rating'] = df['rating'].fillna('Unrated')

    # Clean date and extract info
    df['date_added'] = pd.to_datetime(
        df['date_added'].astype(str).str.strip(), errors='coerce'
    )

    # Handle missing genres (critical for analysis)
    df = df.dropna(subset=['listed_in', 'date_added']).copy()

    # Parse multi-value columns
    df.loc[:, 'genre_list'] = parse_multi_value_column(
        df['listed_in'], fill_na_value=''
    )
    df.loc[:, 'director_list'] = parse_multi_value_column(df['director'])
    df.loc[:, 'cast_list'] = parse_multi_value_column(df['cast'])
    df.loc[:, 'country_list'] = parse_multi_value_column(df['country'])

    return df


def clean_2025_dataset(df_movies, df_tv):
    """Cleans and combines the 2025 movies and TV datasets."""
    print("Cleaning 2025 datasets...")
    df_movies_copy = df_movies.copy()
    df_tv_copy = df_tv.copy()
    df_movies_copy.loc[:, 'type'] = 'Movie'
    df_tv_copy.loc[:, 'type'] = 'TV Show'

    df = pd.concat([df_movies_copy, df_tv_copy], ignore_index=True)

    # Drop duplicates
    df = df.drop_duplicates(subset=['title', 'release_year'],
                            keep='first').copy()

    # Impute missing values
    df['director'] = df['director'].fillna('Unknown Director')
    df['cast'] = df['cast'].fillna('Unknown Cast')
    df['country'] = df['country'].fillna('Unknown Country')
    df['rating'] = df['rating'].fillna('Not Rated')
    df['description'] = df['description'].fillna('No description available')
    df = df.dropna(subset=['genres', 'release_year']).copy()

    # Parse multi-value columns
    df['director_list'] = parse_multi_value_column(df['director'])
    df['cast_list'] = parse_multi_value_column(df['cast'])
    df['genre_list'] = parse_multi_value_column(df['genres'])
    df['country_list'] = parse_multi_value_column(df['country'])

    return df


# -----------------------------------------------------------------------------
# 5. Feature Engineering and Merging
# -----------------------------------------------------------------------------

def engineer_netflix_features(df):
    """Engineers time-based features and primary categories."""
    print("Engineering features (year_added, lag_time, etc.)...")

    # Ensure date_added is datetime
    df.loc[:, 'date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    df = df.dropna(subset=['date_added'])

    # Extract time-based features
    df.loc[:, 'year_added'] = df['date_added'].dt.year
    df.loc[:, 'month_added'] = df['date_added'].dt.month_name()
    df.loc[:, 'quarter_added'] = df['date_added'].dt.quarter
    df.loc[:, 'day_of_week_added'] = df['date_added'].dt.day_name()

    # Ensure release_year is numeric
    df.loc[:, 'release_year'] = pd.to_numeric(df['release_year'],
                                              errors='coerce')
    df = df.dropna(subset=['release_year'])

    # Calculate acquisition lag
    df.loc[:, 'acquisition_lag'] = df['year_added'] - df['release_year']

    # Extract primary categories
    df.loc[:, 'primary_genre'] = df['genre_list'].apply(
        lambda x: x[0] if x else 'Unknown'
    )
    df.loc[:, 'primary_country'] = df['country_list'].apply(
        lambda x: x[0] if x else 'Unknown'
    )

    # Clean up primary country names
    df.loc[:, 'primary_country'] = df['primary_country'].replace(
        'United States of America', 'United States'
    )

    return df


def create_genre_matrix(df):
    """Creates a one-hot encoded genre matrix from the 'genres_list'."""
    print("Creating one-hot encoded genre matrix...")
    # Import locally as it's an optional, heavy dependency
    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer()
    genre_matrix = pd.DataFrame(
        mlb.fit_transform(df['genre_list']),
        columns=mlb.classes_,
        index=df.index
    )
    print(f"Genre matrix created with {len(mlb.classes_)} unique genres.")
    return genre_matrix, mlb.classes_


def map_imdb_to_netflix(netflix_df, imdb_df):
    """Merges Netflix data with IMDb ratings using exact title/year matching."""
    print("Mapping IMDb ratings to Netflix titles...")

    # Prepare IMDb data for merge
    imdb_to_merge = imdb_df[
        ['primaryTitle', 'startYear', 'averageRating', 'numVotes']
    ].copy()

    # Merge on title and release year
    netflix_imdb = pd.merge(
        netflix_df,
        imdb_to_merge,
        left_on=['title', 'release_year'],
        right_on=['primaryTitle', 'startYear'],
        how='left'
    )

    print(f"IMDb Mapping complete. "
          f"{netflix_imdb['averageRating'].notna().sum()} titles matched.")
    return netflix_imdb


# -----------------------------------------------------------------------------
# 6. Main Data Pipeline
# -----------------------------------------------------------------------------

def main_preprocessing_pipeline():
    """
    Runs the complete data loading and preprocessing pipeline.

    Returns:
        A dictionary containing the key processed DataFrames:
        - 'netflix_2021': Cleaned 2021 dataset
        - 'netflix_2025': Cleaned 2025 dataset
        - 'netflix_imdb': Cleaned 2021 data merged with IMDb ratings
        - 'genre_matrix': One-hot encoded matrix for the 2021 dataset
        - 'unique_genres': List of all unique genres
    """
    print("--- Starting Main Data Pipeline ---")

    # Load all datasets
    df_2021, df_movies_2025, df_tv_2025 = load_netflix_datasets()
    if df_2021 is None:
        return None

    imdb_df = load_and_prep_imdb()

    # Clean Netflix datasets
    netflix_2021_clean = clean_2021_dataset(df_2021)
    netflix_2021_clean = engineer_netflix_features(netflix_2021_clean)

    netflix_2025_clean = clean_2025_dataset(df_movies_2025, df_tv_2025)
    # Note: 2025 data lacks 'date_added', so feature engineering is limited

    # Create genre matrix for the 2021 dataset (which has date_added)
    genre_matrix, unique_genres = create_genre_matrix(netflix_2021_clean)

    # Map IMDb data to the 2021 dataset
    if imdb_df is not None:
        netflix_imdb = map_imdb_to_netflix(netflix_2021_clean, imdb_df)
    else:
        netflix_imdb = netflix_2021_clean  # Return base data if IMDb fails
        netflix_imdb['averageRating'] = pd.NA
        netflix_imdb['numVotes'] = pd.NA

    print("--- Data Pipeline Finished Successfully ---")

    return {
        'netflix_2021': netflix_2021_clean,
        'netflix_2025': netflix_2025_clean,
        'netflix_imdb': netflix_imdb,
        'genre_matrix': genre_matrix,
        'unique_genres': unique_genres
    }


def main():
    """
    Main execution function when script is run directly.
    Runs the full pipeline and prints the shapes of the output DataFrames.
    """
    print("Running data_preprocessing.py as main script...")
    dataframes = main_preprocessing_pipeline()

    if dataframes:
        print("\nDataFrames processed:")
        print(f"Netflix 2021 shape: {dataframes['netflix_2021'].shape}")
        print(f"Netflix 2025 shape: {dataframes['netflix_2025'].shape}")
        print(f"Netflix+IMDb shape: {dataframes['netflix_imdb'].shape}")
        print(f"Genre Matrix shape: {dataframes['genre_matrix'].shape}")


if __name__ == '__main__':
    main()